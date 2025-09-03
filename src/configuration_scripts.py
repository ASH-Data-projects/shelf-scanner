# from cv2 import imread, imwrite
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pathlib import Path
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pandas import DataFrame # we don't need the whole library in our case
from sys import stdout
from sklearn.model_selection import train_test_split
from typing import Iterable
from functools import partial
from shutil import copy2
from pandas_path import path
from typing import Optional, Callable
import yaml
import os

def split_yolo_img_set(
        input_path:Path|str,
        output_path:Path|str,
        accepted_extensions:tuple=('jpg','jpeg','png','bmp','gif','heic'),
        n_jobs:Optional[int]=None,
        **kwargs
    ):
    
    if isinstance(output_path, str):
        output_path = Path(output_path)
    paths_df, classes = load_yolo_dir(input_path, accepted_extensions)
    
    train, test = train_test_split(paths_df, **kwargs)
    
    train_dir = output_path/'train'
    test_dir = output_path/'validation'
    train_img = train_dir/'images'
    test_img = test_dir/'images'
    train_labels = train_dir/'labels'
    test_labels = test_dir/'labels'

    train_img.mkdir(exist_ok=True)
    test_img.mkdir(exist_ok=True)
    train_labels.mkdir(exist_ok=True)
    test_labels .mkdir(exist_ok=True)    
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    print('processing training images')
    move_image_to_train_dir = partial(copy_image, output_dir_path=train_dir/'images')
    multy_thread_apply(train.img_path, move_image_to_train_dir, n_jobs)
    
    print('processing validation images')
    move_image_to_test_dir = partial(copy_image, output_dir_path=test_dir/'images')
    multy_thread_apply(test.img_path, move_image_to_test_dir, n_jobs)

    print('processing training label files')
    move_label_to_train_dir = partial(copy_labels, output_dir_path=train_dir/'labels')
    multy_thread_apply(train.label_path, move_label_to_train_dir, n_jobs)
    
    print('processing validation label files')
    move_label_to_test_dir = partial(copy_labels, output_dir_path=test_dir/'labels')
    multy_thread_apply(test.label_path, move_label_to_test_dir, n_jobs)
    
    classes_txt = '\n'.join(classes)
    with open(train_dir/'classes', 'w', encoding='utf-8') as f:
        f.write(classes_txt)
        
    with open(test_dir/'classes', 'w', encoding='utf-8') as f:
        f.write(classes_txt)
    
def multy_thread_apply(
        target:Iterable[Path],
        func:Callable,
        n_jobs:int
    ):
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = executor.map(func, target)
        
        total_items = len(target)
        processed_items = 0
        failed_items = 0
        for result in results:
            processed_items += result
            failed_items -= result-1
            print(f'\r  {processed_items}/{total_items} items processed. '
                  f'({failed_items} failed)', end='')
            stdout.flush()
    print()

def copy_image(img_path:Path, output_dir_path:Path):
    try:
        filename = img_path.name
        new_filename = filename.replace(img_path.suffix, '.jpeg') # Image.save() uses the suffix to determine the file type.
        output_path = output_dir_path/new_filename
        if output_path.exists():
            return True
        
        if isinstance(img_path, zipfile.Path):
            with img_path.open('rb') as f:
                with Image.open(f) as img:
                    img.save(output_path)
                    return True
        else:
            with Image.open(img_path) as img:
                img.save(output_path)
                return True

    except UnidentifiedImageError as e:
        print(f'Error opening {filename}, this may be due to the file'
            'being corrupted, an unsupported format or an invalid path: {e}')
        return False

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return False
        
    except Exception as e:
        print(f"Unexpected error with {filename}: {e}")
        return False

def copy_labels(img_path:Path, output_dir_path:Path):
    try:
        filename = img_path.name
        output_path = output_dir_path/filename
        if output_path.exists():
            return True
        
        if isinstance(img_path, zipfile.Path):
            file_content_bytes = img_path.read_bytes()
            output_path.write_bytes(file_content_bytes)
            return True
        else:
            copy2(img_path, output_dir_path)
            return True
    except Exception as e:
        print(f'something went wrong: {e}')
        return False
            
def load_yolo_dir(input_path:Path|str, accepted_extensions:tuple=('jpg','jpeg','png','bmp','gif','heic')):
    """
    This function reads a directory with the YOLO structure given by Label Studio.
    The function accepts zip files containing a directory with the right structure too.
    """

    match input_path:
        case Path() if input_path.suffix == '.zip':
            input_path = zipfile.Path(input_path)
        case str() if input_path.endswith('.zip'):
            input_path = zipfile.Path(input_path)
        case zipfile.Path():
            pass
        case _ :
            input_path = Path(input_path)
    
    images_dir = input_path/'images'
    labels_dir = input_path/'labels'

    raw_img_paths = (path for path in images_dir.iterdir() if path.name.endswith(accepted_extensions))
    raw_df = DataFrame({"img_path":raw_img_paths})
    raw_df['label_path'] = raw_df.img_path.apply(lambda x: labels_dir/(x.stem + '.txt'))
    raw_df.sort_values('label_path', axis=0, ignore_index=True, inplace=True, key = lambda x:x.astype(str))
    
    with (input_path/'classes.txt').open('r', encoding='utf-8') as f:
        classes = f.read().strip().split('\n')
    return raw_df, classes

def create_data_yaml(data_path:Path|str, path_to_data_yaml:Path|str):
    _, classes = load_yolo_dir(data_path)
    number_of_classes = len(classes)

    # Create data dictionary
    cpd = Path(__file__).parent.parent
    labeled_path = cpd/'data'/'labeled'
    train_path = labeled_path/'train'/'images'
    val_path = labeled_path/'validation'/'images'
    data = {
        'path': labeled_path.as_posix(),
        'train': train_path.relative_to(labeled_path).as_posix(),
        'val': val_path.relative_to(labeled_path).as_posix(),
        'nc': number_of_classes,
        'names': classes
    }

    # Write data to YAML file
    with open(path_to_data_yaml, 'w') as f:
      yaml.dump(data, f, sort_keys=False)
    print(f'Created config file at {path_to_data_yaml}')
    print('\nFile contents:\n')
    print(yaml.dump(data))
    return

def main():
    pass
    
if __name__ == '__main__':        
    main()
