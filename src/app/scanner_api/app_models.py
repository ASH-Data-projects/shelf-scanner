import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from ultralytics.engine.results import Results
from typing import List, Any
from ultralytics import YOLO
from dataclasses import dataclass, field
from numpy import ndarray


class ScannerFrame(pd.DataFrame):
    _required_columns: List[str] = ['x', 'y', 'w', 'h', 'cls', 'pos']

    @property
    def _constructor(self):
        return ScannerFrame
    
    def __init__(self, *args, **kwargs):
        # Identifies if the input is a Results object 
        yolo_result=None
        if len(args)>0 and isinstance(args[0], Results):
            yolo_result = args[0]
        elif isinstance(kwargs.get('data'), Results):
            yolo_result = kwargs.get('data')

        # The initialization of the class changes if input is a Result object.
        if yolo_result:
            # self.names = yolo_result.names
            x,y,h,w = yolo_result.boxes.xywh.T
            cls = yolo_result.boxes.cls
            data = {
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cls': cls
            }
            super().__init__(data)
        
        else:
            super().__init__(*args, **kwargs)

        if 'pos' not in self.columns:
            self['pos'] = np.nan
        
        for col in self._required_columns:
            if col not in self.columns:
                raise ValueError(f"DataFrame is missing a required column: '{col}'")
                    
            if not pd.api.types.is_numeric_dtype(self[col]):
                raise TypeError(f"The '{col}' column must be of a numeric type.")

@dataclass
class ScannerResult:
    """
    dataclass containing the information that is returned by the scanner
    """
    yolo_result: Results = field(repr=False)
    order_result: ndarray = field(repr=False)
    df: ScannerFrame = field(init=False)
    names: dict = field(init=False)
    
    def __post_init__(self):
        self.df = ScannerFrame(self.yolo_result)
        self.df.pos = self.order_result
        self.names = self.yolo_result.names

class OrderModel:
    def __init__(self, shelf_csv=None):
        if shelf_csv:
            df = pd.read_csv(
                shelf_csv,
                sep=' ',
                names=['cls','x','y','w','h']
            )
            self.select_order(df)
    
    @staticmethod
    def _preprocess_boxes_coordinates(df: pd.DataFrame):
        df = df.copy()
        scaler = StandardScaler()
        df.x = scaler.fit_transform(df[['x']])
        df.y = scaler.fit_transform(df[['y']])
        return df
    
    def select_order(self, df:pd.DataFrame):
        df = self._preprocess_boxes_coordinates(df)
        self.position_finder = KNeighborsClassifier(n_neighbors=1)
        self.position_finder.fit(df[['x','y']], np.arange(df.shape[0]))
    
    def predict(self, input:pd.DataFrame|Results) -> pd.DataFrame:
        if isinstance(input, Results):        
            x,y,h,w = input.boxes.xywh.T
            cls = input.boxes.cls
            data = {'x': x, 'y': y, 'w': w, 'h': h, 'cls': cls}
            input = pd.DataFrame(data)
            
        df = self._preprocess_boxes_coordinates(input)
        position = self.position_finder.predict(df[['x','y']])
        
        return position
    

class Scanner:
    """
    A pipeline to connect the YOLO model and the ordering model.
    """
    def __init__(self, yolo_model: YOLO, order_model: OrderModel):
        self.yolo_model = yolo_model
        self.order_model = order_model

    def predict(self, image: Any):
        """
        Accepts any input that the YOLO model accepts
        """
        yolo_result = self.yolo_model.predict(image)[0]
        order_result = self.order_model.predict(yolo_result)
        
        pred = ScannerResult(yolo_result, order_result)
 
        return pred
        
def test():
    from pathlib import Path
    yolo_path = Path(__file__).parent.parent.parent.parent/'models'/'my_model.pt'
    shelf_path = Path(__file__).parent/'shelves'/'BATERIAS (1F) 0,36M.csv'
    img = Path(__file__).parent.parent.parent.parent/'data'/'test'/'test_img.jpg'
    
    yolo_model = YOLO(yolo_path)
    order_model = OrderModel(shelf_path)
    
    
    scanner = Scanner(yolo_model=yolo_model, order_model=order_model)
    
    print(scanner.predict(img))
    

if __name__ == '__main__':
    test()