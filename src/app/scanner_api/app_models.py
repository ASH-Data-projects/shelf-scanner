import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from ultralytics.engine.results import Results
from ultralytics import YOLO
from typing import Any
from scanner_classes import ScannerResult

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