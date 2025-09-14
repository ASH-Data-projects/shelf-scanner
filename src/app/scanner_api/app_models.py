import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from ultralytics.engine.results import Results
from ultralytics import YOLO
from typing import Any
from .scanner_classes import ScannerResult

class OrderModel:
    def __init__(self, shelf_csv):
        df = pd.read_csv(
            shelf_csv,
            sep=' ',
            names=['cls','x','y','w','h']
        )
        df.sort_values('y',inplace=True, ignore_index=True)
        df['pos'] = range(df.shape[0])
        self.base_shelf = df[['x', 'y', 'w', 'h', 'cls', 'pos']]
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
        self.position_finder.fit(df[['x','y']], df.pos)
    
    def predict(self, input:Results) -> pd.DataFrame:
        x,y,h,w = input.boxes.xywh.T
        cls = input.boxes.cls
        data = {'x': x, 'y': y, 'w': w, 'h': h, 'cls': cls}
        detection = pd.DataFrame(data)
            
        detection = self._preprocess_boxes_coordinates(detection)
        detection['pos'] = self.position_finder.predict(detection[['x','y']])
        detection['SKU'] = detection['cls'].map(input.names)
        detection.sort_values('pos', ignore_index=True, inplace=True)
        
        def check_item(row):
            row = row[['cls','pos']]
            return (detection[['cls','pos']] == row).all(axis=1).any()
        
        comparison_df = self.base_shelf[['pos', 'cls']].copy()
        comparison_df['SKU'] = comparison_df['cls'].map(input.names)
        comparison_df['matches'] = comparison_df.apply(check_item, axis=1)
        
        return (comparison_df, detection)

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
        
def main():
    pass

if __name__ == '__main__':
    main()