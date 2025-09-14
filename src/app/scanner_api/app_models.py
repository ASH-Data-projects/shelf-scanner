import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from ultralytics.engine.results import Results
from ultralytics import YOLO
from typing import Any
from .scanner_classes import ScannerResult
from os import PathLike

class OrderModel:
    """
    This model aims to predict the position of an product within a shelf from
    the coordinates of it within an image.
    """
    
    def __init__(self, shelf_csv):
        self.base_shelf = self._load_base_shelf(shelf_csv)
        self.position_finder = self._get_position_finder(self.base_shelf)
    
    @staticmethod
    def _preprocess_boxes_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method expects a DataFrame that represents the boxes of the result
        of a YOLO model containing 'x' and 'y' columns and then applies a
        standard scaling to each one independently.
        """
        
        df = df.copy()
        scaler = StandardScaler()
        df.x = scaler.fit_transform(df[['x']])
        df.y = scaler.fit_transform(df[['y']])
        return df
    
    def _get_position_finder(self, base_shelf:pd.DataFrame) -> KNeighborsClassifier:
        """
        This method creates a position finder model that given the center of
        a scanned box from a YOLO model, returns the position it represents in
        the shelf.
        
        Args:
            base_shelf (DataFrame): A DataFrame containing the data that
                describes the normalized positions of the products in the base
                shelf.
            
        Returns:
            KNeighborsClassifier: The model that predicts the position within
                the shelf that a scanned product has.
        """
        
        position_finder = KNeighborsClassifier(n_neighbors=1)
        position_finder.fit(base_shelf[['x','y']], base_shelf.pos)
        return position_finder
    
    def _load_base_shelf(self, shelf_csv:PathLike) -> pd.DataFrame:
        """
        This method loads the data that describes the base shelf from a csv
        file and stores it in a pd.DataFrame object. 
         
        Args:
            shelf_csv (PathLike): A PathLike object pointing to a .csv file.
        """
        
        df = pd.read_csv(
            shelf_csv,
            sep=' ',
            names=['cls','x','y','w','h']
        )
        df.sort_values('y',inplace=True, ignore_index=True)
        df['pos'] = range(df.shape[0])
        df = self._preprocess_boxes_coordinates(df)
        return df[['x', 'y', 'w', 'h', 'cls', 'pos']]
    
    def predict(self, input:Results) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prediction method of the OrderModel object.
        
        Args:
            input (Results): Results object obtained from the yolo model.
            
        Returns:
            tuple[DataFrame, DataFrame]: The first DataFrame contains the
            expected SKUs in the image that is being scanned. The second
            DataFrame has a similar structure but also contains information
            related to the scanned image (like xy coordinates of each product
            and the dimensions of the box containing it). In this DataFrame the
            `pos` column is calculated by the `OrderModel` model using the x, y
            values and the `detected_SKU` column stores SKU of the equivalent
            class detected by the `YOLO` model. 
        """
        
        x,y,h,w = input.boxes.xywh.T
        cls = input.boxes.cls
        data = {'x': x, 'y': y, 'w': w, 'h': h, 'cls': cls}
        detection_df = pd.DataFrame(data)
            
        norm_coor = self._preprocess_boxes_coordinates(detection_df)
        detection_df['pos'] = self.position_finder.predict(norm_coor[['x','y']])
        detection_df['detected_SKU'] = detection_df['cls'].map(input.names)
        detection_df.sort_values('pos', ignore_index=True, inplace=True)
        
        def check_item(row):
            row = row[['cls','pos']]
            return (detection_df[['cls','pos']] == row).all(axis=1).any()
        
        comparison_df = self.base_shelf[['pos', 'cls']].copy()
        comparison_df['expected_SKU'] = comparison_df['cls'].map(input.names)
        comparison_df['detected'] = comparison_df.apply(check_item, axis=1)
        return (comparison_df, detection_df)

class Scanner:
    """
    A pipeline to connect the YOLO model and the ordering model.
    """
    
    def __init__(self, yolo_model: YOLO, order_model: OrderModel):
        self.yolo_model = yolo_model
        self.order_model = order_model

    def predict(self, image: Any):
        """
        Performs a complete object detection and analysis pipeline.

        This method  first runs the input image through a YOLO model for object
        detection. The raw results from the YOLO model are then passed to a
        custom order model for a second-stage analysis. The combined output is 
        encapsulated in a single `ScannerResult` object.

        Args:
            image (Any): The input image to be processed by the models. It can
            be any input that the YOLO model accepts

        Returns:
            ScannerResult: A complete object containing the results from both
                the YOLO model and the custom order model.
        """

        yolo_result = self.yolo_model.predict(image)[0]
        order_result = self.order_model.predict(yolo_result)
        pred = ScannerResult(yolo_result, order_result)
        return pred
        
def main():
    pass

if __name__ == '__main__':
    main()