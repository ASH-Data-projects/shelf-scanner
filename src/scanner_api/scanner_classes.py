import pandas as pd
from ultralytics.engine.results import Results
from dataclasses import dataclass, field

@dataclass
class ScannerResult:
    """
    dataclass to store the data computed by the Scanner model.
    Args:
        yolo_result (Results): Result of the YOLO model
        
        order_result (tuple[pd.DataFrame, pd.DataFrame]): Result of the 
        OrderModel model.

    """
    yolo_result: Results = field(repr=False)
    order_result: tuple[pd.DataFrame, pd.DataFrame]
    
    def __post_init__(self):
        pass
    
    def product_count(self):
        raise NotImplementedError("method 'product_count' not implemented yet.")
    