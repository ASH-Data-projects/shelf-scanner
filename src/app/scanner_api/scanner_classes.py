import pandas as pd
from ultralytics.engine.results import Results
from dataclasses import dataclass, field

@dataclass
class ScannerResult:
    """
    dataclass containing the information that is returned by the scanner
    """
    yolo_result: Results = field(repr=False)
    order_result: pd.DataFrame
    
    def __post_init__(self):
        pass
    
    def product_count(self):
        raise NotImplementedError("method 'product_count' not implemented yet.")
    