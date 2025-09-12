import pandas as pd
import numpy as np
from typing import List
from ultralytics.engine.results import Results
from dataclasses import dataclass, field

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
    order_result: np.ndarray = field(repr=False)
    df: ScannerFrame = field(init=False)
    names: dict = field(init=False)
    
    def __post_init__(self):
        self.df = ScannerFrame(self.yolo_result)
        self.df.pos = self.order_result
        self.names = self.yolo_result.names