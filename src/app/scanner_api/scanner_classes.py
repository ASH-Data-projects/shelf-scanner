import pandas as pd
from PIL import Image, ImageDraw
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
    order_result: 'OrderResult'
    
    def __post_init__(self):
        pass
    
    def product_count(self):
        raise NotImplementedError("method 'product_count' not implemented yet.")

    def highlighted_boxes(self):
        
        # we load the image to RGBA mode to allow the overlap of other figures.
        image = Image.fromarray(self.yolo_result.orig_img[:, :, ::-1])
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        # overlay is the layer that the rectangles will be drawn on.
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for i, detection in self.order_result.detection_df.iterrows():
            self._draw_box(draw, detection)
        
        img_w_boxes = Image.alpha_composite(image, overlay)
        
        return img_w_boxes

    def _draw_box(self, draw:ImageDraw.ImageDraw, data:pd.Series):
        if self._is_ordered(data):
            color = (0, 255, 0, 64)
            outline = 'green'
        else:
            color = (255, 0, 0, 64)
            outline = 'red'
        
        x0 = data.x - data.h/2
        y0 = data.y - data.w/2
        x1 = data.x + data.h/2
        y1 = data.y + data.w/2
        
        draw.rectangle(
            [x0, y0, x1, y1],
            fill=color,
            outline=outline,
            width=3
        )
        return draw

    def _is_ordered(self, detection:pd.Series):
        if any(self.order_result.comparison_df[self.order_result.comparison_df.pos == detection.pos].detected):
            return True
        else:
            return False



@dataclass
class OrderResult: 
    comparison_df:pd.DataFrame
    detection_df:pd.DataFrame
    