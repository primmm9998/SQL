!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="qb5zteXrTNTRqVwRO5N0")
project = rf.workspace("shelf-product-detection").project("shelf-product-detection-5panm")
version = project.version(4)
dataset = version.download("yolov5")
