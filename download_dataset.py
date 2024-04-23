# Running this Python file downloads the dataset from Roboflow to your current directory.

# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY") # api key is shared within team
project = rf.workspace("furkan-bakkal").project("shopping-cart-1r48s")
version = project.version(1)
dataset = version.download("yolov8")
