from ultralytics import YOLO
from roboflow import Roboflow

import torch

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = YOLO('yolov8n.pt').to(device)
    
    # supplementary code from official yolov8 docs: https://docs.ultralytics.com/usage/python/#train
    # results = model.train(data='coco8.yaml', epochs=3)
    # results = model.val()
    # results = model('https://ultralytics.com/images/bus.jpg')
    # success = model.export(format='onnx')

if __name__ == '__main__':
    main() 