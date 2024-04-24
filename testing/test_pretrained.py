import os
import torch
import cv2
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

model_path = '../datasets/shopping-trolley-5/runs/detect/pretrainedv8n/weights/bestpretrainedv8n.pt'
image_path = '../shopping-trolley-5/test/images'
label_path = '../shopping-trolley-5/test/labels'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(model_path).to(device)
# model.eval()

def main():
    predictions = predict_images(image_path)
    # labels = load_labels(label_path)
    print(predictions)

def predict_images(image_dir):
    predictions = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            path = os.path.join(image_dir, filename)
            image = cv2.imread(path)
            with torch.no_grad():
                prediction = model(image)
                print(prediction[0].boxes.xyxy)
                predictions.append(prediction)
    return predictions
            
def load_labels(label_dir):
    labels = []
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            path = os.path.join(label_dir, filename)
            with open(path, 'r') as file:
                labels[filename[:-4]] = file.readlines()
    return labels

def compute_metrics(predictions, true_labels):
    return 0

if __name__ == '__main__':
    main()