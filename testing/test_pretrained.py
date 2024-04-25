import os
import torch
import cv2
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

model_path = '../datasets/shopping-trolley-5/runs/detect/pretrainedv8n/weights/bestpretrainedv8n.pt'
image_path = '../shopping-trolley-5/test/images'
label_path = '../shopping-trolley-5/test/labels'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = YOLO(model_path, verbose = False).to(device)
# model.eval()

def main():
    predictions = predict_images(image_path)
    # labels = load_labels(label_path)
    print(predictions[0]) 
    print()
    print(predictions[-1])  
    print(len(predictions))

def predict_images(image_dir):
    predictions = []
    for filename in os.listdir(image_dir):
        print(filename)
        if filename.endswith('.jpg'):
            path = os.path.join(image_dir, filename)
            image = cv2.imread(path)
            with torch.no_grad():
                detections = model(image)
                predictions.append(detections[0].boxes.xyxy)
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

# def bboxvisualization(img_tensor, edgecolor=None, labels_ground=None, labels_xyxy=None, transforms=None):
#     # Available to plot twice, just call twice on sequence
#     if labels_ground != None: # Also means there is a transform performed
#         _, cx, cy, w, h = labels_ground
#         transformed_size = img_tensor.shape[-2:] # the resized shape
#         x1 = (cx - w / 2) * transformed_size[0]
#         y1 = (cy - h / 2) * transformed_size[1]
#         x2 = (cx + w / 2) * transformed_size[0]
#         y2 = (cy + h / 2) * transformed_size[1]
#     else:
#         x1, y1, x2, y2 = labels_xyxy

#     fig, ax = plt.subplots(1)
#     ax.imshow(img_tensor.permute(1, 2, 0))
#     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=edgecolor if edgecolor else 'r', facecolor='none')
#     ax.add_patch(rect)
#     plt.tight_layout()
#     plt.axis("off")
#     plt.show()

if __name__ == '__main__':
    main()