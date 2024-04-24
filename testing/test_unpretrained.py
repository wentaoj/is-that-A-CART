import os
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

