# is-that-A-Cart: Shopping-Cart Detection in Retail Environment

## Project Overview

Upon concluding their shopping experience, patrons exiting markets may leave store-owned shopping carts at various locations within the environment without heeding the potential consequences. For instance, carts scattered around retail environments may lead to traffic blockage, and carts on parking spaces may prevent vehicles from parking there. Currently, retail stores such as Target&copy; and Wegmans&copy; hire human resources to retrieve shopping carts and gather them at a specific location for recycling. However, our goal is to automate this process by incorporating computer vision technologies. For our study, we train and fine-tune the **YOLOv8** (You Only Look Once, version 8) network, a state-of-the-art computer vision model, to perform bounding box detection. In addition, we propose a framework with machine integration to streamline shopping trolley retrieval. 

### Project Report
- **Project Report Paper** is available for **is-that-A-Cart** Detection Project: [`ExDarkHierarchyNet-Report.pdf`](./is-that-A-Cart-Report.pdf)
- $\LaTeX$ source code could also be access [here](./Bib-CVPR2024/main.tex)

## Dataset Overview

Kornilov's dataset comprises a total of 6903 images collected from diverse retail environments to ensure a comprehensive representation of various scenarios involving shopping cart placement.

**Source (Download Link):** [TrolleysClassification Image Dataset](https://universe.roboflow.com/ds/SsGLfwfVbl?key=QvaXGDCsr6)

We trained the YOLOv8 model with the training set of Kornilov's dataset and the following parameters:
- **Batch Size:**
    The training was conducted with a batch size of 25 and a total of 5996 images in the training set. This batch size ensures that enough data is processed per update to capture significant patterns without straining the computational resources.
- **Epochs:**
    We trained the model for 100 epochs, allowing multiple iterations over the entire dataset to refine the model weights. This number of epochs was selected to balance between achieving thorough learning and preventing overfitting. This approach ensures the model performs well in real-world conditions.