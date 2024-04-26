import torch
import gdown
import argparse
import torch.nn as nn
import subprocess
from torchvision import  models, transforms
import zipfile
import wandb
import numpy as np
from torchvision import datasets, utils
import matplotlib.pyplot as plt


def myTransform():
    transform = transforms.Compose([ 
        transforms.ToTensor(),           
        # transforms.Normalize(            
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )

        # transforms.Normalize(            
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.5, 0.5, 0.5]
        # )

        transforms.Normalize(            
            mean=[0.2, 0.2, 0.2],
            std=[0.7, 0.7, 0.7]
        )
    ])
    return transform
def plot_class_distribution(dataset):
    # Get the class labels
    classes = dataset.classes

    # Count the number of samples for each class
    class_counts = {}
    for _, label in dataset.samples:
        class_name = classes[label]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.show()

Screw_train_path = "./Screw_test_w_class" + "/test_w_label"
Screw_train = datasets.ImageFolder(Screw_train_path,myTransform)
plot_class_distribution(Screw_train)