import os
import sys
sys.path.append('/home/sean/MSc/code')
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils.helpers as helpers
import utils.classes as classes
import utils.train_utils as train_utils
import random
import torchxrayvision as xrv
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import torch
import wandb
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
import seaborn as sns
import csv



dicom_dir_1 = 'data/MBOD_Datasets/Dataset-1'
dicom_dir_2 = 'data/MBOD_Datasets/Dataset-2'
metadata_1 = pd.read_excel("data/MBOD_Datasets/Dataset-1/FileDatabaseWithRadiology.xlsx")

metadata_2 = pd.read_excel("data/MBOD_Datasets/Dataset-2/Database_Training-2024.08.28.xlsx")

ILO_imgs = 'data/ilo-radiographs-dicom'

d1_cl = classes.DICOMDataset1_CL(dicom_dir=dicom_dir_1, metadata_df=metadata_1)
d1_cl._assign_labels()
d1_cl.set_target("Profusion", 512)  # Add this line, choose appropriate target and size

d2_cl = classes.DICOMDataset2_CL(dicom_dir=dicom_dir_2, metadata_df=metadata_2)
d2_cl._assign_labels()
d2_cl.set_target("Profusion", 512)  # Add this line, choose appropriate target and size

combined_dataset = ConcatDataset([d1_cl, d2_cl])
# Convert 'Profusion Label' to integer for both datasets
d1_cl.metadata_df['Profusion Label'] = d1_cl.metadata_df['Profusion Label'].astype(int)
d2_cl.metadata_df['Profusion Label'] = d2_cl.metadata_df['Profusion Label'].astype(int)

print(d2_cl.metadata_df['Profusion Label'].unique())