'''Running the Model'''

###################
## Prerequisites ##
###################
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3' # should do this before importing torch modules!
import time
import json
import pickle
import random
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from easydict import EasyDict as edict
from timm.data import create_transform
# from materials import CheXpertDataSet, CheXpertTrainer, DenseNet121, EnsemAgg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score


######################
## Create a Dataset ##
######################
class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, nnClassCount, policy, transform = None):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []
        # policy = 'diff'

        with open(data_PATH, 'r') as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                npline = np.array(line)
                idx = [7, 10, 11, 13, 15]
                label = list(npline[idx])
                for i in range(nnClassCount):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == 'diff':
                                if i == 1 or i == 3 or i == 4:  # Atelectasis, Edema, Pleural Effusion
                                    label[i] = 1                    # U-Ones
                                elif i == 0 or i == 2:          # Cardiomegaly, Consolidation
                                    label[i] = 0                    # U-Zeroes
                            elif policy == 'ones':              # All U-Ones
                                label[i] = 1
                            else:
                                label[i] = 0                    # All U-Zeroes
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                image_names.append('/data2/yinuo/' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        '''Take the index of item and returns the image and its labels'''
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

img_type = '-small'

# Traindata_frt = pd.read_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type), on_bad_lines='skip') ###
# Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)
# Traindata_frt.to_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type), index = False) ###
# Traindata_lat = pd.read_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type), on_bad_lines='skip') ###
# Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)
# Traindata_lat.to_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type), index = False) ###

pathFileTrain_frt = './CheXpert-v1.0{0}/train_frt.csv'.format(img_type) ###
# pathFileTrain_lat = './CheXpert-v1.0{0}/train_lat.csv'.format(img_type) ###
pathFileValid_frt = './CheXpert-v1.0{0}/valid_frt.csv'.format(img_type)
# pathFileValid_lat = './CheXpert-v1.0{0}/valid_lat.csv'.format(img_type)
pathFileTest_frt = './CheXpert-v1.0{0}/test_frt.csv'.format(img_type)
# pathFileTest_lat = './CheXpert-v1.0{0}/test_lat.csv'.format(img_type)
# pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)


transform_train = create_transform(224, is_training=True)
transform_val = create_transform(224, is_training=False)


nnClassCount = 5
policy = 'diff'
datasetTrain_frt = CheXpertDataSet(pathFileTrain_frt, nnClassCount, policy, transform_train)
# datasetTrain_lat = CheXpertDataSet(pathFileTrain_lat, nnClassCount, cfg.policy, transformSequence)
datasetValid_frt = CheXpertDataSet(pathFileValid_frt, nnClassCount, policy, transform_val)
# datasetValid_lat = CheXpertDataSet(pathFileValid_lat, nnClassCount, cfg.policy, transformSequence)
datasetTest_frt = CheXpertDataSet(pathFileTest_frt, nnClassCount, policy, transform_val)
# datasetTest_lat = CheXpertDataSet(pathFileTest_lat, nnClassCount, cfg.policy, transformSequence)
# datasetTest_agg = CheXpertDataSet(pathFileTest_agg, nnClassCount, cfg.policy, transformSequence)