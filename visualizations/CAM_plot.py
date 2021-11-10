import os
import sys
# get parent dir
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('USskintumor'))))

import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchsummary import summary

from loss_functions.focal_loss import *
import matplotlib.pyplot as plt
from matplotlib import lines
import numpy
import cv2
import copy
import math
import random

from model.combined_CNN_for_CAM import conv3x3, combined_cnn, _combined_model

## CUDA
if torch.cuda.is_available() == True:
    device = torch.device('cuda')
else:
    raise Exception('cuda is not available')

## get trained model
def combined_net(**kwargs):
    return _combined_model(transfer_learning=True, num_classes = 3,  **kwargs)

model = combined_net().cuda()
date, fold = '0603', 1
model_path = '../final_saved/' + date + '/focal_loss/' + str(fold) + '.pth'
model.load_state_dict(torch.load(model_path))

# evaluation mode
model.eval()

## dataset & transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
val_dataset = torchvision.datasets.ImageFolder('../kfold/' + str(fold) + '/val', transform = transform)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = 0,
                                              pin_memory = True)
## test
fw = model.classifier.weight.cpu().detach().numpy()
fw = fw.transpose()

fw_weights = {0:fw[0:3], 1:fw[3:6], 2:fw[6:9]}
weights = {0:model.classifier1.weight.cpu().detach().numpy(),
           1:model.classifier2.weight.cpu().detach().numpy(),
           2:model.classifier3.weight.cpu().detach().numpy()}

fnumbers = {0:128, 1:256, 2:512}

for inputs, labels in val_data_loader:
    output, g_att1, g_att2, x8 = model(inputs.cuda())   
    fmaps = {0:g_att1.cpu().detach().numpy(),
             1:g_att2.cpu().detach().numpy(),
             2:x8.cpu().detach().numpy()}


    pred = torch.softmax(output, dim = 1)
    pred_id = torch.argmax(pred).item()
    

    for a in range(3):
        fn,w,fm, fw = fnumbers[a], weights[a], fmaps[a], fw_weights[a].transpose()
        for i in range(fn):
            # prediction : pred_id
            if i == 0: c_cam = w[pred_id][i] * fm[0][i]
            else: c_cam += w[pred_id][i] * fm[0][i]
            
        c_cam * np.mean(fw[pred_id])
        c_cam = cv2.resize(c_cam, (224,224), interpolation = cv2.INTER_CUBIC)
        if a == 0 : whole_cam = c_cam
        else: whole_cam += c_cam
    
    if pred_id == 1:
        whole_cam = whole_cam.max() - whole_cam        
    image = inputs.numpy().squeeze()
    plt.imshow(image, cmap = 'gray')
    plt.imshow(whole_cam, alpha = 0.4)
    plt.axis('off')
    plt.title('label : {} / prediction : {}'.format(labels.item(), pred_id))
    plt.show()
    break
