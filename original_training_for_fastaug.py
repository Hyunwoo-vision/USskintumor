## pakages
import sys
import time
import os

import torch
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from loss_functions.focal_loss import FocalLoss_Ori
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy

from transforms import *

# model
from model.combined_CNN import conv3x3, combined_cnn, _combined_model
## CUDA
if torch.cuda.is_available():
    global device
    device = torch.device('cuda')
else:
    raise Exception('cuda is not available')

## load_dataset
def load_dataset(data_dir, transform, fold, iteration, batch_size = 64):
    train_dataset = torchvision.datasets.ImageFolder(data_dir + '/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(data_dir + '/val', transform=transform)


    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    num_workers = 4,
                                                    pin_memory = True)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle = False,
                                                  num_workers = 4,
                                                  pin_memory = True)
    data_loader = {'train': train_data_loader,
                   'val':val_data_loader}

    return data_loader

## training
def train_model(model, data_loader, criterion, optimizer, scheduler, fold, iteration,num_epochs = 30):
    global device
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epochs):
        print('{}th epoch training...'.format(epoch+1))
        print('-'*10)

        # 2 phases
        for phase in ['train', 'val']:
            if phase == 'train': model.train() # train mode
            else: model.eval() # evaluation mode

            running_loss, running_corrects = 0.0, 0 # whole loss and corrects for current epoch
            for inputs, labels in data_loader[phase]:
                # get a batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad() # initialize optimizer before forwarding

                with torch.set_grad_enabled(phase == 'train'): # use gradient only training
                    outputs = model(inputs) # forwarding
                    _, preds = torch.max(outputs, 1) # make max value as 1 (pred)
                    outputs = torch.softmax(outputs, 1) # probilities

                    # labels = labels.type(torch.cuda.LongTensor) # longtensors for focal loss
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward() # calculate the gradients
                        optimizer.step() # optimize

                running_loss += loss.item()*inputs.size(0) # batch size * loss
                running_corrects += torch.sum(preds == labels.data)

        # an epoch finished
            if phase == 'train': scheduler.step() # scheduler step for training

            epoch_loss = running_loss/data_loader[phase].dataset.__len__()
            epoch_acc = running_corrects.double()/data_loader[phase].dataset.__len__()

            # print the process
            print('{}_loss = {:.4f} , {}_accuracy = {:.4f}'.format(phase, epoch_loss, phase,
                                                                        epoch_acc))

            # update model with best acc value
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save model
                # if not os.path.exists('./original_trained_models/celoss/' + str(fold) + '/' + str(iteration) + '.pth'):
                #     os.makedirs('./original_trained_models/celoss/' + str(fold) + '/' + str(iteration) + '.pth')
                # torch.save(model.state_dict(),
                #            './original_trained_models/celoss/' + str(fold) + '/' + str(iteration) + '.pth')
        print('-' * 30)
        print()

    time_elapsed = time.time() - since
    print("{}th fold completed in {:0f}m {:0f}s".format(fold, time_elapsed//60, time_elapsed % 60))
    print('Best Val Acc : {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

## main
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) ])

batch_size = 16
# define model
def combined_net(**kwargs):
    return _combined_model(transfer_learning=True, num_classes = 3,  **kwargs)

# 5-fold cross validation
for fold in range(1,6):
    for iteration in range(1,5):

        data_dir = './kfold_fastaug/' + str(fold) + '/' + str(iteration)
        data_loader = load_dataset(data_dir, transform, fold,  iteration, batch_size)

        '''get model'''
        model = combined_net().cuda()
        # criterion = FocalLoss_Ori(num_class=3, alpha=0.25, gamma=3, balance_index=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma = 0.1)
        model_ft = train_model(model, data_loader, criterion, optimizer, scheduler, fold, iteration,num_epochs = 30)