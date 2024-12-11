## pakages
import sys
import time
import os

import torch
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transform
from torchsummary import summary

from loss_functions.focal_loss import FocalLoss_Ori
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy

from utils.transforms import *

# model
from model.combined_CNN import conv3x3, combined_cnn, _combined_model

## CUDA
if torch.cuda.is_available():
    global device
    device = torch.device('cuda')
else:
    raise Exception('cuda is not available')

## load_dataset
def load_dataset(data_dir, original_transform, fast_aug_transforms, iteration, batch_size = 64):
    train_dataset_origin = torchvision.datasets.ImageFolder(data_dir + 'train/', transform = original_transform)
    val_dataset = torchvision.datasets.ImageFolder(data_dir + 'val/', transform = original_transform)

    '''fast auto augmentation 적용하기'''
    for i in range(len(fast_aug_transforms)):
        c_augment = fast_aug_transforms[i][0]
        c_augment_transform = transform.Compose([
            *c_augment, #get all augmentations
            transform.Grayscale(num_output_channels=1),
            transform.ToTensor(),
            transform.Normalize([0.5],[0.5])
        ])

        train_dataset_augment = torchvision.datasets.ImageFolder(data_dir+'train/', transform = c_augment_transform)
        if i == 0:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset_origin, train_dataset_augment])
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_augment])

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
## define training
def train_model(model, data_loader, criterion, optimizer, scheduler, iteration, date, num_epochs = 30):
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

                    labels = labels.type(torch.cuda.LongTensor) # longtensors for focal loss
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
                if not os.path.exists('./final_saved/' + date + '/with_focal_loss/'):
                    os.makedirs('./final_saved/' + date + '/with_focal_loss/')
                torch.save(model.state_dict(),
                           './final_saved/' + date + '/with_focal_loss/' + str(iteration) +'.pth' )
        print('-' * 30)
        print()

    time_elapsed = time.time() - since
    print("{}th fold completed in {:0f}m {:0f}s".format(iteration, time_elapsed//60, time_elapsed % 60))
    print('Best Val Acc : {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

## main
original_transform = transform.Compose([
    transform.Grayscale(num_output_channels=1),
    transform.ToTensor(),
    transform.Normalize([0.5], [0.5]) ])

batch_size = 16
date = '211109'
# define model
def combined_net(**kwargs):
    return _combined_model(in_ch = 1, num_classes = 3,  **kwargs)

# 5-fold cross validation
for iteration in range(1,6):

    '''fast autoaugment for this fold'''
    fast_aug_transforms = np.load('./subpolicies/celoss/' + str(iteration) + 'th_fold.npy'
                                  , allow_pickle=True).tolist()

    data_dir = './kfold/' + str(iteration) + '/'
    data_loader = load_dataset(data_dir, original_transform, fast_aug_transforms, iteration, batch_size)

    '''get model'''
    # model = combined_cnn(transfer_learning=True, num_classes=3,
    #                      groups=1, width_per_group=64, aggregation_mode='ft', norm_layer=None)
    model = combined_net().cuda()
    criterion = FocalLoss_Ori(num_class=3, alpha=0.25, gamma=3, balance_index=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma = 0.1)
    model_ft = train_model(model, data_loader, criterion, optimizer, scheduler, iteration, date, num_epochs=30)




##
for i, l in data_loader['train']:
    print(i.shape, l.shape)
    model = model.to(device)
    model.train()
    output = model(i.cuda())
    break


##
os.makedirs('./final_saved/211109/with_focal_loss/')