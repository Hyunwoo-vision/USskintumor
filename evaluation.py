##
import sys
import os

import torch
import torchvision
import torchvision.transforms as transforms

from loss_functions.focal_loss import FocalLoss_Ori
import matplotlib.pyplot as plt
from matplotlib import lines
import numpy as np
import cv2
import copy
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import sklearn.metrics as metrics

## CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise Exception('cuda is not available')

## model
from model.combined_CNN import conv3x3, combined_cnn, _combined_model
def combined_net(**kwargs):
    return _combined_model(transfer_learning=True, num_classes=3,  **kwargs)


## transform
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
## evaluation def
def evaluation(model, data_loader, criterion, device):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    pred, label = [], []
    y_score = torch.Tensor([])
    probs = [[]]
    since = time.time()

    for batch, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if labels.item() == 0: b_label = np.array([[1, 0, 0]])
        elif labels.item() == 1: b_label = np.array([[0, 1, 0]])
        else: b_label = np.array([[0, 0, 1]])

        output = model(inputs)
        output = torch.softmax(output, 1)
        _, preds = torch.max(output, 1)

        lab = torch.Tensor.cpu(labels)
        lab = lab.numpy()

        if batch == 0: one_hot = b_label
        else: one_hot = np.concatenate((one_hot, b_label), axis=0)

        label = np.concatenate((label, lab), axis=0)
        labelss = labels.type(torch.cuda.FloatTensor)

        b = labels.cpu().numpy()
        prob = torch.softmax(output, -1)
        prob = torch.Tensor.cpu(prob)
        prob = prob.detach().numpy()

        if batch == 0: probs = prob
        else: probs = np.concatenate((probs, prob), axis=0)

        predict = torch.Tensor.cpu(preds)
        predict = predict.detach().numpy()

        pred = np.concatenate((pred, predict), axis=0)
        loss = criterion(output, labels.long())

        running_loss += loss.item() * inputs.size(0)
        if preds.item() == int(labels.item()):
            running_corrects += 1

    test_loss = running_loss / len(data_loader.dataset.targets)
    test_acc = running_corrects / len(data_loader.dataset.targets)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', test_loss, test_acc))
    time_e = time.time() - since
    print('evaluation complete in {:.0f}m {:.0f}s'.format(
        time_e // 60, time_e % 60))
    print('-' * 50)

    ############### confusion matrix
    for i in range(3):
        labels = np.zeros(label.shape, dtype='float64')
        preds = np.zeros(label.shape, dtype='float64')

        if i == 0:
            for j in range(len(labels)):
                if not (label[j] == 0.0): labels[j] = 0.0
                else: labels[j] = 1.0

                if not (pred[j] == 0.0): preds[j] = 0.0
                else: preds[j] = 1.0

        if i == 1:
            for j in range(len(labels)):
                if not (label[j] == 1.0): labels[j] = 0.0
                else: labels[j] = 1.0
                if not (pred[j] == 1.0): preds[j] = 0.0
                else: preds[j] = 1.0

        if i == 2:
            for j in range(len(labels)):
                if not (label[j] == 2.0): labels[j] = 0.0
                else: labels[j] = 1.0

                if not (pred[j] == 2.0): preds[j] = 0.0
                else: preds[j] = 1.0

        mat = confusion_matrix(labels, preds, labels=[0,1])
        print(mat)
        TN,FP,FN,TP = mat[0,0], mat[0,1], mat[1,0], mat[1,1]
        print('TP:{} FN:{} FP:{} TN:{}'.format(TP,FN,FP,TN))

        sensitivity = (TP / (TP + FN) * 100)
        specificity = (TN / (FP + TN) * 100)
        precision = (TP / (TP + FP) * 100)
        accuracy = ((TP + TN) / (TP + TN + FP + FN) * 100)
        f1 = (2 * precision * sensitivity) / (precision + sensitivity)

        print('sensitivity:{:.4f} specificity:{:.4f} precision:{:.4f} accuracy:{:.4f} f1_score:{:4f}'.format(sensitivity,
                                                                                                            specificity,
                                                                                                            precision,
                                                                                                            accuracy,
                                                                                                            f1))
    auc = roc_auc_score(one_hot, probs, multi_class = 'ovr')
    print('auc:{:.4f}'.format(roc_auc_score(one_hot,probs)))

    label_1, label_2, label_3 = np.zeros(label.shape), np.zeros(label.shape), np.zeros(label.shape)
    probs_1, probs_2, probs_3 = np.zeros(probs.shape[0]), np.zeros(probs.shape[0]), np.zeros(probs.shape[0])

    for i in range(label.shape[0]):
        probs_1[i] = probs[i][1] + probs[i][2]
        probs_2[i] = probs[i][0] + probs[i][2]
        probs_3[i] = probs[i][0] + probs[i][1]
        if label[i] == 0:
            label_1[i] = 0
            label_2[i] = 1
            label_3[i] = 1
        elif label[i] == 1:
            label_1[i] = 1
            label_2[i] = 0
            label_3[i] = 1
        else:
            label_1[i] = 1
            label_2[i] = 1
            label_3[i] = 0

    fpr1, tpr1, thresholds1 = roc_curve(label_1, probs_1)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    fpr2, tpr2, thresholds2 = roc_curve(label_2, probs_2)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    fpr3, tpr3, thresholds3 = roc_curve(label_3, probs_3)
    roc_auc3 = metrics.auc(fpr3, tpr3)

    print('AUC of epi: {}'.format(roc_auc1))
    print('AUC of lip: {}'.format(roc_auc2))
    print('AUC of pil: {}'.format(roc_auc3))

    # plot_roc_curve(fpr,tpr)
    # plt.title('{}th fold'.format(int(iteration)))

    plt.plot(fpr1, tpr1, 'b', linewidth=0.8, label='Epidermal cyst AUCROC : %0.4f' % roc_auc1)
    plt.plot(fpr2, tpr2, 'g', linewidth=0.8, label='Lipoma AUCROC : %0.4f' % roc_auc2)
    plt.plot(fpr3, tpr3, 'y', linewidth=0.8, label='Pilomatricoma AUCROC : %0.4f' % roc_auc3)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    #     plt.grid('--')
    # plt.savefig('{}th fold ROC'.format(iteration) + '.svg', format='svg', bbox_inches='tight')
    plt.show()

##
criterion = FocalLoss_Ori(num_class=3, alpha=0.25, gamma = 3, balance_index=2)
for fold in range(1,6):
    data_dir = './kfold/' + str(fold) + '/'
    val_dataset = torchvision.datasets.ImageFolder(data_dir + 'val/', transform=transform)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    date = '0603'
    model = combined_net().cuda()
    model_path = './final_saved/' + date + '/focal_loss/' + str(fold) + '.pth'
    model.load_state_dict(torch.load(model_path))
    evaluation(model, val_data_loader, criterion, device)



