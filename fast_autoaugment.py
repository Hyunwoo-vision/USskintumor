##
import sys
import os

import torch
import torchvision

import copy
import time
import random

from utils.transforms import * #autment methods

# hyperopt for bayesian optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from model.combined_CNN import conv3x3, combined_cnn, _combined_model
## CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise Exception('cuda is not available')

##  transform candidates

transform_candidates = [ShearXY, TranslateXY,
                        AutoContrast, Invert,
                        Flip, Equalize,
                        Solarize, Posterize,
                        Contrast, Brightness,
                        Speckle_noise, Gaussian_noise,
                        Sharpness, Cutout]
## select top n subpolices
def get_topn_subpolicies(subpolicies, N=5):
    return sorted(subpolicies, key=lambda subpolicy : subpolicy[1])[:N]

## get data loader
def get_dataloader(dataset, shuffle=False, pin_memory=True):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 2,
                                              shuffle = False, num_workers = 4,
                                              pin_memory = pin_memory)
    return data_loader
## validate
def validate_child(model, dataset, dataset_original, transform1, transform2, device=device):
    criterion = nn.CrossEntropyLoss()
    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    '''apply augmentation'''
    dataset.transform = transform1
    dataset_original.transform = transform2

    augmented_dataset = torch.utils.data.ConcatDataset([dataset, dataset_original])
    data_loader = get_dataloader(augmented_dataset, shuffle=False, pin_memory=False)

    '''validation'''
    model.eval()
    with torch.no_grad():
        whole_loss = 0.0
        for images, targets in data_loader:
            images = images.to(device); targets = targets.to(device)
            # forward
            output = model(images)
            output = torch.softmax(output, 1)

            loss = criterion(output, targets.long())
            loss = loss*images.shape[0]

            whole_loss += loss

        loss = whole_loss/(len(data_loader.dataset.datasets[0].targets)+len(data_loader.dataset.datasets[1].targets))

    return loss

## main
'''whole dataset : 5 folds
   trainset of each fold : 4 folds'''

for k in range(1,6):
    print('{}th fold proceeding...'.format(k))
    print('-'*50)
    global iteration
    iteration = k

    for kk in range(1,5):
        data_dir = './kfold_fastaug/' + str(k) + '/' + str(kk) + '/val'
        dataset = torchvision.datasets.ImageFolder(data_dir, transform=None)
        dataset_original = torchvision.datasets.ImageFolder(data_dir, transform=None)

        # using hyperopt
        def _objective(sampled):
            global iteration
            def combined_net(**kwargs):
                return _combined_model(transfer_learning=True, num_classes=3, **kwargs)
            child_model = combined_net()
            # get original pretrained model with celoss
            model_path = './original_trained_models/celoss/' + str(iteration) + '/' + str(kk) + '.pth'
            child_model.load_state_dict(torch.load(model_path))

            # define subpolicy using hyperopt
            subpolicy = [transform(prob, mag) for transform, prob, mag in sampled]

            transform_subpolicy =  torchvision.transforms.Compose([
                *subpolicy,
                torchvision.transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            trainsform_original = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            val_res = validate_child(child_model, dataset, dataset_original,transform_subpolicy, trainsform_original,
                                     device)
            loss = val_res.cpu().numpy()

            return {'loss':loss, 'status':STATUS_OK}

        space = [(hp.choice('transform1', transform_candidates), hp.uniform('prob1', 0, 1.0), hp.uniform('mag1', 0, 1.0)),
                 (hp.choice('transform2', transform_candidates), hp.uniform('prob2', 0, 1.0), hp.uniform('mag2', 0, 1.0))]

        trials = Trials()
        best = fmin(_objective,
                    space = space,
                    algo = tpe.suggest,
                    max_evals = 100,
                    trials = trials)

        best_subpolicy = [transform_candidates[best['transform1']](best['prob1'], best['mag1']),
                          transform_candidates[best['transform2']](best['prob2'], best['mag2'])]

        subpolicies = []
        for t in trials.trials:
            vals = t['misc']['vals']
            subpolicy = [transform_candidates[vals['transform1'][0]](vals['prob1'][0], vals['mag1'][0]),
                         transform_candidates[vals['transform2'][0]](vals['prob2'][0], vals['mag2'][0])]
            subpolicies.append((subpolicy, t['result']['loss']))

        top5_subpolicies = get_topn_subpolicies(subpolicies, N=5)
        print(top5_subpolicies)
        print('-'*50)

        if kk == 1: policies = top5_subpolicies
        else: policies.extend(top5_subpolicies)

    np.save('./subpolicies/celoss/' + str(iteration) + 'th_fold.npy', policies)



