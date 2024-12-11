import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from Gridattentionblock import *

# resiual block + attention block
class combined_cnn(nn.Module): 
    def __init__(self, in_ch = 1, num_classes=3, groups=1, width_per_group=64, aggregation_mode='ft', norm_layer=None): 
        super(combined_cnn, self).__init__() # super : 상위 클래스의 메소드 호출 (nn.Module의 메소드 호출)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # get residual blocks from torchvision.models
        rblocks = models.resnet18(pretrained=False)

        self._norm_layer = norm_layer  # default norm_layers are batch_norm
        self.inplanes = 64
        self.dilation = 1

        self.groups = groups # number of skip connections in one block
        self.base_width = width_per_group

        # first layer of model
        self.conv1 = nn.Conv2d(in_ch,64,7,2,3,bias=False)
        self.bn1 = rblocks.bn1
        self.relu = rblocks.relu
        self.maxpool = rblocks.maxpool

        # residual blocks
        self.layer1 = rblocks.layer1
        self.layer2 = rblocks.layer2
        self.layer3 = rblocks.layer3
        self.layer4 = rblocks.layer4
        self.fc = nn.Linear(512*1, num_classes)

        # defining comp scores using grid attention block
        filters = [128, 256, 512] # output channels of each residual blocks

        # get compatibility score from two output channels

        # compatibility score from residual block 1
        self.compatibility_score1 = GridAttentionBlock2D(in_channels=filters[0],
                                                         gating_channels=filters[2],
                                                         inter_channels=filters[2], dimension=2,
                                                         mode='concatenation_softmax', bn_layer=True)

        self.compatibility_score2 = GridAttentionBlock2D(in_channels=filters[1],
                                                         gating_channels=filters[2],
                                                         inter_channels=filters[2], dimension=2,
                                                         mode='concatenation_softmax', bn_layer=True)

        # self.attention_filter_sizes = [filters[0], filters[1]]
        # we are using just linear aggregation
        self.classifier1 = nn.Linear(filters[0], num_classes)
        self.classifier2 = nn.Linear(filters[1], num_classes)
        self.classifier3 = nn.Linear(filters[2], num_classes)
        self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

        if aggregation_mode == 'ft':
            self.classifier = nn.Linear(num_classes*3, num_classes)
            self.aggregate = self.aggregation_ft
        else:
            raise NotImplementedError

    # classifiers means make each feature maps as 3 output channels
    # classifier means last classifier
    def aggregation_sep(self, *attended_maps):
        return [cls(atts) for cls, atts in zip(self.classifiers, attended_maps)]
        # channels : [3,3,3]
    def aggregation_ft(self, *attended_maps):
        preds = self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1)) # dim=1 means the channel
        #
    def _forward_impl(self, x): # custom forward function
        batch_size = x.shape[0]

        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)

        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)

        g_att1, att1 = self.compatibility_score1(x6, x8)
        g_att2, att2 = self.compatibility_score2(x7, x8)

        g1 = F.adaptive_avg_pool2d(g_att1, (1,1)).view(batch_size, -1)
        g2 = F.adaptive_avg_pool2d(g_att2, (1,1)).view(batch_size, -1)
        x9 = F.adaptive_avg_pool2d(x8, (1,1)).view(batch_size, -1)

        return self.aggregate(g1,g2,x9)

    def forward(self, x):
        return self._forward_impl(x)

def _combined_model(in_ch = 1, num_classes = 3,  **kwargs):
    model = combined_cnn(in_ch, num_classes,  **kwargs)
    return model


