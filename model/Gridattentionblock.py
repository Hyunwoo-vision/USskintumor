##
import torch
import torch.nn as nn
from torch.nn import functional as F
from weight_inits import init_weights
##
class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2,
                 mode = 'concatenation_softmax', bn_layer = True):
        super(_GridAttentionBlockND, self).__init__()
        assert dimension == 2
        assert mode in ['concatenation', 'concatenation_softmax',
                        'concatenation_sigmoid', 'concatenation_mean',
                        'concatenation_range_normalise', 'concatenation_mean_flow']

        self.mode = 'concatenation_softmax'
        self.dimension = dimension

        # Number of channels (pixel dimensions)
        # this channels are : [                             ]
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None: # To not chaning the output channels
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        bn = nn.BatchNorm2d
        self.upsample_mode = 'bilinear'

        # initialize id functions
        # W_f * x_ij + W_g * g_ij + bias (compatibility score)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.gating_channels, self.inter_channels,  kernel_size=1, stride=1, padding=0, bias=False)
        # In paper, conv layer for gating signal has bias value but it did not applied
        self.psi = nn.Conv2d(self.inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        # concatenation modes
        if not 'concatenation' in mode:
            raise NotImplementedError('Unknown operation function.')
        else:
            self.operation_function = self._concatenation

        if self.mode == 'concatenation_softmax':
            nn.init.constant(self.psi.bias.data, 10.0) # initialize the bias for psi

        # Initialize weights
        # Initialize with kaiming normal
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x, g):
        output = self.operation_function(x,g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # get wf and wg
        Wf_x = self.theta(x)
        Wf_x_size = Wf_x.size()
        # upsampling for wg to be added with wf
        Wg_g = F.upsample(self.phi(g), size=Wf_x_size[2:], mode=self.upsample_mode)

        comp_score = Wf_x + Wg_g
        comp_score = self.relu(comp_score)

        psi_comp = self.psi(comp_score)

        # concate mode == concatenation_softmax
        if self.mode == 'concatenation_softmax': # applying softmax for all batches
            sigm_psi_comp = F.softmax(psi_comp.view(batch_size, 1, -1), dim=2)
            sigm_psi_comp = F.sigmoid(sigm_psi_comp.view(batch_size,1,*Wf_x_size[2:]))
            # normalization 0~1
        else:
            raise NotImplementedError

        # upsampling for index multiplying with feature map
        sigm_psi_comp = F.upsample(sigm_psi_comp, size = input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_comp.expand_as(x) * x

        return y, sigm_psi_comp # attended feature map and attention map

##
class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2,
                 mode = 'concatenation_softmax', bn_layer = True):

        super(GridAttentionBlock2D, self).__init__(
            in_channels = in_channels, gating_channels=gating_channels,
            inter_channels=inter_channels, dimension=2,
                 mode = mode, bn_layer = True)



















