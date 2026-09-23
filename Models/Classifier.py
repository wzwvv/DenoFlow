"""Auxiliary classifier used to supervise the integrated output.

The backbone is SSVEPNet: a spatial convolution with a max-norm constraint, a strided
temporal convolution, a Bi-LSTM and two dense layers. Spectral normalisation bounds the
gradient the class term sends back into the field network. It is used during training only;
denoising at test time needs the field network alone.
"""
import torch
from torch import nn
import math
from torch.nn.utils import spectral_norm
def Spectral_Normalization(m):
    for name, layer in m.named_children():
        m.add_module(name, Spectral_Normalization(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, X):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(X)

class LSTM(nn.Module):
    '''
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1,batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2 ,1)  # (b, c, T) -> (b, T, c)
        r_out, _ = self.rnn(x)  # r_out 
        return r_out


class ESNet(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride),padding=(0, kernel_size//2-1)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        net = []
        net.append(self.spatial_block(num_channels, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                           self.K, self.S))

        self.conv_layers = nn.Sequential(*net)

        self.fcSize = self.calculateOutSize(self.conv_layers, num_channels, T)
        self.fcUnit = self.fcSize[0] * self.fcSize[1] * self.fcSize[2] * 2
        self.D1 = self.fcUnit // 10
        self.D2 = self.D1 // 5

        self.rnn = LSTM(input_size=self.F[1], hidden_size=self.F[1])

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU(),
            nn.Linear(self.D1, self.D2),
            nn.PReLU(),
            nn.Dropout(self.dropout_level))

        self.head1 = nn.Sequential(
                    nn.Linear(self.D2, num_classes),
                    nn.Softmax(dim=1)
                )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.squeeze(2)
        r_out = self.rnn(out)
        features = self.dense_layers(r_out)
        # The class head ends in a softmax, so the logits behind it are not otherwise
        # reachable. Any correct cross-entropy needs them: logging the softmax output instead
        # gives a gradient that scales as 1/p and blows up once the classifier is confidently
        # wrong. Callers that want a loss read cls_logits; 'cls' is unchanged for callers that
        # want a probability.
        cls_logits = self.head1[0](features)
        return {'cls': self.head1[1](cls_logits), 'cls_logits': cls_logits}
