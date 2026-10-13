import math
import torch
from einops import rearrange
from torch import nn, Tensor,einsum
from einops.layers.torch import Rearrange
from torch.autograd import Function
from torch.nn import functional as F
class FeatureLearner(nn.Module):
    def __init__(self,block_num=5):
        super().__init__()
        block = nn.Sequential(
                nn.Conv1d(1, 32, 3, 1, 1), nn.GroupNorm(32,32), nn.LeakyReLU(0.2),
                nn.Conv1d(32, 32, 3, 1, 1), nn.GroupNorm(32,32), nn.LeakyReLU(0.2),
                nn.AvgPool1d(2, stride=2),
            )
        block_list =  [block]
        in_channels = 32
        for i in range(block_num):
            block = nn.Sequential(
                Rearrange('N C L -> N L C'),
                nn.TransformerEncoderLayer(in_channels,8,dim_feedforward=in_channels,batch_first=True),
                Rearrange('N L C -> N C L'),
                nn.Conv1d(in_channels, 2 * in_channels, 3, 2, 1), nn.GroupNorm(32,2*in_channels), nn.LeakyReLU(0.2),
            )
            block_list.append(block)
            in_channels = 2 * in_channels
        self.block_list = nn.Sequential(*block_list)
    def forward(self,x):
        '''

        :param x: [N,L]
        :return: [N,1024,L//(2^6)]
        '''
        x = torch.unsqueeze(x,dim=1)
        return self.block_list(x)
class Head(nn.Module):
    def __init__(self,signal_len,feature_shape,head_type=0):
        '''
        :param feature_shape: list  [c,l]
        :param head_type:
        '''
        super().__init__()
        self.head_type = head_type
        c,l = feature_shape
        if head_type == 0:
            self.process_block = nn.Sequential(
                nn.Conv1d(c,c,l,1,0,groups=c),nn.GroupNorm(32, c), nn.LeakyReLU(0.2),
                nn.Conv1d(c,c,1),nn.GroupNorm(32, c), nn.LeakyReLU(0.2),
                nn.Flatten()
            )
            self.head = nn.Sequential(
                nn.Linear(c, c),
                nn.GroupNorm(32,c),
                nn.LeakyReLU(0.2),

                nn.Linear(c, c//2),
                nn.GroupNorm(32,c//2),
                nn.LeakyReLU(0.2),

                nn.Linear(c // 2, signal_len)
            )
        else:
            self.process_block = nn.Sequential(
                nn.Conv1d(c, c, 3, 1, 1), nn.GroupNorm(32, c), nn.LeakyReLU(0.2),
                nn.Conv1d(c, c, 3, 1, 1), nn.GroupNorm(32, c), nn.LeakyReLU(0.2),
                nn.Flatten()
            )
            self.head =  nn.Linear(c*l, signal_len)

    def forward(self,x):
        '''

        :param x: [N,c,l]
        :return:  [N,signal_len]
        '''
        #print(x.shape)
        # if self.head_type == 1:
        h_f = self.process_block(x)
        y = self.head(h_f)
        return y
        # o,_=self.process_block(x)
        # return self.head(o)
class I_Operator(Function):
    def forward(ctx, x):
        return x.clone()
    def backward(ctx, grad_output):
        return grad_output / 4 # 2

