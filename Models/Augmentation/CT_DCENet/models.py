import math
import torch
from torch import nn
from torch.nn import functional as F
from Models.Augmentation.CT_DCENet.components import FeatureLearner, Head, I_Operator

def model_structure(model):
    blank = ' '
    #print('-' * 90)
    #print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
    #      + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
    #      + ' ' * 3 + 'number' + ' ' * 3 + '|')
    #print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        #print('| {} | {} | {} |'.format(key, shape, str_num))
    #print('-' * 90)
    #print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    #print('-' * 90)

class CT_DCENet(nn.Module):
    def __init__(self,signal_len=512,head_type_list=[0,0,1,1],block_num=5):
        super().__init__()
        self.signal_len = signal_len
        self.head_num = len(head_type_list)
        self.feature_learner = FeatureLearner(block_num=block_num)
        head_list = []
        feature_shape = [32 * (2**block_num) ,math.ceil(signal_len/(2**(block_num+1)))]
        for head_type in head_type_list:
            head = Head(signal_len,feature_shape,head_type)
            head_list.append(head)
        self.heads = nn.ModuleList(head_list)
        self.sl = nn.Sequential(
            nn.Linear(self.head_num, 1),
            nn.Flatten()
        )
        self.supplement = nn.Sequential(
            FeatureLearner(block_num=block_num),
            Head(signal_len, feature_shape, 0)
        )
        self.beta = nn.Parameter(torch.randn(size=(1,)))
    def forward(self,eegc):
        '''

        :param eegc:  [N,L]
        :return:
        '''
        feature = self.feature_learner(eegc)
        fl_out_list = []
        for i in range(self.head_num):
            head = self.heads[i]
            x = head(feature)
            head_out = I_Operator.apply(x)
            fl_out_list.append(head_out)
        return fl_out_list
    def denoise(self,eegc):
        fl_out_list = self.forward(eegc)

        fl_out = torch.stack(fl_out_list, dim=-1)  # [N,L,num_head]
        fl_out = fl_out.detach()
        # fl_out = torch.stack(fl_out_list, dim=1)  # [N,num_head,L]
        # fl_out = fl_out.detach()
        sl_out = self.sl(fl_out)

        sup = self.supplement(eegc - sl_out)
        w  = F.sigmoid(self.beta)
        eegr = sl_out + w * sup

        return eegr





