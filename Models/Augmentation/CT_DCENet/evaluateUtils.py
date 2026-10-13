import matplotlib.pyplot as plt
import torch
from scipy.signal import welch
# import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch
from audtorch.metrics import PearsonR

from models.models import model_structure


def Cal_RMS(signal):
    '''

    :param signal: (N,T)  N条信号
    :return:(N,1)  N条信号的均方根能量
    '''
    return torch.sqrt((signal**2).mean(dim=1,keepdim=True))
def Cal_SNR(restructed_eeg,clean_eeg,device = 'cpu',reduction = 'mean'):
    '''
    计算restructed_eeg的信噪比
    :param restructed_eeg: 模型重构后的eeg (N,T)
    :param clean_eeg: 干净eeg (N,T)
    :param device: 进行计算的设备
    :param reduction:

    :return:
        reduction:
            'none' :(N,1)
            'mean' :(,)
            'sum': (,)
    '''
    assert reduction == 'none' or reduction == 'mean' or reduction == 'sum','reduction输入错误'
    restructed_eeg = restructed_eeg.to(device) #(N,T)
    clean_eeg = clean_eeg.to(device) #(N,T)
    noise = restructed_eeg - clean_eeg #(N,T)
    clean_s = (clean_eeg ** 2).sum(dim = 1,keepdim = True) #(N,1)
    noise_s = (noise ** 2).sum(dim = 1,keepdim = True) #(N,1)
    snr = 10 * torch.log10(clean_s / noise_s) #(N,1)
    if reduction == 'none':
        return snr #(N,1)
    elif reduction == 'mean':
        return snr.mean() #(,)
    else:
        return snr.sum() #(,)
def Cal_RRMSE(restructed_eeg,clean_eeg,reduction = 'mean'):
    '''
    计算相对均方根误差
    :param restructed_eeg: 模型重构后的eeg (N,T)
    :param clean_eeg: 干净eeg (N,T)
    :param reduction:
    :return:
        reduction:
            'none' :(N,1)
            'mean' :(,)
            'sum': (,)
    '''
    assert reduction == 'none' or reduction == 'mean' or reduction == 'sum', 'reduction输入错误'
    noise = restructed_eeg - clean_eeg#(N,T)
    noise_rms = Cal_RMS(noise) #(N,1)
    clean_rms = Cal_RMS(clean_eeg)#(N,1)
    rrms = noise_rms / clean_rms #(N,1)
    if reduction == 'none':
        return rrms
    elif reduction == 'mean':
        return rrms.mean()
    else:
        return rrms.sum()
def Cal_RRMSE_t(restructed_eeg,clean_eeg,device = 'cpu',reduction = 'mean'):
    '''
    计算时域相对均方根误差
    :param restructed_eeg: 模型重构后的eeg (N,T)
    :param clean_eeg: 干净eeg (N,T)
    :param device: 进行计算的设备
    :param reduction:
    :return:
        reduction:
            'none' :(N,1)
            'mean' :(,)
            'sum': (,)
    '''
    restructed_eeg,clean_eeg = restructed_eeg.to(device),clean_eeg.to(device)
    return Cal_RRMSE(restructed_eeg,clean_eeg,reduction)
def Cal_CC(pred_eeg,real_eeg,reduction='mean',device='cpu'):
    '''
    计算预测结果的相关系数CC
    :param pred_eeg:(N,T)
    :param real_eeg:(N,T)
    :return:(N,1)
    '''
    cc = PearsonR(reduction=reduction,batch_first=True)
    return cc(pred_eeg,real_eeg)
def evaluate(model,test_iter,device, show=True):
    #model_structure(model)
    model.eval()
    model.to(device)
    snr_list = []
    rrmset_list = []
    cc_list = []
    with torch.no_grad():
        for eegc,eegp in test_iter:
            eegc = eegc.to(device)
            eegp = eegp.to(device)
            # EEG时域信号内部单独做标准化(既不考虑其他信号，等价于LayerNorm)，并且均值使用经验常数0
            std = eegc.std(dim=1, keepdims=True)
            eegc = eegc / std
            eegp = eegp / std
            pred_eegp = model.denoise(eegc)  # (N,L)
            snr_list.append(Cal_SNR(pred_eegp, eegp, device, 'none'))
            rrmset_list.append(Cal_RRMSE_t(pred_eegp, eegp, device, 'none'))
            cc_list.append(Cal_CC(pred_eegp, eegp, 'none'))
    if show:
        plt.plot(eegp[-1].cpu().detach().numpy(), color='green', label='clean EEG')
        plt.plot(pred_eegp[-1].cpu().detach().numpy(), color='blue', label='reconstructed EEG')
        plt.plot(eegc[-1].cpu().detach().numpy(), color='red', label='contaminated EEG')
        plt.legend()  # 添加图例
        plt.show()
    snr = torch.cat(snr_list, dim=0)
    rrmse_t = torch.cat(rrmset_list, dim=0)
    cc = torch.cat(cc_list, dim=0)
    print(f"snr = {snr.mean():.3f}, rrmse_t = {rrmse_t.mean():.3f}, cc = {cc.mean():.3f}")
    return snr.mean(),rrmse_t.mean(),cc.mean()