import random

import torch
from torch.nn import functional as F
def cal_col(pred_list,i): # cal q_(h)
    # H = len(pred_list)
    # y_consensus = torch.stack(pred_list,dim=1) #[N,H,L]
    # y_consensus[:,i,:] = torch.zeros_like(y_consensus[:,i,:])
    # y_consensus =  y_consensus.sum(dim=1) # [N,L]
    # y_consensus = y_consensus / (H-1)
    # return y_consensus.detach()
    # choice_list = pred_list[0:i] + pred_list[i+1:]
    # return random.choice(choice_list).detach()
    # j = 1 + 4*(i//2) - i
    # return pred_list[j]
    choice_list = pred_list[0:i] + pred_list[i + 1:]
    choice = torch.stack(choice_list,dim=1) #[N,N-1,L]
    choice = choice.mean(dim=1) #[N,L]
    choice_list.append(choice)
    col = random.choice(choice_list).detach()
    return col
def collaborative_loss(i,pred_list,y_true,d_f,beta_height=0.05):
    '''

    :param y_pred:
    :param y_consensus:
    :param y_true:
    :param d_f:  loss function
    :param lamda:
    :param p:
    :return:
    '''
    # print(beta_height)
    pred = pred_list[i] # eeg_d_i
    l_target = d_f(pred,y_true)
    col = cal_col(pred_list,i) # eeg_col
    lamda = random.uniform(0,beta_height)
    l_cor = d_f(pred, col)
    loss = l_target + lamda * l_cor

    return loss