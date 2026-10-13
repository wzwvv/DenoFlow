import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# 训练数据
class dataSet(Dataset):
    def __init__(self,eegc,eegp):#污染eeg 干净eeg
        super().__init__()
        self.eegc = eegc
        self.eegp = eegp

    def __len__(self):
        return len(self.eegc)

    def __getitem__(self, item):
        return self.eegc[item],self.eegp[item]
def get_trainvalid_iter(data_dir,fileName_list,batch_size,num_workers=0):
    eegc_train_list = []
    eegp_train_list = []
    eegc_val_list = []
    eegp_val_list = []
    for fileName in fileName_list:
        print(fileName)
        train_eegc_path = f'{data_dir}/{fileName}/train/eegc.pkl' #污染 eeg
        train_eegp_path = f'{data_dir}/{fileName}/train/eegp.pkl' #干净 eeg
        val_eegc_path = f'{data_dir}/{fileName}/val/eegc.pkl'
        val_eegp_path = f'{data_dir}/{fileName}/val/eegp.pkl'
        eegc_train_list.append(torch.load(train_eegc_path))
        eegp_train_list.append(torch.load(train_eegp_path))
        eegc_val_list.append(torch.load(val_eegc_path))
        eegp_val_list.append(torch.load(val_eegp_path))
    eegc_train = torch.cat(eegc_train_list, dim=0)
    eegp_train = torch.cat(eegp_train_list, dim=0)
    print(eegc_train.shape)
    eegc_val = torch.cat(eegc_val_list, dim=0)
    eegp_val = torch.cat(eegp_val_list, dim=0)
    print(eegc_val.shape)
    train_dataSet = dataSet(eegc_train,eegp_train)
    valid_dataSet = dataSet(eegc_val,eegp_val)

    #迭代器
    train_iter = DataLoader(train_dataSet,shuffle=True,batch_size=batch_size,num_workers=num_workers)
    valid_iter = DataLoader(valid_dataSet, shuffle=True, batch_size=batch_size,num_workers=num_workers)

    return train_iter,valid_iter
#测试数据
def get_test_iter(data_dir,fileName_list,batch_size,snr_list = None,num_workers=0,ex=1):
    '''

    :param data_dir:
    :param fileName_list:
    :param batch_size:
    :param snr_list:
    :return: 迭代器每个元素为 (EEG_C,EEG_P)
    '''

    eegc_list = []
    eegp_list = []
    for fileName in fileName_list:
        eegc_list.append(torch.load(f'{data_dir}/{fileName}/test/eegc.pkl'))
        eegp_list.append(torch.load(f'{data_dir}/{fileName}/test/eegp.pkl'))
    eegc = torch.cat(eegc_list, dim=0)
    eegp = torch.cat(eegp_list, dim=0)
    test_dataSet = dataSet(eegc, eegp)
    test_iter = DataLoader(test_dataSet, shuffle=False, batch_size=batch_size, num_workers=num_workers)  # 按snr顺序给出不打乱
    if ex == 2:
        return test_iter

    eegc_snr_list = []
    eegp_snr_list = []
    for fileName in fileName_list:
        if snr_list is None:
            snr_value = [f'snr={snr}' for snr in [-7, -5, -4, -3, -2, -1, 0, 1, 2]]
        else:
            snr_value = [f'snr={snr}' for snr in snr_list]
        for snr in snr_value:
            test_eegc_path = f'{data_dir}/{fileName}/test/{snr}/eegc.pkl'
            test_eegp_path = f'{data_dir}/{fileName}/test/{snr}/eegp.pkl'
            eegc_snr_list.append(torch.load(test_eegc_path))
            eegp_snr_list.append(torch.load(test_eegp_path))
    eegc_snr = torch.cat(eegc_snr_list, dim=0)
    eegp_snr = torch.cat(eegp_snr_list, dim=0)

    test_snr_dataSet = dataSet(eegc_snr, eegp_snr)
    test_snr_iter = DataLoader(test_snr_dataSet, shuffle=False, batch_size=batch_size,
                               num_workers=num_workers)  # 按snr顺序给出不打乱
    return test_iter, test_snr_iter
