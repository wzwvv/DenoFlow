import time
import torch
import torch.nn.functional as F
from dataUtils import get_trainvalid_iter
from modelUtils import setup_seed
from trainUtils import train_individual, train_ensemble
import argparse
import ast
def train_ex1(configs):
    # from configs.ex1 import configs
    #setup_seed(3407)
    # 数据集
    # 数据集(迭代器)  (干净eeg,污染eeg)   (N,L)
    data_dir = configs.data_dir
    fileName_list = configs.fileName_list
    batch_size = configs.batch_size
    num_workers = configs.num_workers
    save_root = configs.save_dir
    block_num = configs.block_num
    print(configs.beta_height)
    
    for fileName in fileName_list:
        # save_dir = f'{save_root}/{beta_name_dict[configs.beta_height]}/{fileName}'\
        save_dir = f'{save_root}/block={block_num}/{fileName}'
        train_iter, val_iter = get_trainvalid_iter(data_dir, [fileName], batch_size, num_workers=num_workers)
        # 训练+训练曲线
        start = time.time()
        train_individual(train_iter,val_iter,save_dir,configs)
        train_ensemble(train_iter,val_iter,save_dir,configs)
        end = time.time()
        print(end - start)

def parse_list_arg(list_arg):
    try:
        return ast.literal_eval(list_arg)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list argument: {list_arg}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/datasetI') # 数据根目录
    parser.add_argument('--fileName_list', type=parse_list_arg,default=['EMG']) # 噪声目录
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--num_individual_epochs",type=int, default=50)
    parser.add_argument("--num_ensemble_epochs",type=int, default=50)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--num_workers",type=int, default=0)
    parser.add_argument("--block_num",type=int, default=5)
    parser.add_argument("--save_dir",type=str, default='save_models/ex1/CT-DCENet')
    parser.add_argument("--lr",type=float, default=0.001)
    parser.add_argument("--betas",type=parse_list_arg, default=[0.9, 0.99])
    parser.add_argument("--eps",type=float, default=1e-8)
    parser.add_argument("--clip_norm",type=bool, default=None)
    parser.add_argument("--milestones",type=parse_list_arg, default=[20,30,40])
    parser.add_argument("--head_type_list",type=parse_list_arg, default=[0,0,1,1])
    parser.add_argument("--loss_list",type=parse_list_arg, default=[F.l1_loss,F.mse_loss,F.l1_loss,F.mse_loss])
    parser.add_argument("--beta_height",type=float, default=0.05) ####
    parser.add_argument("--ablation",type=bool, default=False) ####
    configs = parser.parse_args()
    train_ex1(configs)



