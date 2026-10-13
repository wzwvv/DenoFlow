

import torch
from torch import nn
import numpy as np
from scipy import signal
import torch.nn.functional as F

def denoise_loss_mse(denoise, clean):      
  loss = torch.nn.MSELoss()
  return loss(denoise, clean)

def denoise_loss_l1(denoise, clean):      
  loss = torch.nn.L1Loss()
  return loss(denoise, clean)

