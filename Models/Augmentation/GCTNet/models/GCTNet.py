import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange

class GeneratorCNN(nn.Module):
    def __init__(self, data_num=512):
        super(GeneratorCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2), 
            nn.Conv1d(32, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2))
        self.pool1 = nn.AvgPool1d(2,stride=2)
        
        # block2
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        self.pool2 = nn.AvgPool1d(2,stride=2)
        
        # block3
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), 
            nn.Conv1d(128, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.pool3 = nn.AvgPool1d(2,stride=2,ceil_mode=True)
        
        # block4
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), 
            nn.Conv1d(256, 256, 3, 1, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.pool4 = nn.AvgPool1d(2,stride=2,ceil_mode=True)
        
        # block5
        self.block5 = nn.Sequential(
            nn.Conv1d(256, 512, 3, 1, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), 
            nn.Conv1d(512, 512, 3, 1, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.pool5 = nn.AvgPool1d(2,stride=2, ceil_mode=True)
        
        # block6
        self.block6 = nn.Sequential(
            nn.Conv1d(512, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), 
            nn.Conv1d(1024, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.pool6 = nn.AvgPool1d(2,stride=2,ceil_mode=True)
        
        # block7
        self.block7 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), 
            nn.Conv1d(1024, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.linear = nn.Linear(16 * 512, data_num)
    
    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.pool5(self.block5(x))
        x = self.pool6(self.block6(x))
        x = self.block7(x).reshape(x.shape[0], -1)
        
        return self.linear(x)

class GeneratorTransformer(nn.Module):
    def __init__(self, data_num=512):
        super(GeneratorTransformer, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2), 
            nn.Conv1d(32, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2))
        self.pool1 = nn.AvgPool1d(2,stride=2)
        
        # block2
        self.s2 = nn.Sequential(
            TransformerEncoderBlock(emb_size=32),
            nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        
        # block3
        self.s3 = nn.Sequential(
            TransformerEncoderBlock(emb_size=64),
            nn.Conv1d(64, 128, 3, 2, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        
        # block4
        self.s4 = nn.Sequential(
            TransformerEncoderBlock(emb_size=128),
            nn.Conv1d(128, 256, 3, 2, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        
        # block5
        self.s5 = nn.Sequential(
            TransformerEncoderBlock(emb_size=256),
            nn.Conv1d(256, 512, 3, 2, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        
        # block6
        self.s6 = nn.Sequential(
            TransformerEncoderBlock(emb_size=512),
            nn.Conv1d(512, 1024, 3, 2, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        
        # block7
        self.block7 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2), 
            nn.Conv1d(1024, 1024, 3, 1, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.linear = nn.Linear(16 * 512, data_num)
    
    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.s6(x)
        x = self.block7(x).reshape(x.shape[0], -1)
        
        return self.linear(x)
       
def align_length(a, b):
    """
    对齐两个 (B, C, L) Tensor 的时间长度
    """
    min_len = min(a.size(-1), b.size(-1))
    return a[..., :min_len], b[..., :min_len]


class Generator(nn.Module):
    def __init__(self, data_num=512):
        super(Generator, self).__init__()

        # block1
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        self.pool1 = nn.AvgPool1d(2, stride=2)

        # block2
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.s2 = nn.Sequential(
            TransformerEncoderBlock(emb_size=32),
            nn.Conv1d(32, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.pool2 = nn.AvgPool1d(2, stride=2)
        self.ffm2 = nn.Sequential(
            nn.Conv1d(128, 64, 3, 1, 1),
            nn.BatchNorm1d(64)
        )

        # block3
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.s3 = nn.Sequential(
            TransformerEncoderBlock(emb_size=64),
            nn.Conv1d(64, 128, 3, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.pool3 = nn.AvgPool1d(2, stride=2, ceil_mode=True)
        self.ffm3 = nn.Sequential(
            nn.Conv1d(256, 128, 3, 1, 1),
            nn.BatchNorm1d(128)
        )

        # block4
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        self.s4 = nn.Sequential(
            TransformerEncoderBlock(emb_size=128),
            nn.Conv1d(128, 256, 3, 2, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        self.pool4 = nn.AvgPool1d(2, stride=2, ceil_mode=True)
        self.ffm4 = nn.Sequential(
            nn.Conv1d(512, 256, 3, 1, 1),
            nn.BatchNorm1d(256)
        )

        # block5
        self.block5 = nn.Sequential(
            nn.Conv1d(256, 512, 3, 1, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 3, 1, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )
        self.s5 = nn.Sequential(
            TransformerEncoderBlock(emb_size=256),
            nn.Conv1d(256, 512, 3, 2, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )
        self.pool5 = nn.AvgPool1d(2, stride=2, ceil_mode=True)
        self.ffm5 = nn.Sequential(
            nn.Conv1d(1024, 512, 3, 1, 1),
            nn.BatchNorm1d(512)
        )

        # block6
        self.block6 = nn.Sequential(
            nn.Conv1d(512, 1024, 3, 1, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        self.s6 = nn.Sequential(
            TransformerEncoderBlock(emb_size=512),
            nn.Conv1d(512, 1024, 3, 2, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )
        self.pool6 = nn.AvgPool1d(2, stride=2, ceil_mode=True)
        self.ffm6 = nn.Sequential(
            nn.Conv1d(2048, 1024, 3, 1, 1),
            nn.BatchNorm1d(1024)
        )

        # block7
        self.block7 = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.linear = nn.Linear(8 * 512, data_num)

    def forward(self, x):
        # block1
        x = self.block1(x)
        x = self.pool1(x)

        # block2
        x_main = self.pool2(self.block2(x))
        x_skip = self.s2(x)
        x_main, x_skip = align_length(x_main, x_skip)
        x = self.ffm2(torch.cat([x_main, x_skip], dim=1))

        # block3
        x_main = self.pool3(self.block3(x))
        x_skip = self.s3(x)
        x_main, x_skip = align_length(x_main, x_skip)
        x = self.ffm3(torch.cat([x_main, x_skip], dim=1))

        # block4
        x_main = self.pool4(self.block4(x))
        x_skip = self.s4(x)
        x_main, x_skip = align_length(x_main, x_skip)
        x = self.ffm4(torch.cat([x_main, x_skip], dim=1))

        # block5
        x_main = self.pool5(self.block5(x))
        x_skip = self.s5(x)
        x_main, x_skip = align_length(x_main, x_skip)
        x = self.ffm5(torch.cat([x_main, x_skip], dim=1))

        # block6
        x_main = self.pool6(self.block6(x))
        x_skip = self.s6(x)
        x_main, x_skip = align_length(x_main, x_skip)
        x = self.ffm6(torch.cat([x_main, x_skip], dim=1))

        # block7 + fc
        x = self.block7(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            Swish(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b t (h d) -> b h t d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b t (h d) -> b h d t", h=self.num_heads)
        values = rearrange(self.values(x), "b t (h d) -> b h t d", h=self.num_heads)
        energy = torch.matmul(queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # out = torch.matmul(att, values)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.projection(out)
        return out
       
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.1,
                 forward_expansion=1,
                 forward_drop_p=0.1):
        super().__init__(  
            nn.Sequential(
                Rearrange('n (h) (w) -> n (w) (h)'),
            ),
                     
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
            nn.Sequential(
                Rearrange('n (w) (h) -> n (h) (w)'),
            )    
        )

class Discriminator(nn.Module):
    def __init__(self, Nf=12):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride = 2,padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2), 
            
            # layer1
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 2,padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
        )
        
        self.conv2 = nn.Sequential(
            # layer2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride = 2,padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            
            # layer3
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 2,padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2)
        )
        
        self.conv3 = nn.Sequential(
            # layer4
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 2,padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            
            # layer5
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 2,padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
        )
        
        self.model = nn.Sequential(
            # layer6
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 2,padding=1),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            
            # layer7
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 2,padding=1),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
        )

        self.dense_layers = nn.Sequential(nn.Linear(512, 1))
        self.cls_layers = nn.Sequential(nn.Linear(512, Nf))
        

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-2)
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.model(x3)
        out = out.view(out.size(0), -1)
        out1 = self.dense_layers(out)
        out2 = self.cls_layers(out)
       
        return out1,out2, x1, x2, x3
          
if __name__ == '__main__':
    x1 = torch.rand(128, 1, 512)
    model = Generator(data_num=512)
    output = model(x1)
    
    print('output is:', output.shape)
