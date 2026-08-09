# handles cnn+attention feature extractor

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAttentionExtractor(nn.Module):
    def __init__(self):
        super(CNNAttentionExtractor, self).__init__()

        #phase2a: downsampling cnn
        #input: batch 4,64,64 --> output: batch 8,32,32
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,stride = 2, padding=1)
        #input: batch 8,32,32 --> 16,16,16
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride = 2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride = 2, padding=1)

        #phase 2b: lightwt self attention
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=1,batch_first= True)

        #posiotional encoding 
        self.pos_encoding = nn.Parameter(torch.randn(1,64,32)*0.02)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        batch_size, channels, height, width = x.size()

        #reshape the attention
        x = x.view(batch_size, channels, height*width)

        x = x.permute(0,2,1)
        x=x+self.pos_encoding

        attn_output, _ = self.attention(x,x,x)

        #flatten into latent vector
        latent_vector = torch.flatten(attn_output, start_dim=1)

        return latent_vector
