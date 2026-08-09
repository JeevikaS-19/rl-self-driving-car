import torch
from perception import CNNAttentionExtractor

model = CNNAttentionExtractor()
dummy_input = torch.randn(2, 4, 64, 64)  # batch of 2
output = model(dummy_input)
print("Output shape:", output.shape)  # expect torch.Size([2, 2048])