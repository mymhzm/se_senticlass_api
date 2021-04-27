# test.py
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    # 不需要生成图了,
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs = torch.max(outputs, dim=1)[1] # 1位置是index
            ret_output += outputs.int().tolist()

    return ret_output