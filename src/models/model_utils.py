import torch
import torch.nn as nn

class Attention(nn.Module):
    tanh = nn.Tanh()
    softmax = nn.Softmax(dim=1)
    def __init__(self, dim=300):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1)
        
    def forward(self, x, mask=None):
        h1 = self.W(x)
        h1 = self.tanh(h1)
        s = self.v(h1)

        if torch.is_tensor(mask):
            s.squeeze(-1)[mask==0] = -1e5
        a = self.softmax(s)
        output = torch.sum(a*x, dim=1)
        return output
