import torch
from torch.nn import functional as F
from torch import nn

class DenseNN(nn.Module):
    def __init__(self, seq_length):
        super(DenseNN, self).__init__()
        
        self.fc1 = nn.Linear(seq_length*2, int(seq_length*1.5) )
        self.fc2 = nn.Linear(int(seq_length*1.5), int(seq_length*0.5))
        self.fc3 = nn.Linear(int(seq_length*0.5), 2)

    def forward(self, x, seq_length):
        h1 = F.relu( self.fc1(x.contiguous().view(-1, seq_length*2)) )
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

class DenseNN_init(nn.Module):
    def __init__(self, seq_length):
        super(DenseNN_init, self).__init__()
        
        self.fc1 = nn.Linear(seq_length*2, int(seq_length*1.5) )
        self.fc2 = nn.Linear(int(seq_length*1.5), int(seq_length*0.5))
        self.fc3 = nn.Linear(int(seq_length*0.5), 2)
        with torch.no_grad():
            self.fc1.weight = torch.nn.Parameter( torch.load('fc1.pt') )
            self.fc2.weight = torch.nn.Parameter( torch.load('fc2.pt') )
            self.fc3.weight = torch.nn.Parameter( torch.load('fc3.pt') )

    def forward(self, x, seq_length):
        h1 = F.relu( self.fc1(x.view(-1, seq_length*2)) )
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)
