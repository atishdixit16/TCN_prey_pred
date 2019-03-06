import torch

class DenseNN(nn.Module):
    def __init__(self, seq_length):
        super(DenseNN, self).__init__()

        self.fc1 = nn.Linear(seq_length*2, int(seq_length*1.5) )
        self.fc2 = nn.Linear(int(seq_length*1.5), int(seq_length*0.5)
        self.fc3 = nn.Linear(int(seq_length*0.5), 2)

    def forward(self, x):
        h1 = self.fc1(x.view(-1, seq_length*2))

        return self.decode(z), z
