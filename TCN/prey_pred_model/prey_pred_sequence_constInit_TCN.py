import argparse
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from TCN.prey_pred_model.model import TCN
from TCN.prey_pred_model.utils import data_generator
from TCN.prey_pred_model.prey_pred_data import prey_pred_data_constinit

import matplotlib.pyplot as plt

os.system('mkdir const_init')

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size (default: 100)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=150,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=1,
                    help='# of levels (default: 1)')
parser.add_argument('--seq_len', type=int, default=10,
                    help='sequence length (default: 10)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    args.cuda = False

input_channels = 2
n_classes = 2
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs
total_examples = 1500
training_examples = 800

print(args)
print("Producing data...")
data = prey_pred_data_constinit(total_examples, seq_length)
X_train = torch.from_numpy ( data[0][:training_examples,:,:] ).float()
Y_train = torch.from_numpy ( data[1][:training_examples,:]).float()
X_test = torch.from_numpy ( data[0][training_examples:,:,:]).float()
Y_test = torch.from_numpy ( data[1][training_examples:,:]).float()

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

if args.cuda:
    model.cuda()
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

train_loss_data = []
val_loss_data = []

for epoch in range(1, epochs+1):
    # train    
    model.train()
    batch_idx = 1
    total_loss = 0
    mse_loss = 0
    for i in range(0, X_train.size()[0], batch_size):
        if i + batch_size > X_train.size()[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.data[0]
        mse_loss += ( x.size()[0] / X_train.size()[0] ) * loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, X_train.size()[0])
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size()[0], 100.*processed/X_train.size()[0], lr, cur_loss))
            total_loss = 0
    train_loss_data.append(mse_loss)
    #test
    model.eval()
    output = model(X_test)
    test_loss = F.mse_loss(output, Y_test)
    print('Epoch ({}) Test set: Average loss: {:.6f}\n'.format(epoch, test_loss.data[0]))
    val_loss_data.append(test_loss)

plt.figure()
plt.plot(range(2, args.epochs + 1), train_loss_data[1:])
plt.plot(range(2, args.epochs + 1), val_loss_data[1:])
plt.xlabel('epoch')
plt.ylabel('losses')
plt.title('Trainng and Validation Losses')
plt.legend(['Training Loss','Validation Loss'])
plt.savefig('const_init/prey_pred_losses_constInit.jpg')

plt.figure()
plt.plot(range(total_examples),data[1][:,0])
plt.plot(range(total_examples),data[1][:,1])
plt.plot(range(training_examples,total_examples),output[:,0].cpu().detach().numpy(), '--')
plt.plot(range(training_examples,total_examples),output[:,1].cpu().detach().numpy(), '--')
plt.legend(['True prey population', 'True predator population', 'Predicted prey population', 'Predicted predator population'] ,loc='center left')
plt.savefig('const_init/prey_pred_prediction_vizualization_constInit.jpg')