import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from TCN.prey_pred_model.model import TCN
from TCN.prey_pred_model.utils import data_generator
from TCN.prey_pred_model.prey_pred_data import *
from TCN.prey_pred_model.dense_model import *
import matplotlib.pyplot as plt
import os

os.system('mkdir rand_init_denseNN')

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size (default: 100)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=12000,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 1)')
parser.add_argument('--seq_len', type=int, default=10,
                    help='sequence length (default: 10)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=5,
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
total_examples = 6000
training_examples = 3000

print(args)
print("Producing data...")
data = prey_pred_data_randinit(total_examples, seq_length)
X_train = data[0][:training_examples,:,:]
Y_train = data[1][:training_examples,:]
X_test = data[0][training_examples:,:,:]
Y_test = data[1][training_examples:,:]

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
# # with dt
# model = DenseNN_init(seq_length + 1)
# without dt
model = DenseNN(seq_length)


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
        # # with dt
        # output = model(x, seq_length+1)
        # without dt
        output = model(x, seq_length)
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
    # # with dt
    # output = model(X_test, seq_length + 1)
    # without dt
    output = model(X_test, seq_length )
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
plt.savefig('rand_init_denseNN/prey_pred_losses_randInit.jpg')

# # with dt
# stack_data = prey_pred_data_init_withdt(total_examples, seq_length, [80,50])
# X = stack_data[:,:,:-1]
# y = stack_data[:,:,-1]
# data = (x,y)
# X_test = X[training_examples:,:,:]
# without dt
data = prey_pred_data(total_examples, seq_length)
X_test = data[0][training_examples:,:,:]
if args.cuda:
    X_test = X_test.cuda()
model.eval()
# # with dt
# output = model(X_test, seq_length + 1)
# without dt
output = model(X_test, seq_length )

plt.figure()
plt.plot(range(total_examples),data[1][:,0].cpu().numpy())
plt.plot(range(total_examples),data[1][:,1].cpu().numpy())
plt.plot(range(training_examples,total_examples),output[:,0].cpu().detach().numpy(), '--')
plt.plot(range(training_examples,total_examples),output[:,1].cpu().detach().numpy(), '--')
plt.grid(True)
plt.legend(['True prey population', 'True predator population', 'Predicted prey population', 'Predicted predator population'] ,loc='center left')
plt.savefig('rand_init_denseNN/prey_pred_prediction_vizualization_randInit.jpg')
