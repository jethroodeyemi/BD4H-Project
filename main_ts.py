# Based on: https://github.com/pytorch/examples/tree/main/word_language_model

import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import data
import model_ts

parser = argparse.ArgumentParser(description='PyTorch SCIDATOS Time Series RNN/LSTM Model')
parser.add_argument('--data', type=str, default='.',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nfeatures', type=int, default=43,
                    help='number of input features (time series)')
parser.add_argument('--insize', type=int, default=200,
                    help='size of input for RNN')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--min_epochs', type=int, default=10,
                    help='minimum epoch before early stopping')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=48,
                    help='sequence length')
parser.add_argument('--seqoverlap', type=float, default=0.5,
                    help='sequence overlap')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

timeseries = data.TimeseriesNumPy(args.data, args.nfeatures)

def batchify(rawdata, bsz):
    nbatch = rawdata[0].size(0) // bsz
    data = rawdata[0].narrow(0, 0, nbatch * bsz)
    data = data.view(-1, nbatch, args.nfeatures)
    data = data.permute(1,0,2).contiguous()

    ys = rawdata[1].narrow(0, 0, nbatch * bsz)
    ys = ys.view(-1, nbatch, 1)
    ys = ys.permute(1,0,2).contiguous()
    
    return [data.to(device), ys.to(device)]

def expand_and_batchify(rawdata, bsz, step=0.5):
    seqlen = args.bptt
    rawlen = int(len(rawdata[0]))
    stepsize = int(seqlen * step)
    
    timesteps = (math.floor((rawlen-seqlen)/stepsize)+1) * seqlen
    timesteps += rawlen - (math.floor((rawlen-seqlen)/stepsize)+1) * stepsize
    
    steps = torch.Tensor(timesteps, args.nfeatures)
    targets = torch.Tensor(timesteps)

    pos = 0
    for i in range(0, rawlen-seqlen+1, stepsize):
        for j in range(0, seqlen):
            steps[pos] = torch.from_numpy(rawdata[0][i + j])
            targets[pos] = float(rawdata[1][i + j])
            pos += 1
    
    remainderstart = (math.floor((rawlen-seqlen)/stepsize)+1) * stepsize
    for k in range(remainderstart, rawlen):
        steps[pos] = torch.from_numpy(rawdata[0][k])
        targets[pos] = float(rawdata[1][k])
        pos += 1
    
    return batchify([steps, targets], bsz)




eval_batch_size = 10

train_data = expand_and_batchify(timeseries.train, args.batch_size)
val_data = expand_and_batchify(timeseries.valid, eval_batch_size)
test_data = expand_and_batchify(timeseries.test, eval_batch_size)

model = model_ts.RNNModelTS(args.model, args.nfeatures, args.insize, 
                            args.nhid, args.nlayers, args.dropout).to(device)

criterion = nn.MSELoss()

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source[0]) - 1 - i)
    data = source[0][i:i+seq_len]
    target = source[1][i:i+seq_len]
    return data, target

def flatten(mydata):
    return mydata.permute(1,0,2).contiguous().view(-1,1)

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source[0].size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source[0]) - 1)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data[0].size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | mse {:8.5f}'.format(
                epoch, batch, len(train_data[0]) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, cur_loss))
            total_loss = 0
            start_time = time.time()

def export_onnx(path, batch_size, seq_len):
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

lr = args.lr
best_val_loss = None

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid mse {:8.5f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, val_loss))
        print('-' * 89)
        if epoch >= args.min_epochs and (not best_val_loss or val_loss < best_val_loss):
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:
    print('Exiting from training early')

with open(args.save, 'rb') as f:
    model = torch.load(f, weights_only=False)
    model.rnn.flatten_parameters()

test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test mse {:8.5f}'.format(test_loss, test_loss))
print('=' * 89)

if len(args.onnx_export) > 0:
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
