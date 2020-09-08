import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch import optim
import  matplotlib.pyplot as plt
from tqdm import tqdm, trange

def make_vocab(vocab_path, train=None):
    vocab = {}
    if os.path.isfile(vocab_path):
        file = open(vocab_path, 'r', encoding='utf-8')
        for line in file.readlines():
            line = line.rstrip()
            key, value = line.split('\t')
            vocab[key] = value
        file.close()
    else:
        count_dict = defaultdict(int)
        for index, data in tqdm(train.iterrows(), desc='make vocab', total=len(train)):
            sentence = data['Phrase'].lower()
            tokens = sentence.split(' ')
            for token in tokens:
                count_dict[token] += 1

        file = open(vocab_path, 'w', encoding='utf-8')
        file.write('[UNK]\t0\n[PAD]\t1\n')
        vocab = {'[UNK]': 0, '[PAD]': 1}
        for index, (token, count) in enumerate(sorted(count_dict.items(), reverse=True, key=lambda item: item[1])):
            vocab[token] = index + 2
            file.write(token + '\t' + str(index + 2) + '\n')
        file.close()

    return vocab

def read_data(train, test, vocab, max_len):
    x_train = np.ones(shape=(len(train), max_len))
    for i, data in tqdm(enumerate(train['Phrase']), desc='make x_train data', total=len(train)):
        data = data.lower()
        tokens = data.split(' ')
        for j, token in enumerate(tokens):
            if j == max_len:
                break
            x_train[i][j] = vocab[token]

    x_test = np.ones(shape=(len(test), max_len))
    for i, data in tqdm(enumerate(test['Phrase']), desc='make x_test data', total=len(test)):
        data = data.lower()
        tokens = data.split(' ')
        for j, token in enumerate(tokens):
            if j == max_len:
                break
            if token not in vocab.keys():
                x_test[i][j] = 0
            else:
                x_test[i][j] = vocab[token]

    y_train = train['Sentiment'].to_numpy()
    # one_hot = np.zeros(shape=(len(y_train), 5))
    # for i, y in enumerate(y_train):
    #       one_hot[i][y] = 1

    return x_train, y_train, x_test 

class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_lyaers=1, bidirec=False, device='cuda'):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_lyaers
        if bidirec:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.device = device

        self.embed = nn.Embedding(input_size, embed_size, padding_idx=1)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_lyaers, batch_first=True, bidirectional=bidirec)
        self.linear = nn.Linear(hidden_size * self.num_directions, output_size)

    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)

        return hidden, cell

    def forward(self, inputs):
        """
        inputs : B,T
        """
        embed = self.embed(inputs) # word vector indexing
        hidden, cell = self.init_hidden(inputs.size(0)) # initial hidden,cell

        output, (hidden, cell) = self.lstm(embed, (hidden, cell))

        # Many-to-Many
        # output = self.linear(output) # B,T,H -> B,T,V

        # Many-to-One
        hidden = hidden[-self.num_directions:] # (num_directions,B,H)
        hidden = torch.cat([h for h in hidden], 1)
        output = self.linear(hidden) # last hidden

        return output

def get_acc(pred, answer):
    correct = 0
    for p, a in zip(pred, answer):
        pv, pi = p.max(0)
        # av, ai = a.max(0)
        if pi == a:
            correct += 1

    return correct / len(pred)

def train(x, y, max_len, embed_size, hidden_size, output_size, batch_size, epochs, lr, device, model=None):
    x = torch.from_numpy(x).long()
    y = torch.from_numpy(y).long()
    if model is None:
        model = RNN(max_len, embed_size, hidden_size, output_size, device=device)
    model.to(device)
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size, shuffle=True)
    loss_total = []
    acc_total = []
    for epoch in trange(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for batch_data in data_loader:
            x_batch, y_batch = batch_data
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # forward
            pred = model(x_batch)

            # backwrd
            loss = loss_function(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()

            # update
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += get_acc(pred, y_batch)
        epoch_loss /= len(data_loader)
        epoch_acc /= len(data_loader)
        loss_total.append(epoch_loss)
        acc_total.append(epoch_acc)
        print("\nEpoch [%d] Loss: %.3f\tAcc: %.3f" % (epoch + 1, epoch_loss, epoch_acc))

    torch.save(model, 'model.out')

    return model, loss_total, acc_total

def test(model, x, batch_size, device):
    model.to(device)
    model.eval()
    x = torch.from_numpy(x).long()
    data_loader = torch.utils.data.DataLoader(x, batch_size, shuffle=False)

    predict = []
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        pred = model(batch_data)
        for p in pred:
            pv, pi = p.max(0)
            predict.append(pi.item())

    return predict

def draw_graph(data):
    plt.plot(data)
    plt.show()

def save_submission(pred):
    submit = pd.read_csv('./data/sampleSubmission.csv', index_col='PhraseId')
    submit["Sentiment"] = pred
    submit = pd.DataFrame(submit)
    submit.to_csv('./data/my_submission.csv')

if __name__ == '__main__':
    train_path = 'data/train.tsv'
    test_path = 'data/test.tsv'
    vocab_path = 'data/vocab.txt'

    train_data = pd.read_csv(train_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    vocab = make_vocab(vocab_path, train_data)
    ##model = torch.load('model.out')

    device = torch.device('cuda:0')
    max_len = 50
    input_size = len(vocab)
    embed_size = 50
    hidden_size = 100
    output_size = 5
    batch_size = 1024
    epochs = 10
    lr = 0.001

    x_train, y_train, x_test = read_data(train_data, test_data, vocab, max_len)

    model, loss_total, acc_total = train(x_train, y_train, input_size, embed_size, hidden_size, output_size, batch_size, epochs, lr, device)
    draw_graph(loss_total)
    draw_graph(acc_total)
    predict = test(model, x_test, batch_size, device)
    save_submission(predict)