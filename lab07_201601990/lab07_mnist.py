import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt



def read_data(train_path, test_path):

    train = pd.read_csv(train_path)
    y_train = train['label']
    y_list = np.zeros(shape=(y_train.size, 10))
    for i, y in enumerate(y_train):
        y_list[i][y] = 1
    y_train = y_list

    del train['label']
    x_train = train.to_numpy() / 255

    test = pd.read_csv(test_path)
    x_test = test.to_numpy() / 255

    return x_train, y_train, x_test



def get_acc(pred, answer):

    correct = 0
    for p, a in zip(pred, answer):
        pv, pi = p.max(0)
        av, ai = a.max(0)
        if pi == ai:
            correct += 1
    return correct / len(pred)



class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 10)

    def forward(self, x):

        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        # x = torch.relu(self.fc4(x))
        # x = self.fc5(x)
        return x3



def train(x_train, y_train, batch, lr, epoch):

    # model 셋팅
    model = MNISTModel()
    model.train()

    loss_function = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # data 처리
    x = torch.from_numpy(x_train).float()
    y = torch.from_numpy(y_train).float()

    data_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch, shuffle=True)

    epoch_loss = []
    epoch_acc = []
    for e in range(epoch):
        total_loss = 0
        total_acc = 0
        for data in data_loader:
            x_data, y_data = data

            #forward 문제 풀이
            pred = model(x_data)

            #backward 채점 및 학습
            loss = loss_function(pred, y_data)
            optimizer.zero_grad()
            loss.backward()

            #update 학습 반영
            optimizer.step()

            total_loss += loss.item()
            total_acc += get_acc(pred, y_data)

        epoch_loss.append(total_loss / len(data_loader))
        epoch_acc.append(total_acc / len(data_loader))
        print("Epoch [%d] Loss:  %.3f\tAcc: %.3f" % (e + 1, epoch_loss[e], epoch_acc[e]))

    return model, epoch_loss, epoch_acc



def test(model, x_test, batch):
    model.eval()

    x = torch.from_numpy(x_test).float()
    data_loader = torch.utils.data.DataLoader(x, batch, shuffle=False)

    preds = []
    for data in data_loader:
        pred = model(data)
        for p in pred:
            pv, pi = p.max(0)
            preds.append(pi.item())

    return preds



def save_pred(pred_path, preds):
    submit = pd.read_csv('./data/sample_submission.csv', index_col='ImageId')
    submit["Label"] = preds
    submit = pd.DataFrame(submit)
    submit.to_csv(pred_path)



def draw_graph(epoch_loss, epoch_acc):
    plt.plot(epoch_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch per Loss')
    plt.show()

    plt.plot(epoch_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch per Accuracy')
    plt.show()



if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    save_path = 'data/my_submissionn.csv'
    batch = 128
    lr = 0.001
    epoch = 10

    x_train, y_train, x_test = read_data(train_path, test_path)
    model, epoch_loss, epoch_acc = train(x_train, y_train, batch, lr, epoch)
    preds = test(model, x_test, batch)

    #저장
    save_pred(save_path, preds)
    #시각화
    draw_graph(epoch_loss, epoch_acc)
