import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from .financedata import WindowTokenizerParser, typeset, Var
import torch
from math import sqrt
from torch.autograd import Variable

class StockPriceModel(torch.nn.Module):
    def __init__(self, input, hidden, output, old=False):
        super().__init__()
        self.hidden_size = hidden
        self.lstm = torch.nn.LSTM(input, hidden, batch_first = True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param, gain=0.001)
        self.drop1 = torch.nn.Dropout(.5)
        self.linear = torch.nn.Linear(hidden, output)
        self.drop2 = torch.nn.Dropout(.1)
        self.output_size = output
        # self.old = old
    def forward(self, x):
        output , (hidden , _) = self.lstm(x)
        hn = hidden.view(-1, self.hidden_size)
        # if self.old:
        #     return self.linear(hn)
        on = output.view(-1, self.hidden_size)
        # print(hidden.shape, output.shape)
        o1 = self.drop1(hn)
        ol = self.linear(o1)
        o2 = self.drop2(ol)
        return o2 #self.linear(hn)
    def load_from_path(path):
        return torch.load(path)


class Predictor():
    def __init__(self, model: StockPriceModel, optimizer, loss_fn, batch_size = 256):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.loaded = False
        self.trained = False
    def train(self, train_x, train_y, num_epochs = 100, logs = True, log_step = 10):
        print(len(train_x), len(train_y))
        assert len(train_x) == len(train_y)
        old_loss = 0.0

        if self.loaded:
            print("The model is already loaded and trained!")
        for epoch in range(num_epochs):
            print("\n\n")
            acc_loss = 0.0
            n_batches = train_x.shape[0] // self.batch_size + 1
            for i in range(train_x.shape[0] // self.batch_size + 1):
                le = train_x.shape[0]
                self.model.eval()
                outputs = self.model(train_x[(i * self.batch_size):min((i+1) * self.batch_size, le)])
                self.model.train()

                loss = self.loss_fn(outputs , train_y[(i * self.batch_size):min((i+1) * self.batch_size, le)])
                # print(f"EPOCH: {epoch}\n", outputs,':outputs\npredict_y:\n',  train_y, "\n\n@\n\n", train_x, " loss = " + "\n\n\n")
                if loss.detach().numpy() < 1:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    acc_loss += loss.detach().numpy()
                    print(epoch, " Batch: ", i, " ] loss: ", old_loss, " | ", loss.detach().numpy(), " [" ,acc_loss,"]")
            old_loss = acc_loss / n_batches
            if logs and epoch != 0:
                print(epoch , "epoch loss" , " > batches " , n_batches , " : ", acc_loss / n_batches)
        self.trained = True
    def predict(self, test_x):
        if not self.loaded:
            if self.trained == False:
                print("Train the model first!")
                return
        self.model.eval()
        with torch.no_grad():
            output = self.model(test_x)
        return output
    
if __name__ == "__main__":
    symbols = pd.read_csv("./symbols.csv")['Name'].values
    print(symbols)
    INPUT_DIM = 30
    OUTPUT_DIM = 5
    
    parser = WindowTokenizerParser(INPUT_DIM, OUTPUT_DIM, diff=True)
    d = parser.get_stocks(symbols=symbols)
    X_train, y_train, x_test, y_test = parser.splitset(d, 0.8)
    X_train, y_train, x_test, y_test = typeset(X_train, y_train, x_test, y_test)
    X_train, y_train, x_test, y_test = Var(X_train, y_train, x_test, y_test) 
    X_train = torch.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    x_test = torch.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    print(X_train.shape, y_train.shape)
    m = StockPriceModel(INPUT_DIM, 100, output=OUTPUT_DIM)
    print(m.parameters())
    optimizer = torch.optim.Adam(m.parameters() , lr = 1e-2)
    loss_fn = torch.nn.MSELoss()
    pred = Predictor(m, optimizer, loss_fn=loss_fn)
    
    pred.train(X_train, y_train, num_epochs=150)
    
    print("Save model? 0/1")
    a = input()
    print("Set name: ")
    name = input()
    if a == '1': 
        print(pred.model)
        torch.save(pred.model, "./" + name + ".t7")
        torch.save(pred.model.state_dict(), "./" + name + "-states.t7")

    