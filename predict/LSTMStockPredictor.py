import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from financedata import WindowTokenizerParser, typeset, Var
import torch
from torch.autograd import Variable
from symbols import abbrevations
class StockPriceModel(torch.nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.hidden_size = hidden
        self.lstm = torch.nn.LSTM(input, hidden, batch_first = True)
        self.linear = torch.nn.Linear(hidden, output)
    def forward(self, x):
        _ , (hidden , _) = self.lstm(x)
        hn = hidden.view(-1, self.hidden_size)
        return self.linear(hn)

class Predictor():
    def __init__(self, model: StockPriceModel, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loaded = False
        self.trained = False
    def train(self, train_x, train_y, num_epochs = 100, logs = True, log_step = 10):
        print(len(train_x), len(train_y))
        assert len(train_x) == len(train_y)
        if self.loaded:
            print("The model is already loaded and trained!")
        for epoch in range(num_epochs):
            outputs = self.model(train_x)
            loss = self.loss_fn(outputs , train_y)
            # print(f"EPOCH: {epoch}\n", outputs,':outputs\npredict:\n',  train_y, "\n\n@\n\n", train_x, "\n\n\n")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if logs and epoch % log_step == 0 and epoch != 0:
                print(epoch , "epoch loss" , loss.detach().numpy())
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
    
    parser = WindowTokenizerParser(INPUT_DIM, OUTPUT_DIM)
    d = parser.get_stocks(symbols=symbols)
    X_train, y_train, x_test, y_test = parser.splitset(d, 0.8)
    X_train, y_train, x_test, y_test = typeset(X_train, y_train, x_test, y_test)
    X_train, y_train, x_test, y_test = Var(X_train, y_train, x_test, y_test) 
    X_train = torch.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    x_test = torch.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    print(X_train.shape, y_train.shape)
    m = StockPriceModel(INPUT_DIM, 1000, output=OUTPUT_DIM)
    optimizer = torch.optim.Adam(m.parameters() , lr = 0.001)
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
    