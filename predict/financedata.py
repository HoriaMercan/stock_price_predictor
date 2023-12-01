import yfinance as yf
import pandas as pd
import numpy as np
from multiprocessing import Pool
from math import floor
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from symbols import abbrevations
pd.options.display.float_format = '{:.2f}'.format
np.random.seed(42)

class CSVData:
    pass

class FinanceData(CSVData):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def getInfo(stock):
        msft = yf.Ticker(stock)
        # get all stock info
        return msft.info
    
    def getHistory(stock, period):
        msft = yf.Ticker(stock)
        h = msft.history(period=period)
        return h
    
class WindowTokenizer:
    def __init__(self, data : CSVData):
        self.data = data
        self.selected = data
        
    def make_window(self, window, column):
        x = np.concatenate([[np.nan] * (window-1), self.data[column].values])
        shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
        strides = x.strides + (x.strides[-1],)
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    
    def make(self, x_size, y_size, column, store = True):
        d = self.make_window(x_size + y_size, column=column)
        d = d[~np.isnan(d).any(axis=1)]
        X = d[:, :x_size]
        y = d[:, x_size:]
        if store:
            self.X = X
            self.y = y
            return self.X, self.y
        return X, y

from yfinance import shared

class WindowTokenizerParser:
    def __init__(self, input, output):
        self.input = input
        self.output = output
    def get_stock(self, symbol):
        data = FinanceData.getHistory(symbol, "1y")
        tok = WindowTokenizer(data=data)
        X, y = tok.make(self.input, self.output, "Close")
        return X, y

    def get_stocks(self, symbols, num_threads=9):
        with Pool(num_threads) as p:
            return p.map(self.get_stock, symbols)
        
    def unzip(self, stocks_array_predictions):
        a = stocks_array_predictions[0][0]
        b = stocks_array_predictions[0][1]
        for i in range(1, len(stocks_array_predictions)):
            a = np.concatenate((a, stocks_array_predictions[i][0]))
            b = np.concatenate((b, stocks_array_predictions[i][1]))
        return a, b
        
    def splitset(self, stocks_array_predictions, train_percentage, to_tensor=True):
        X, y = self.unzip(stocks_array_predictions)
        
        scaler = MinMaxScaler()
        scaler.fit(X.transpose())
        new_data = scaler.transform(np.concatenate((X.transpose(), y.transpose()))).transpose()
        X = new_data[:,:self.input]
        y = new_data[:,self.input:]

        train_len = floor(train_percentage * len(X))
        assert len(X) == len(y)
        p = np.random.permutation(len(X))
        X_train, y_train = X[p][:train_len,:], y[p][:train_len,:]
        X_test, y_test = X[p][train_len:,:], y[p][train_len:,:]
        scaler = MinMaxScaler()
        scaler.fit(X_test.transpose())
        X_test = scaler.transform(X_test.transpose()).transpose()

        if to_tensor:
            return torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_test), torch.from_numpy(y_test)
        return X_train, y_train, X_test, y_test

def Var(X_train, y_train, x_test, y_test):
    return Variable(X_train), Variable(y_train), Variable(x_test), Variable(y_test) 

def typeset(X_train, y_train, x_test, y_test):
    X_train = X_train.to(torch.float)
    y_train = y_train.to(torch.float)
    x_test = x_test.to(torch.float)
    y_test = y_test.to(torch.float)
    return X_train, y_train, x_test, y_test


 
if __name__ == "__main__":
    symbols = [abbrevations[k] for k in abbrevations]
    print(symbols)
    d = WindowTokenizerParser(30, 5).get_stocks(["AMZN", "AAPL", "QBTL", "DISC", "STPL"])
    print(d, file=open("tasks.txt", "wt"))
    X, y = WindowTokenizerParser.unzip(d)
    print(X, y)
    print(len(X), len(y))