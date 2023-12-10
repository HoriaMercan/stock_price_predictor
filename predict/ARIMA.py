# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pandas
import matplotlib.pyplot as pyplot
import numpy as numpy
import pmdarima
import torch
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima.utils import ndiffs
import statsmodels.api as api

from financedata import WindowTokenizerParser, typeset, Var


class ARIMA_MODEL:
    data: DataFrame | TextFileReader
    data_train: DataFrame | TextFileReader
    data_test: DataFrame | TextFileReader
    AR: int = 0
    I: int = 0
    MA: int = 0

    def __init__(self, filename: str, logaritm: bool = False):
        self.data = pandas.read_csv(filename)['high']
        if logaritm:
            self.data = numpy.log(self.data)

        mask = (self.data.index < len(self.data) - 30)
        self.data_train = self.data[mask].copy()
        self.data_test = self.data[~mask].copy()
        self.I = ndiffs(self.data_train, test='adf')



    def plot_pacf_and_acf(self):
        data_copy = self.data_train.copy()
        for i in range(0, self.I):
            data_copy = data_copy.diff().dropna().copy()
        plotpacf = plot_pacf(data_copy)
        plotacf = plot_acf(data_copy)
        acf, ci_acf = api.tsa.acf(data_copy, alpha=0.05)
        pacf, ci_pacf = api.tsa.pacf(data_copy, alpha=0.05)
        pyplot.show()
        return acf, pacf, ci_acf, ci_pacf

    def determine_AR_and_MA(self, pacf_sign_level, acf_sign_level):
        acf, pacf, ci_acf, ci_pacf = self.plot_pacf_and_acf()
        # AR - parameter
        # se noteaza cu p si reprezinta numarul de spikeuri care sunt 'significante' in PACF aka nu intra in threshhold
        for i in range(1, len(pacf)): # nu l luam pe 0 in considerare
            current_pacf = pacf[i]
            # print(current_pacf)
            if not (current_pacf <= pacf_sign_level and -pacf_sign_level <= current_pacf):
                self.AR += 1

        for i in range(1, len(acf)): # nu l luam pe 0 in considerare, luam primul spike ce iese din bounds
            current_acf = acf[i]
            # print(current_acf)
            if not (current_acf <= acf_sign_level and -acf_sign_level <= current_acf):
                self.MA += 1


        # MA parameter

    def model(self):
        model = ARIMA(self.data_train, order=(self.AR,self.I,self.MA))
        model_fit = model.fit()
        print(model_fit.summary())
        residuals = model_fit.resid[1:]
        fig, ax = pyplot.subplots(1,2)
        residuals.plot(title="Res", ax=ax[0])
        residuals.plot(title="density", kind='kde', ax=ax[1])
        pyplot.show()

        forecast = model_fit.forecast(len(self.data_test))
        print(forecast)
        pyplot.plot(self.data)
        pyplot.plot([None] * len(self.data_train) + list(forecast))
        pyplot.show()



    def auto_model(self):
        model = pmdarima.auto_arima(self.data_train, stepwise=False, seasonal=True)
        model_fit = model
        print(model_fit.summary())

        forecast = model.predict(len(self.data_test))

        pyplot.plot(self.data)
        pyplot.plot([None] * len(self.data_train) + list(forecast))
        pyplot.show()


if __name__ == '__main__':
    arima = ARIMA_MODEL('Tesla stock price.csv', logaritm=False)
    print(f"MANUAL: ----- {arima.I} -----")
    arima.determine_AR_and_MA(0.06, 0.06)
    print(f"---- {arima.AR} ---- {arima.MA} ----")
    arima.auto_model()
    arima.model()



