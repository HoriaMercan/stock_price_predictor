import requests
import os
from pyspark import SparkContext, sql
import numpy as np
# import yfinance as yf
import yahoo_fin as yf
from yahoo_fin.stock_info import get_data
import dotenv
# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo'
# r = requests.get(url)
# data = r.json()

# print(data)
my_dotenv = dotenv.load_dotenv(".env")
class StockPricesAPI():
    spark = sql.SparkSession\
        .builder.appName("Python Spark SQL").getOrCreate()
    def __init__(self):
        None
        
    def getPrices(symbol, start_date="12/03/2009", end_date="12/04/2021", interval="1d"):
        data = get_data("AMZN", start_date=start_date, end_date=end_date, index_as_date=False, interval=interval)
        
        # print(data[2:3])
        # print(data[-2:-1])
        N = len((data))
        
        __days = 20
        mean_matrix = (np.tri(N=N, k=__days) - np.tri(N=N, k = -(__days + 1))) / (2 * __days + 2)
        print(mean_matrix, file=open('matrix.out', "wt"))
        print(np.shape(np.array((list)(data['high'].to_numpy()))))
        print("My size {}".format(np.shape(mean_matrix)))
        
        all_highs = data["high"].to_numpy().dot( mean_matrix)
        all_lows = data["low"].to_numpy().dot(mean_matrix)
        all_dates = data['date'].to_numpy()
        
        
        
        # print(all_dates)
        print(all_highs)
        print(all_lows)
        print(all_dates)
        import matplotlib.pyplot as plt
        
        plt.plot(all_dates, all_lows)

        plt.plot(all_dates, all_highs)
        plt.show()
    def showPrices(symbol):
        None
        
StockPricesAPI.getPrices("AMZN")
        