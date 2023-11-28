import requests
import os
from pyspark import SparkContext, sql
import pyspark
# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo'
# r = requests.get(url)
# data = r.json()

# print(data)

class StockPricesAPI():
    GeneralURL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval={}min&apikey=demo&apikey='
    spark = sql.SparkSession\
        .builder.appName("Python Spark SQL").getOrCreate()
    def __init__(self):
        None
        
    def getPrices(symbol, time_interval=5):
        r = requests.get(StockPricesAPI.GeneralURL.format(symbol, time_interval))
        print({
            "symbol": symbol,
            "times" : r.json()['Time Series (5min)']
            }, file = open("request_ans.json", "wt"))
        myJson = StockPricesAPI.spark.read.json("request_ans.json")
        # myJson.show()
        # myJson.printSchema()
        
    def showPrices(symbol):
        None
        
StockPricesAPI.getPrices("AMZN")
        