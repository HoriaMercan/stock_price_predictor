# AMG Study on Stock Price Prediction


This is a stock price predictor using various methods created by second
year students Ariton Adrian, Ariton Alexandru, Gasan Carol-Luca and Mercan
Horia.

To predict the trends of the stocks we use the following 3 methods:
- LSTM network
- ARIMA
- Sentiment Analisys

The application is presented in 2 parts: Web and Jupyter Notebook.
Among the libraries we use are the following: 
- pytorch - for model training
- flask - for hosting
- pandas - for data parsing
- numpy - for quick mathematical calculations with tensors
- sklearn - for data processing (aka scaling)
- multiprocessing - for parralel data retrieval

# Running the APP

```
docker build -t pria .
docker run -p 5000:5000 pria
http://127.0.0.1
```

# Project Structure

```js
* predict - LSTM models and data parsing + ARIMA model
* templates - flask jinja templates
* text_analisys - text prediction module
    * database - data related operations
    * requests - input file query
* arima_test - arima ipynb
```

# Flask appplication

Our app is a web server written in Flask (used for routing and integrated with Jinja). All .html templates can be found in ./templates
It was very easy to integrate ML actions with the web interface. For efficiency, we cached the information retrievals for the ML models that are updated every day.
The app requests some popular libraries used for stock market prices (yfinance) and then the data are modeled by ML engines (torch - for LSTM and pmdarima - for Arima).
For all charts we used Charts.js engine developed in JavaScript (Bokeh alternative wasn't looking as good).

# LSTM

Using a range of models trained on scaled differences or values we manage
to accurately predict the trend of a stock.

```bash
python3 LSTMStockPredictor.py
```

The model consists of a LSTM layer follower by a Dropout, a linear network and another Dropout. It is trained on a dataset tokenized by a length 35 window by running 

# ARIMA model

The ARIMA models has 3 parameters (p, d, q), p represents the number of lags to be considered for the partial autocorelation part of the model ( AR - autoregressive ). q represents the number of lags to be taken into account for the autocorellation part of the model ( MA - moving average ).
d represents the number of times the dataset should be differentiated in order to make it stationary ( I - integrated ).

These three parts together constitute the ARIMA model.

## manual ARIMA

- uses the PACF (partial autocorellation) and ACF (autocorellation) to determine the optimal lag by choosing the first lag to go over the significance threshhold.

## auto-ARIMA

- determines the optimal parameters by analysing the dataset.




# Sentiment Analysis

We propose to determine a relation between recent news in the media and stock prices. In order to achieve the aforementioned, we need to gather data from a trustworthy source to overcome the cleaning process of fake news. We use FinViz to do so, a screener that provides instant data on prices and news associated with recognized stocks. To retrieve the data, there is no available API, but the actual structure of FinViz is very permisive and scrape-friendly. We also conducted different experiments to enlighten the correlation between the trendline and the sentiment expressed in the media.

# Text Interpreter

A named entity recognition model to idenitfy stock names in a natural language query (for example: "AAPL" in the query "predict me the apple stock for the following 3 days"), but with the huge quanitity of stocks whose names also represent correctly written english words such as "WHO", "AM" etc., the model only performed decently well on time entities.


## Other ideas for modelling: Using trends for building a better arima model

As future work, we want to integrate the sentiments extracted from the news in the ARIMA model as a mathematical distribution (a distribution defined by us, or determined experimentally). In [arima_test/test trends.ipynb] there is an implementation of trends retrieval and comparisons between price and trends for a company.
