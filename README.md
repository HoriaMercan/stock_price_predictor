# AMG Stock Price Predictor


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

# LSTM

Using a range of models trained on scaled differences or values we manage
to accurately predict the trend of a stock.

The model consists of a LSTM layer follower by a Dropout, a linear network and another Dropout. It is trained on a dataset tokenized by a length 35 window by running 

```bash
python3 LSTMStockPredictor.py
```

# ARIMA

# Sentiment Analisys

# Text Interpreter