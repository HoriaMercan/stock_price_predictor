from flask import Flask, render_template, request
import pandas as pd
from predict.ARIMA import ARIMA_MODEL
app = Flask(__name__)

@app.route('/')
def hello():
    args = request.args
    data = pd.read_csv("./predict/stocks.csv")
    symbols = data['Symbol'].tolist()
    name = data['Name'].tolist()
    
    print(symbols, name)

    return render_template(template_name_or_list='index.html',
                           symbols=symbols,
                           names=name,
                           size=len(symbols))
@app.route('/marketPrice')
def market_price():
    return render_template('markets.html')

from predict.LSTMStockPredictor import Predictor, StockPriceModel
from predict.financedata import FinanceData
# This function will use the name of the stock and will return the evolution over time

def getpred(symbol, path="./predict/mdlfull.t7", offset=19):
    print(symbol)
    feat, scale, unscale, real = FinanceData.getFeatures(stock=symbol, delta=offset, diff=False)
    if (len(feat) == 0):
        return "Error"
    model = StockPriceModel.load_from_path(path)
    # model.old = True
    first_parameter = next(model.parameters())
    input_shape = first_parameter.size()[1]
    print(model, input_shape)
    print("Feat", unscale(feat.detach().numpy())[0].tolist()[0])
    output = model(feat)
    print("Model " , output)
    result = unscale(output.detach().numpy())
    print("Unscaled: ", result)
    base = unscale(feat.detach().numpy())[0].tolist()[0]
    pr = result[0].tolist()
    re = real[:5].tolist()
    print("BASE PR RE")
    print(base, '\n\n', pr, '\n\n' , re)
    return base, pr, re

import torch
def getdiffpred(symbol, path="./predict/mdlbatched.t7", offset=19):
    
    print(symbol)
    feat, scale, unscale, real, start = FinanceData.getFeatures(stock=symbol, delta=offset, diff=True)
    if (len(feat) == 0):
        return "Error"
    print("KKDFDKSJSJKLFJ ", start, feat)
    model = StockPriceModel.load_from_path(path=path)
    # model.load_state_dict(torch.load("./predict/andumodelul-states.t7"))
    model.eval()
    detached_feat = feat.detach().numpy()[0][0].tolist()
    output = model(feat)
    print("DF bef ", start, detached_feat)

    detached_feat[0] += start
    for i in range(len(detached_feat)):
        if i > 0:
            detached_feat[i] = detached_feat[i] + detached_feat[i-1]
    base = unscale(detached_feat)
    print("\n\nDF", detached_feat, base)
    
    print("Model " , output)
    detached_out  = output.detach().numpy()
    last = detached_feat[-1]
    for i in range(len(detached_out)):
        if i > 0:
            detached_out[i] = detached_out[i] + detached_out[i-1]
        else:
            detached_out[i] = detached_out[i] + last
    result = unscale(detached_out)
    
    print("Unscaled: ", result)

    pr = result[0].tolist()
    re = real[:5].tolist()
    print(base, pr)
    return base.tolist(), pr, re

@app.route('/chart/<symbol>', methods=['GET'])
def chart(symbol):
    args = request.args
    # base2, pr2, re2 = getpred(symbol=symbol)
    base, pr, re = getdiffpred(symbol=symbol)

    print("BASE PR RE")
    print(base, '\n\n', pr, '\n\n' , re)
    return render_template(template_name_or_list='charts.html', 
                        time = [i for i in range(35)],
                        predicted = base + pr,
                        real = base + re,
                        predicted2 = base + pr,
                        real2 = base + re,
                        plot_stock_symbol = symbol)



@app.route('/arima/<symbol>')
def arima_chart(symbol):
    arima: ARIMA_MODEL = ARIMA_MODEL(symbol=symbol, logaritm=False)
    arima.determine_AR_and_MA(0.04, 0.045)
    data_train = arima.data_train.values
    data = arima.data.values
    predicted = arima.model().values
    total_len = len(data_train) + len(predicted)
    print('PREDICTED', predicted, 'REAL', data)
    THRESH = 40

    print("AR-I-MA", arima.AR, arima.I, arima.MA)
    return render_template(template_name_or_list='arima.html',
                           time=[i for i in range(total_len - THRESH, total_len)],
                           predicted=[
                               (list(data_train) + list(predicted))[i] for i in range(total_len - THRESH, total_len)
                           ],
                           real=[data[i] for i in range(total_len - THRESH, total_len)],
                           plot_stock_symbol=symbol)


@app.route('/auto_arima/<symbol>')
def auto_arima_chart(symbol):
    arima: ARIMA_MODEL = ARIMA_MODEL(symbol=symbol, logaritm=False)
    arima.determine_AR_and_MA(0.06, 0.06)
    data_train = arima.data_train.values
    data = arima.data.values
    predicted = arima.auto_model().values
    total_len = len(data_train) + len(predicted)
    THRESH = 40
    return render_template(template_name_or_list='arima.html',
                           time=[i for i in range(total_len - THRESH, total_len)],
                           predicted=[
                               (list(data_train) + list(predicted))[i] for i in range(total_len - THRESH, total_len)
                           ],
                           real=[data[i] for i in range(total_len - THRESH, total_len)],
                           plot_stock_symbol=symbol)


if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=5000, debug=True)