from flask import Flask, render_template, request

from predict.ARIMA import ARIMA_MODEL

app = Flask(__name__)

@app.route('/')
def hello():
    args = request.args
    return render_template('index.html')
@app.route('/marketPrice')
def market_price():
    return render_template('markets.html')

from predict.LSTMStockPredictor import Predictor, StockPriceModel
from predict.financedata import FinanceData
# This function will use the name of the stock and will return the evolution over time
@app.route('/chart/<symbol>', methods=['GET'])
def chart(symbol):
    args = request.args
    print(symbol)
    feat, scale, unscale, real = FinanceData.getFeatures(stock=symbol, delta=20)
    if (len(feat) == 0):
        return "Error"
    model = StockPriceModel.load_from_path("./predict/mdlfull.t7")
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
    return render_template(template_name_or_list='charts.html', 
                        time = [i for i in range(35)],
                        predicted = base + pr,
                        real = base + re,
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