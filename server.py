from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('charts.html')

@app.route('/marketPrice')
def market_price():
    return render_template('markets.html')


# This function will use the name of the stock and will return the evolution over time
@app.route('/chart', methods=['GET'])
def chart():
    args = request.args
    return render_template(template_name_or_list='charts.html', 
                        time = [1, 2, 3, 4],
                        price = [10, 9, 7, 11],
                        plot_stock_symbol = args.get("name", default = "AMZN", type=str))

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=5000, debug=True)