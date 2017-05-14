# APP ROUTINE
# ===========
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result = request.form

    print('request received:', result['TICKER'])

    # LIBRARIES
    # =========
    from keras.models import model_from_json
    from keras.optimizers import Adam
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from utils import create_dataset
    import numpy as np
    from dateutil.relativedelta import relativedelta
    import datetime

    # load json and create model
    # ==========================
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    A = Adam(lr=0.0001)
    loaded_model.compile(loss='mse', optimizer=A)

    # get the data
    # ============
    print('loading data')
    df = pd.read_csv('prices.csv', header=0)
    other_features = pd.read_csv('prices.csv', header=0)

    # TICKER SELECTION
    # ================
    ticker = result['TICKER']

    print('TICKER: ', ticker)

    if ticker not in df.symbol.unique():
        print('no ticker {}!'.format(ticker))
        exit()

    ticker_data = df[df['symbol'] == ticker]
    ticker_data_feats = other_features[other_features['symbol'] == ticker]
    # the output variable
    stock_price = ticker_data.close.values.astype('float32')
    stock_price = stock_price.reshape(len(stock_price), 1)

    features = list(set(ticker_data_feats.columns) - set(['date', 'symbol']))

    # SCALE
    # =====
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_price = scaler.fit_transform(stock_price)

    feature_scaler = StandardScaler()
    ticker_data_feats = feature_scaler.fit_transform(ticker_data_feats[features].values)

    # LOOK BACK
    # =========
    # reshape into X=t and Y=t+1
    look_back = 5
    lag = 2
    X, Y = create_dataset(stock_price, look_back, lag=lag)

    # RESHAPE LSTM STYLE
    # ==================
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # PREDICT
    # =====
    yhat = loaded_model.predict([X, ticker_data_feats])

    # OUTPUT DATA
    # ===========
    ts = pd.DataFrame(stock_price[look_back+1:], columns=['actual'])
    ts['ticker'] = ticker
    ts['pred'] = scaler.inverse_transform(yhat.ravel())
    ts['actual'] = scaler.inverse_transform(ts['actual'])
    ts['probability'] = loaded_model.predict_proba([X, ticker_data_feats]) * 100
    ts['date'] = ticker_data[look_back+1:]['date'].values
    ts['error'] = abs(ts['pred'] - ts['actual'])/ts['actual']
    print('RMSE: ', (abs(ts['pred'] - ts['actual'])/ts['actual']).mean() ) #0.032


    try:
        output = output.append(ts)
    except NameError:
        output = pd.DataFrame(ts)

    output.to_csv('scoring_out.csv', index=False)

    # SEND VALUES
    # ===========
    OUT_DATE = "{:%Y-%m-%d}".format(datetime.datetime.strptime(result['OUT_DATE'], "%Y-%m-%d")
                                    + relativedelta(days=1))
    PREV_DATE = "{:%Y-%m-%d}".format(datetime.datetime.strptime(result['OUT_DATE'], "%Y-%m-%d"))

    prev_value = output[output.date == PREV_DATE]['actual'].values
    # just over the weekend
    if prev_value < 1:
        print('weekend', prev_value)
        PREV_DATE = "{:%Y-%m-%d}".format(datetime.datetime.strptime(result['OUT_DATE'], "%Y-%m-%d")
                                         - relativedelta(days=3))
        prev_value = output[output.date == PREV_DATE]['actual'].values

    app_result = {'OUT_DATE': OUT_DATE}
    app_result['OUT_VALUE'] = output[output.date == OUT_DATE]['pred']
    app_result['OUT_DELTA'] = prev_value - \
                              output[output.date == OUT_DATE]['pred'].values


    return render_template('result.html', out_date=app_result['OUT_DATE'],
                           out_value=app_result['OUT_VALUE'].values,
                           out_delta=app_result['OUT_DELTA'])

if __name__ == '__main__':
	app.run()