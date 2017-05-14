from keras.models import model_from_json
from keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import create_dataset, load_model_from_disk
import numpy as np

# PARAMETERS
# ==========
tikkkers = ['YHOO', 'JPM', 'WU', 'ACN']         # which ticker to score
look_back = 5           # LSTM lookback
lag = 2                 # based on yesterday's prices, predict tomorrow's (no info on today's close)
learning_rate = 0.0001

# load network, weights and compile model
# =======================================
try:
    loaded_model = load_model_from_disk(model_from_json)
    A = Adam(lr=learning_rate)
    loaded_model.compile(loss='mse', optimizer=A)

except FileNotFoundError:
    print('no model found!')
    exit()

# get the data
# ============
df = pd.read_csv('prices.csv', header=0)
other_features = pd.read_csv('prices.csv', header=0)

# TICKER SELECTION
# ================
for ticker in tikkkers:

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
    X, Y = create_dataset(stock_price, look_back, lag=lag)

    # RESHAPE LSTM STYLE
    # ==================
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # SCORE
    # =====
    yhat = loaded_model.predict([X, ticker_data_feats])

    ts = pd.DataFrame(stock_price[look_back+1:], columns=['actual'])
    ts['ticker'] = ticker
    ts['pred'] = scaler.inverse_transform(yhat.ravel())
    ts['actual'] = scaler.inverse_transform(ts['actual'])
    ts['probability'] = loaded_model.predict_proba([X, ticker_data_feats]) * 100
    ts['date'] = ticker_data[look_back+1:]['date'].values
    ts['error'] = abs(ts['pred'] - ts['actual'])/ts['actual']
    print('RMSE: ', (abs(ts['pred'] - ts['actual'])/ts['actual']).mean() )


    try:
        output = output.append(ts)
    except NameError:
        output = pd.DataFrame(ts)

output.to_csv('scoring_out.csv', index=False)

