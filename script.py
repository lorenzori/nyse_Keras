# LIBS
# ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import create_dataset, create_model

# IMPORT DATA
# ===========
df = pd.read_csv('prices.csv', header=0)

other = pd.read_csv('prices.csv', header=0)

# TICKER SELECTION
# ================
ticker = 'YHOO'
yahoo = df[df['symbol'] == ticker]
yahoo_fu = other[other['symbol'] == ticker]

stock_price = yahoo.close.values.astype('float32')
stock_price = stock_price.reshape(len(stock_price), 1)

plt.plot(stock_price)
plt.title(ticker)
plt.show()

features = list(set(yahoo_fu.columns) - set(['date', 'symbol']))


# SCALE
# =====
scaler = MinMaxScaler(feature_range=(0, 1))
stock_price = scaler.fit_transform(stock_price)

feature_scaler = StandardScaler()
yahoo_fu = feature_scaler.fit_transform(yahoo_fu[features].values)

# SPLIT
# =====
train_size = int(len(stock_price) * 0.80)
test_size = len(stock_price) - train_size
train, test = stock_price[0:train_size, :], stock_price[train_size:len(stock_price), :]
train2, test2 = yahoo_fu[0:train_size, :], yahoo_fu[train_size:len(stock_price), :]

# LOOK BACK
# =========
# reshape into X=t and Y=t+1
look_back = 20
lag = 5
trainX, trainY = create_dataset(train, look_back, lag=lag)
testX, testY = create_dataset(test, look_back)

# RESHAPE LSTM STYLE
# ==================
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# MODEL
# =====
model = create_model(1, len(features), look_back, lr=0.0001)

history = model.fit([trainX, train2[look_back+1:]], trainY, batch_size=32, epochs=50, validation_split=0.1, verbose=2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# OUTPUT ROUTINE
# ==============
yhat = model.predict([testX, test2])

ts = pd.DataFrame(test[look_back+1:], columns=['actual'])
ts['pred'] = scaler.inverse_transform(yhat.ravel())
ts['actual'] = scaler.inverse_transform(ts['actual'])
print('RMSE: ', (abs(ts['pred'] - ts['actual'])/ts['actual']).mean() ) #0.044
plt.plot(ts)
ts.to_csv('res_out.csv')

