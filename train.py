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

other_features = pd.read_csv('prices.csv', header=0)

# TICKER SELECTION
# ================
ticker = 'YHOO' # parameter
ticker_data = df[df['symbol'] == ticker]
ticker_data_feats = other_features[other_features['symbol'] == ticker]
# the output variable
stock_price = ticker_data.close.values.astype('float32')
stock_price = stock_price.reshape(len(stock_price), 1)

plt.plot(stock_price)
plt.title(ticker)
plt.show()

features = list(set(ticker_data_feats.columns) - set(['date', 'symbol']))


# SCALE
# =====
scaler = MinMaxScaler(feature_range=(0, 1))
stock_price = scaler.fit_transform(stock_price)

feature_scaler = StandardScaler()
ticker_data_feats = feature_scaler.fit_transform(ticker_data_feats[features].values)

# SPLIT
# =====
split_rate = 0.8
train_size = int(len(stock_price) * split_rate)
test_size = len(stock_price) - train_size
train, test = stock_price[0:train_size, :], stock_price[train_size:len(stock_price), :]
train2, test2 = ticker_data_feats[0:train_size, :], ticker_data_feats[train_size:len(stock_price), :]

# LOOK BACK
# =========
# reshape into X=t and Y=t+1
look_back = 5
lag = 2
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
model.summary()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# OUTPUT ROUTINE
# ==============
yhat = model.predict([testX, test2])
prob_out = model.predict_proba([testX, test2])
ts = pd.DataFrame(test[look_back+1:], columns=['actual'])
ts['pred'] = scaler.inverse_transform(yhat.ravel())
ts['actual'] = scaler.inverse_transform(ts['actual'])
print('RMSE: ', (abs(ts['pred'] - ts['actual'])/ts['actual']).mean() ) #0.032
plt.plot(ts)
ts.to_csv('res_out.csv')

# SAVE MODEL
# ==========
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

