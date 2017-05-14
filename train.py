# LIBS
# ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import create_dataset, create_model, output_routine, save_model_to_disk

# PARAMETERS
# ==========
ticker = 'YHOO'         # which ticker to use for learning
split_rate = 0.8
look_back = 5           # LSTM lookback
lag = 2                 # based on yesterday's prices, predict tomorrow's (no info on today's close)
n_batch_size = 32       # model's hyperparameter
n_epochs = 50           # model's hyperparameter
learning_rate = 0.0001

# IMPORT DATA
# ===========
df = pd.read_csv('prices.csv', header=0)
other_features = pd.read_csv('prices.csv', header=0)

# TICKER SELECTION
# ================
ticker_data = df[df['symbol'] == ticker]
ticker_data_feats = other_features[other_features['symbol'] == ticker]
# the output variable
stock_price = ticker_data.close.values.astype('float32')
stock_price = stock_price.reshape(len(stock_price), 1)

# plot the stock's history
plt.plot(stock_price)
plt.title(ticker)
plt.show()

features = list(set(ticker_data_feats.columns) - set(['date', 'symbol']))


# SCALE FEATURES AND OUTPUT
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
stock_price = scaler.fit_transform(stock_price)

feature_scaler = StandardScaler()
ticker_data_feats = feature_scaler.fit_transform(ticker_data_feats[features].values)

# SPLIT
# =====
train_size = int(len(stock_price) * split_rate)
test_size = len(stock_price) - train_size
train, test = stock_price[0:train_size, :], stock_price[train_size:len(stock_price), :]
train2, test2 = ticker_data_feats[0:train_size, :], ticker_data_feats[train_size:len(stock_price), :]

# LOOK BACK
# =========
trainX, trainY = create_dataset(train, look_back, lag=lag)
testX, testY = create_dataset(test, look_back)

# RESHAPE LSTM STYLE
# ==================
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# MODEL TRAINING
# ==============
model = create_model(1, len(features), look_back, lr=learning_rate)

history = model.fit([trainX, train2[look_back+1:]], trainY, batch_size=n_batch_size,
                    epochs=n_epochs, validation_split=0.1, verbose=2)
model.summary()

# plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# OUTPUT ROUTINE
# ==============
output_routine(test, testX, test2, model, scaler, look_back, plt, pd)


# SAVE MODEL
# ==========
save_model_to_disk(model)


