import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils import create_dataset, create_model

df = pd.read_csv('prices.csv', header=0)

yahoo = df[df['symbol']=='YHOO']
stock_price = yahoo.close.values.astype('float32')
stock_price = stock_price.reshape(1762, 1)

plt.plot(stock_price)
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
stock_price = scaler.fit_transform(stock_price)

train_size = int(len(stock_price) * 0.80)
test_size = len(stock_price) - train_size
train, test = stock_price[0:train_size,:], stock_price[train_size:len(stock_price),:]

# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = create_model(1, 10, lr=0.0001)

history = model.fit(trainX, trainY, batch_size=32, epochs=30, validation_split=0.1, verbose=2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# OUTPUT ROUTINE
# ==============
yhat = model.predict(testX)

ts = pd.DataFrame(test[look_back+1:], columns=['actual'])
ts['pred'] = scaler.inverse_transform(yhat.ravel())
ts['actual'] = scaler.inverse_transform(ts['actual'])

ts.to_csv('res_out.csv')

