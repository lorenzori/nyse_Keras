# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1, lag=1):
    import numpy as np
    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back + (-lag+1), 0])
    return np.array(dataX), np.array(dataY)


def create_model(feats, feats2, lookback, lr=0.001):

    from keras.layers.core import Dense, Activation, Dropout
    from keras.optimizers import Adam
    from keras.layers.recurrent import LSTM
    from keras.layers import Merge
    from keras.models import Sequential
    import time
    # Step 2 Build Model
    branch1 = Sequential()

    branch1.add(LSTM(input_shape=(feats, lookback), output_dim=10, return_sequences=True))
    branch1.add(Dropout(0.2))
    branch1.add(LSTM(50, return_sequences=False))
    branch1.add(Dropout(0.2))
    branch1.add(Dense(10, activation='linear'))

    branch2 = Sequential()
    branch2.add(Dense(10, input_shape=(feats2,), init='normal', activation='relu'))
    branch2.add(Dense(10, activation='relu'))

    model = Sequential()
    model.add(Merge([branch1, branch2], mode='sum'))
    model.add(Dense(1, init='normal', activation='sigmoid'))


    start = time.time()
    A = Adam(lr=lr)
    model.compile(loss='mse', optimizer=A)
    print('compilation time : ', time.time() - start)

    return model