# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1, lag=1):
    """
    :param dataset:
    :param look_back:
    :param lag:
    :return:
    """
    import numpy as np
    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back + (-lag+1), 0])
    return np.array(dataX), np.array(dataY)


def create_model(feats, feats2, lookback, lr=0.001):
    """
    :param feats: features for LSTM
    :param feats2: features for MLP
    :param lookback:
    :param lr:
    :return:
    """

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

def output_routine(test, testX, test2, model, scaler, look_back, plt, pd):
    """
    :param testX:
    :param test2:
    :param model:
    :return:
    """

    yhat = model.predict([testX, test2])

    ts = pd.DataFrame(test[look_back+1:], columns=['actual'])
    ts['pred'] = scaler.inverse_transform(yhat.ravel())
    ts['actual'] = scaler.inverse_transform(ts['actual'])
    print('RMSE: ', (abs(ts['pred'] - ts['actual'])/ts['actual']).mean() )
    plt.plot(ts)
    ts.to_csv('res_out.csv')


def save_model_to_disk(model):
    """
    :param model:
    :return:
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model_from_disk(model_from_json):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    return loaded_model
    print("Loaded model from disk")