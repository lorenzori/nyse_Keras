# NYSE_Keras

using the NYSE dataset to predict stock prices in the short term. 

## Data
the dataset is publicly available on Kaggle: https://www.kaggle.com/dgawlik/nyse

## Model
the model is a Keras + Tensorflow network that merges together LSTM and MLP branches. Trained with the "training.py" script.

## Batch Scoring
Can score and forecast for all tickers with the "scoring.py".

## App
The "app.py" script embeds the scoring engine in a Flask application for web workflow.

## Requirments
following libraries are required:
- Python 3.x
- Tensorflow (or Theano)
- Keras
- Pandas
- Numpy
- Scikit-learn
- Flask
- Datetime
- Dateutil
- requests
