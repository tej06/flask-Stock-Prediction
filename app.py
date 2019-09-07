from __future__ import division, print_function
# coding=utf-8
import sys
import os, glob

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
#Adjusting the size of matplotlib
import matplotlib as mpl
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import io
import base64
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score  # Can't be used here as it is used only in case of classifications
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

app = Flask(__name__)

mpl.rc('figure', figsize=(8, 7))
# Adjusting the style of matplotlib
style.use('ggplot')

#temp start and end time

def get_data(stock, start, end):
    print("Get output")
    df = web.DataReader(stock, "yahoo", start, end)
    #df.set_index("Date", inplace=True)
    return df

def feature_engineer(dataframe):
    dfreg = dataframe.loc[:, ["Adj Close", "Volume"]]
    dfreg["HL_PCT"] = (dataframe["High"] - dataframe["Low"]) / dataframe["Close"] * 100.0
    dfreg["PCT_change"] = (dataframe["Close"] - dataframe["Open"]) / dataframe["Open"] * 100.0
    return dfreg

def preprocess_data(dataframe):
    dataframe.dropna(inplace=True)

    # cast out 1% of data for future prediction
    forecast_out = int(math.ceil(0.01*len(dataframe)))
    forecast_col = "Adj Close"
    dataframe["label"] = dataframe[forecast_col].shift(-forecast_out, axis=0) #axis=0 implies shift all columns

    X = np.array(dataframe.drop(["label"], axis=1))
    X = preprocessing.scale(X)

    X_lately = X[-forecast_out:]
    X_test = X[-(2*forecast_out):-forecast_out]
    X = X[:-forecast_out]

    y = np.array(dataframe['label'])
    y = y[:-forecast_out]
    y_test = np.array(dataframe[forecast_col])[-forecast_out:]

    return dataframe, X, y, X_test, y_test, X_lately

def train_model(regressor, X, y, X_test, y_test, hidden_size=(100,)):
    reg_model = None
    if regressor == "LinearRegression":
        reg_model = LinearRegression(n_jobs=-1)
    elif regressor == "PolynomialRegression":
        reg_model = make_pipeline(PolynomialFeatures(3), Ridge())
    elif regressor == "KNNRegression":
        reg_model = KNeighborsRegressor(n_neighbors=2)
    else: # regressor == "MLPRegression":
        reg_model = MLPRegressor()
    reg_model.fit(X, y)
    confidence_score = reg_model.score(X_test, y_test)
    return reg_model, confidence_score

def save_model(model, name):
    joblib.dump(model, "./"+name)

def load_model(name):
    model = joblib.load("./"+name)
    return model

def predict(reg_model, X_lately):
    forecast_set = reg_model.predict(X_lately)
    return forecast_set

def get_output_dataframe(dataframe, forecast_set):
    dataframe["Forecast"] = np.nan
    last_date = dataframe.iloc[-1].name
    #last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d")
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)
    for i in forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        #dataframe.loc[(str(next_date)[:11])] = [np.nan for _ in range(len(dataframe.columns)-1)]+[i]
        dataframe.loc[next_date] = [np.nan for _ in range(len(dataframe.columns)-1)]+[i]
    return dataframe, dataframe["Forecast"].tail(len(forecast_set))

def get_plot_prediction_url(dataframe):
    img = io.BytesIO()
    dataframe['Adj Close'].tail(500).plot()
    dataframe['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    graph_url = 'data:image/png;base64,{}'.format(graph_url)
    return graph_url

@app.route("/", methods=["GET"])
def index():
    #Index page
    return render_template("index.html")

@app.route("/query", methods=["GET", "POST"])
def submit_query():
    if request.method == "POST":
        plt.clf()
        plt.cla()
        #plt.close()
        stock = request.form["stock"]
        start = request.form["start-date-name"]
        end = request.form["end-date-name"]
        regressor = request.form["regressor"]
        start = list(map(int, start.split("-")))
        start = datetime.datetime(start[0], start[1], start[2])
        end = list(map(int, end.split("-")))
        end = datetime.datetime(end[0], end[1], end[2])
        print("Details: ", stock, start, end, regressor)
        dfmain = get_data(stock, start, end)
        dffeateng = feature_engineer(dfmain)
        dataframe, X, y, X_test, y_test, X_lately = preprocess_data(dffeateng)
        model, score = train_model("LinearRegression", X, y, X_test, y_test)
        forecast_set = predict(model, X_lately)
        dffinal, dfpred = get_output_dataframe(dataframe, forecast_set)
        graph_url = get_plot_prediction_url(dffinal)
        return graph_url
    return None


if __name__ == "__main__":
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
