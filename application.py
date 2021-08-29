from flask import Flask, render_template , request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import numpy as np

app = Flask(__name__)
@app.route('/predict')
def predict():
    if request.method == "GET":
        return render_template('predict.html' )



@app.route('/result')
def result():
    if request.method == "GET":
        data = pd.read_csv('USA_housing.csv')
        x = data.drop('Price', axis=1)
        y = data['Price']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30)
        model = LinearRegression()
        model.fit(x_train, y_train)
        var1 = float(request.GET['n1'])
        var2 = float(request.GET['n2'])
        var3 = float(request.GET['n3'])
        var4 = float(request.GET['n4'])
        var5 = float(request.GET['n5'])

        pred = model.predict(np.array([var1, var2, var3, var4, var5]).reshape(1,-1))
        pred = round(pred[0])
        price = "The Predicted price is  $" + str(pred)

        return render_template('predict.html', {"result2": price})


@app.route('/')
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug = True)

