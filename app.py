# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from flask import Flask, render_template,url_for,request
import pandas as pd

import pickle

loaded_AQIxgmodel=pickle.load(open('RandomforestAQI.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("real_2015.csv")
    my_predict = loaded_AQIxgmodel.predict(df.iloc[:,:-1].values)
    my_predict = my_predict.tolist()
    return render_template('result.html',prediction=my_predict)


if __name__ == '__main__':
    app.run(debug=True)
    
