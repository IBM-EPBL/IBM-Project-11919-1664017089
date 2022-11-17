# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

import urllib.request
import urllib.parse
import socket    
import csv

import matplotlib.dates as mdates
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="rainfall_prediction"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

   
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('index.html',msg=msg)



@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""
    pid=""

   
            
        
    return render_template('admin.html',msg=msg)

@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    msg=""
    
    weather_df = pd.read_csv("static/dataset/weather_train.csv")
    print(weather_df.shape)
    dat=weather_df.head(200)

    data=[]
    for ss in dat.values:
        data.append(ss)
    

    return render_template('load_data.html',data=data)


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    msg=""

    weather_df = pd.read_csv("static/dataset/weather_train.csv")
    print(weather_df.shape)
    weather_df.head()

    print(weather_df.isnull().sum())

    unknown_weather_df = pd.read_csv("static/dataset/weather_test.csv")
    print(unknown_weather_df.shape)
    unknown_weather_df.head()

    #unknown_weather_df.info()

    print(unknown_weather_df.isnull().sum())

    updated_weather_df = weather_df
    updated_weather_df = updated_weather_df.drop(['row ID'], axis = 1)
    updated_weather_df['MinTemp']=updated_weather_df['MinTemp'].fillna(updated_weather_df['MinTemp'].mean())
    updated_weather_df['MaxTemp']=updated_weather_df['MaxTemp'].fillna(updated_weather_df['MaxTemp'].mean())
    updated_weather_df['Rainfall']=updated_weather_df['Rainfall'].fillna(updated_weather_df['Rainfall'].mean())
    updated_weather_df['Evaporation']=updated_weather_df['Evaporation'].fillna(updated_weather_df['Evaporation'].mean())
    updated_weather_df['Sunshine']=updated_weather_df['Sunshine'].fillna(updated_weather_df['Sunshine'].mean())
    updated_weather_df['WindGustSpeed']=updated_weather_df['WindGustSpeed'].fillna(updated_weather_df['WindGustSpeed'].mean())
    updated_weather_df['WindSpeed9am']=updated_weather_df['WindSpeed9am'].fillna(updated_weather_df['WindSpeed9am'].mean())
    updated_weather_df['WindSpeed3pm']=updated_weather_df['WindSpeed3pm'].fillna(updated_weather_df['WindSpeed3pm'].mean())
    updated_weather_df['Humidity9am']=updated_weather_df['Humidity9am'].fillna(updated_weather_df['Humidity9am'].mean())
    updated_weather_df['Humidity3pm']=updated_weather_df['Humidity3pm'].fillna(updated_weather_df['Humidity3pm'].mean())
    updated_weather_df['Pressure9am']=updated_weather_df['Pressure9am'].fillna(updated_weather_df['Pressure9am'].mean())
    updated_weather_df['Pressure3pm']=updated_weather_df['Pressure3pm'].fillna(updated_weather_df['Pressure3pm'].mean())
    updated_weather_df['Cloud9am']=updated_weather_df['Cloud9am'].fillna(updated_weather_df['Cloud9am'].mean())
    updated_weather_df['Cloud3pm']=updated_weather_df['Cloud3pm'].fillna(updated_weather_df['Cloud3pm'].mean())
    updated_weather_df['Temp9am']=updated_weather_df['Temp9am'].fillna(updated_weather_df['Temp9am'].mean())
    updated_weather_df['Temp3pm']=updated_weather_df['Temp3pm'].fillna(updated_weather_df['Temp3pm'].mean())
    print(updated_weather_df.isnull().sum())

    updated_weather_df['WindGustDir'].value_counts()

    updated_weather_df['WindDir9am'].value_counts()

    updated_weather_df['WindGustDir']=updated_weather_df['WindGustDir'].fillna(updated_weather_df['WindGustDir'].value_counts().idxmax())
    updated_weather_df['WindDir9am']=updated_weather_df['WindDir9am'].fillna(updated_weather_df['WindDir9am'].value_counts().idxmax())
    updated_weather_df['WindDir3pm']=updated_weather_df['WindDir3pm'].fillna(updated_weather_df['WindDir3pm'].value_counts().idxmax())
    print(updated_weather_df.isnull().sum())

    updated_weather_df['RainToday'] = updated_weather_df['RainToday'].fillna(updated_weather_df['RainTomorrow'].shift())
    print(updated_weather_df.isnull().sum())

    updated_weather_df.loc[updated_weather_df.RainToday == "Yes", "RainToday"] = 1
    updated_weather_df.loc[updated_weather_df.RainToday == "No", "RainToday"] = 0
    updated_weather_df['RainToday'] = updated_weather_df['RainToday'].astype(int)
    updated_weather_df.head()

    #Pre-processing of Unknown Weather Data
    updated_unknown_weather_df = unknown_weather_df
    updated_unknown_weather_df = updated_unknown_weather_df.drop(['row ID'], axis = 1)
    updated_unknown_weather_df['MinTemp']=updated_unknown_weather_df['MinTemp'].fillna(updated_unknown_weather_df['MinTemp'].mean())
    updated_unknown_weather_df['MaxTemp']=updated_unknown_weather_df['MaxTemp'].fillna(updated_unknown_weather_df['MaxTemp'].mean())
    updated_unknown_weather_df['Rainfall']=updated_unknown_weather_df['Rainfall'].fillna(updated_unknown_weather_df['Rainfall'].mean())
    updated_unknown_weather_df['Evaporation']=updated_unknown_weather_df['Evaporation'].fillna(updated_unknown_weather_df['Evaporation'].mean())
    updated_unknown_weather_df['Sunshine']=updated_unknown_weather_df['Sunshine'].fillna(updated_unknown_weather_df['Sunshine'].mean())
    updated_unknown_weather_df['WindGustSpeed']=updated_unknown_weather_df['WindGustSpeed'].fillna(updated_unknown_weather_df['WindGustSpeed'].mean())
    updated_unknown_weather_df['WindSpeed9am']=updated_unknown_weather_df['WindSpeed9am'].fillna(updated_unknown_weather_df['WindSpeed9am'].mean())
    updated_unknown_weather_df['WindSpeed3pm']=updated_unknown_weather_df['WindSpeed3pm'].fillna(updated_unknown_weather_df['WindSpeed3pm'].mean())
    updated_unknown_weather_df['Humidity9am']=updated_unknown_weather_df['Humidity9am'].fillna(updated_unknown_weather_df['Humidity9am'].mean())
    updated_unknown_weather_df['Humidity3pm']=updated_unknown_weather_df['Humidity3pm'].fillna(updated_unknown_weather_df['Humidity3pm'].mean())
    updated_unknown_weather_df['Pressure9am']=updated_unknown_weather_df['Pressure9am'].fillna(updated_unknown_weather_df['Pressure9am'].mean())
    updated_unknown_weather_df['Pressure3pm']=updated_unknown_weather_df['Pressure3pm'].fillna(updated_unknown_weather_df['Pressure3pm'].mean())
    updated_unknown_weather_df['Cloud9am']=updated_unknown_weather_df['Cloud9am'].fillna(updated_unknown_weather_df['Cloud9am'].mean())
    updated_unknown_weather_df['Cloud3pm']=updated_unknown_weather_df['Cloud3pm'].fillna(updated_unknown_weather_df['Cloud3pm'].mean())
    updated_unknown_weather_df['Temp9am']=updated_unknown_weather_df['Temp9am'].fillna(updated_unknown_weather_df['Temp9am'].mean())
    updated_unknown_weather_df['Temp3pm']=updated_unknown_weather_df['Temp3pm'].fillna(updated_unknown_weather_df['Temp3pm'].mean())
    print(updated_unknown_weather_df.isnull().sum())


    ##
    list_of_column_names=[]
    with open("static/dataset/weather_train.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        list_of_column_names = []
        for row in csv_reader:
            list_of_column_names.append(row)
            break
    ##

    print(list_of_column_names)
 
    dat4=updated_unknown_weather_df.isnull().sum()
    dr=np.stack(dat4)
    print(dr)
    
    data4=[]
    i=0
    for ss4 in dr:
        dt=[]
        dt.append(list_of_column_names[0][i])
        dt.append(ss4)
        data4.append(dt)
        i+=1

    updated_unknown_weather_df['WindGustDir']=updated_unknown_weather_df['WindGustDir'].fillna(updated_unknown_weather_df['WindGustDir'].value_counts().idxmax())
    updated_unknown_weather_df['WindDir9am']=updated_unknown_weather_df['WindDir9am'].fillna(updated_unknown_weather_df['WindDir9am'].value_counts().idxmax())
    updated_unknown_weather_df['WindDir3pm']=updated_unknown_weather_df['WindDir3pm'].fillna(updated_unknown_weather_df['WindDir3pm'].value_counts().idxmax())
    print(updated_unknown_weather_df.isnull().sum())

    updated_unknown_weather_df=updated_unknown_weather_df.dropna()
    #print(updated_unknown_weather_df.isnull().sum())

    updated_unknown_weather_df.loc[updated_unknown_weather_df.RainToday == "Yes", "RainToday"] = 1
    updated_unknown_weather_df.loc[updated_unknown_weather_df.RainToday == "No", "RainToday"] = 0
    updated_unknown_weather_df['RainToday'] = updated_unknown_weather_df['RainToday'].astype(int)
    dat2=updated_unknown_weather_df
    dat1=updated_unknown_weather_df.head(200)
    
    rows=len(dat2.values)
    cnt=0
    data=[]
    for ss3 in dat1.values:
        cnt=len(ss3)
        data.append(ss3)
    cols=cnt
    mem=float(rows)*0.75


    return render_template('preprocess.html',data4=data4,data=data,rows=rows,cols=cols,mem=mem)



@app.route('/data_analysis', methods=['GET', 'POST'])
def data_analysis():
    msg=""

    weather_df = pd.read_csv("static/dataset/weather_train.csv")
    print(weather_df.shape)
    weather_df.head()

    print(weather_df.isnull().sum())

    unknown_weather_df = pd.read_csv("static/dataset/weather_test.csv")
    print(unknown_weather_df.shape)
    unknown_weather_df.head()

    #unknown_weather_df.info()

    print(unknown_weather_df.isnull().sum())

    updated_weather_df = weather_df
    updated_weather_df = updated_weather_df.drop(['row ID'], axis = 1)
    updated_weather_df['MinTemp']=updated_weather_df['MinTemp'].fillna(updated_weather_df['MinTemp'].mean())
    updated_weather_df['MaxTemp']=updated_weather_df['MaxTemp'].fillna(updated_weather_df['MaxTemp'].mean())
    updated_weather_df['Rainfall']=updated_weather_df['Rainfall'].fillna(updated_weather_df['Rainfall'].mean())
    updated_weather_df['Evaporation']=updated_weather_df['Evaporation'].fillna(updated_weather_df['Evaporation'].mean())
    updated_weather_df['Sunshine']=updated_weather_df['Sunshine'].fillna(updated_weather_df['Sunshine'].mean())
    updated_weather_df['WindGustSpeed']=updated_weather_df['WindGustSpeed'].fillna(updated_weather_df['WindGustSpeed'].mean())
    updated_weather_df['WindSpeed9am']=updated_weather_df['WindSpeed9am'].fillna(updated_weather_df['WindSpeed9am'].mean())
    updated_weather_df['WindSpeed3pm']=updated_weather_df['WindSpeed3pm'].fillna(updated_weather_df['WindSpeed3pm'].mean())
    updated_weather_df['Humidity9am']=updated_weather_df['Humidity9am'].fillna(updated_weather_df['Humidity9am'].mean())
    updated_weather_df['Humidity3pm']=updated_weather_df['Humidity3pm'].fillna(updated_weather_df['Humidity3pm'].mean())
    updated_weather_df['Pressure9am']=updated_weather_df['Pressure9am'].fillna(updated_weather_df['Pressure9am'].mean())
    updated_weather_df['Pressure3pm']=updated_weather_df['Pressure3pm'].fillna(updated_weather_df['Pressure3pm'].mean())
    updated_weather_df['Cloud9am']=updated_weather_df['Cloud9am'].fillna(updated_weather_df['Cloud9am'].mean())
    updated_weather_df['Cloud3pm']=updated_weather_df['Cloud3pm'].fillna(updated_weather_df['Cloud3pm'].mean())
    updated_weather_df['Temp9am']=updated_weather_df['Temp9am'].fillna(updated_weather_df['Temp9am'].mean())
    updated_weather_df['Temp3pm']=updated_weather_df['Temp3pm'].fillna(updated_weather_df['Temp3pm'].mean())
    print(updated_weather_df.isnull().sum())

    updated_weather_df['WindGustDir'].value_counts()

    updated_weather_df['WindDir9am'].value_counts()

    updated_weather_df['WindGustDir']=updated_weather_df['WindGustDir'].fillna(updated_weather_df['WindGustDir'].value_counts().idxmax())
    updated_weather_df['WindDir9am']=updated_weather_df['WindDir9am'].fillna(updated_weather_df['WindDir9am'].value_counts().idxmax())
    updated_weather_df['WindDir3pm']=updated_weather_df['WindDir3pm'].fillna(updated_weather_df['WindDir3pm'].value_counts().idxmax())
    print(updated_weather_df.isnull().sum())

    updated_weather_df['RainToday'] = updated_weather_df['RainToday'].fillna(updated_weather_df['RainTomorrow'].shift())
    print(updated_weather_df.isnull().sum())

    updated_weather_df.loc[updated_weather_df.RainToday == "Yes", "RainToday"] = 1
    updated_weather_df.loc[updated_weather_df.RainToday == "No", "RainToday"] = 0
    updated_weather_df['RainToday'] = updated_weather_df['RainToday'].astype(int)
    updated_weather_df.head()

    #Pre-processing of Unknown Weather Data
    updated_unknown_weather_df = unknown_weather_df
    updated_unknown_weather_df = updated_unknown_weather_df.drop(['row ID'], axis = 1)
    updated_unknown_weather_df['MinTemp']=updated_unknown_weather_df['MinTemp'].fillna(updated_unknown_weather_df['MinTemp'].mean())
    updated_unknown_weather_df['MaxTemp']=updated_unknown_weather_df['MaxTemp'].fillna(updated_unknown_weather_df['MaxTemp'].mean())
    updated_unknown_weather_df['Rainfall']=updated_unknown_weather_df['Rainfall'].fillna(updated_unknown_weather_df['Rainfall'].mean())
    updated_unknown_weather_df['Evaporation']=updated_unknown_weather_df['Evaporation'].fillna(updated_unknown_weather_df['Evaporation'].mean())
    updated_unknown_weather_df['Sunshine']=updated_unknown_weather_df['Sunshine'].fillna(updated_unknown_weather_df['Sunshine'].mean())
    updated_unknown_weather_df['WindGustSpeed']=updated_unknown_weather_df['WindGustSpeed'].fillna(updated_unknown_weather_df['WindGustSpeed'].mean())
    updated_unknown_weather_df['WindSpeed9am']=updated_unknown_weather_df['WindSpeed9am'].fillna(updated_unknown_weather_df['WindSpeed9am'].mean())
    updated_unknown_weather_df['WindSpeed3pm']=updated_unknown_weather_df['WindSpeed3pm'].fillna(updated_unknown_weather_df['WindSpeed3pm'].mean())
    updated_unknown_weather_df['Humidity9am']=updated_unknown_weather_df['Humidity9am'].fillna(updated_unknown_weather_df['Humidity9am'].mean())
    updated_unknown_weather_df['Humidity3pm']=updated_unknown_weather_df['Humidity3pm'].fillna(updated_unknown_weather_df['Humidity3pm'].mean())
    updated_unknown_weather_df['Pressure9am']=updated_unknown_weather_df['Pressure9am'].fillna(updated_unknown_weather_df['Pressure9am'].mean())
    updated_unknown_weather_df['Pressure3pm']=updated_unknown_weather_df['Pressure3pm'].fillna(updated_unknown_weather_df['Pressure3pm'].mean())
    updated_unknown_weather_df['Cloud9am']=updated_unknown_weather_df['Cloud9am'].fillna(updated_unknown_weather_df['Cloud9am'].mean())
    updated_unknown_weather_df['Cloud3pm']=updated_unknown_weather_df['Cloud3pm'].fillna(updated_unknown_weather_df['Cloud3pm'].mean())
    updated_unknown_weather_df['Temp9am']=updated_unknown_weather_df['Temp9am'].fillna(updated_unknown_weather_df['Temp9am'].mean())
    updated_unknown_weather_df['Temp3pm']=updated_unknown_weather_df['Temp3pm'].fillna(updated_unknown_weather_df['Temp3pm'].mean())
    print(updated_unknown_weather_df.isnull().sum())

    updated_unknown_weather_df['WindGustDir']=updated_unknown_weather_df['WindGustDir'].fillna(updated_unknown_weather_df['WindGustDir'].value_counts().idxmax())
    updated_unknown_weather_df['WindDir9am']=updated_unknown_weather_df['WindDir9am'].fillna(updated_unknown_weather_df['WindDir9am'].value_counts().idxmax())
    updated_unknown_weather_df['WindDir3pm']=updated_unknown_weather_df['WindDir3pm'].fillna(updated_unknown_weather_df['WindDir3pm'].value_counts().idxmax())
    print(updated_unknown_weather_df.isnull().sum())

    updated_unknown_weather_df=updated_unknown_weather_df.dropna()
    #print(updated_unknown_weather_df.isnull().sum())

    updated_unknown_weather_df.loc[updated_unknown_weather_df.RainToday == "Yes", "RainToday"] = 1
    updated_unknown_weather_df.loc[updated_unknown_weather_df.RainToday == "No", "RainToday"] = 0
    updated_unknown_weather_df['RainToday'] = updated_unknown_weather_df['RainToday'].astype(int)
    updated_unknown_weather_df.head()
    

    
    ######
    #sns.displot(updated_weather_df, x="MinTemp", hue='RainToday', kde=True)
    #plt.title("Minimum Temperature Distribution", fontsize = 14)
    #plt.show()
    #plt.savefig("static/graph/graph1.png")
    #plt.close()
    ##
    #sns.displot(updated_weather_df, x="MaxTemp", hue='RainToday', kde=True)
    #plt.title("Maximum Temperature Distribution", fontsize = 14)
    #plt.show()
    #plt.savefig("static/graph/graph2.png")
    #plt.close()
    ###
    

    '''sns.displot(updated_weather_df, x="WindGustSpeed", hue='RainToday', kde=True)
    plt.title("Wind Gust Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="WindSpeed9am", hue='RainToday', kde=True)
    plt.title("WindSpeed at 9am Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="WindSpeed3pm", hue='RainToday', kde=True)
    plt.title("WindSpeed at 3pm Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="Humidity9am", hue='RainToday', kde=True)
    plt.title("Humidity at 9am Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="Humidity3pm", hue='RainToday', kde=True)
    plt.title("Humidity at 3pm Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="Pressure9am", hue='RainToday', kde=True)
    plt.title("Pressure at 9am Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="Pressure3pm", hue='RainToday', kde=True)
    plt.title("Pressure at 3pm Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="Cloud9am", hue='RainToday', kde=True)
    plt.title("Cloud at 9am Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="Cloud3pm", hue='RainToday', kde=True)
    plt.title("Cloud at 3pm Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="Temp9am", hue='RainToday', kde=True)
    plt.title("Temperature at 9am Distribution", fontsize = 14)
    plt.show()

    sns.displot(updated_weather_df, x="Temp3pm", hue='RainToday', kde=True)
    plt.title("Temperature at 3pm Distribution", fontsize = 14)
    plt.show()'''

    return render_template('data_analysis.html')




@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


