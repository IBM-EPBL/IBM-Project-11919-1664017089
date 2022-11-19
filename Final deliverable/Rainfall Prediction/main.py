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

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

   
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM rf_register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('test_data'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    act=request.args.get("act")
    if request.method=='POST':
        name=request.form['name']
        location=request.form['location']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        
    
        
        mycursor = mydb.cursor()

        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
    
        mycursor.execute("SELECT count(*) from rf_register where uname=%s",(uname,))
        cnt = mycursor.fetchone()[0]
    
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM rf_register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO rf_register(id,name,location,mobile,email,uname,pass,create_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid,name,location,mobile,email,uname,pass1,rdate)
            mycursor.execute(sql, val)
            mydb.commit()            
            #print(mycursor.rowcount, "Registered Success")
            msg="sucess"
            #if mycursor.rowcount==1:
            return redirect(url_for('register',act='1'))
        else:
            msg='User Already Exist!'
    return render_template('register.html',msg=msg,act=act)

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

@app.route('/feature_extract', methods=['GET', 'POST'])
def feature_extract():
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
    windspeed_weather_df = updated_weather_df.groupby(['Location'])[['WindSpeed9am', 'WindSpeed3pm']].mean()
    windspeed_weather_df = windspeed_weather_df.reset_index()
    windspeed_weather_df.head()

    x = windspeed_weather_df.loc[:, 'Location']
    y1 = windspeed_weather_df['WindSpeed9am'] 
    y2 = windspeed_weather_df['WindSpeed3pm']

    #plt.figure(figsize = (15, 8))

    #plt.plot(x, y1, marker='D', color = 'darkgreen', label = 'WindSpeed at 9am') 
    #plt.plot(x, y2, marker='D', color = 'darkorange', label = 'WindSpeed at 3pm')

    #plt.xlabel('Location', fontsize = 14)
    #plt.ylabel('WindSpeed', fontsize = 14)
    #plt.title('Location-wise observation of Average WindSpeed', fontsize = 18)
    #plt.legend(fontsize = 10, loc = 'best')
    #plt.xticks(rotation=80)
    #plt.show()

    ##
    humidity_weather_df = updated_weather_df.groupby(['Location'])[['Humidity9am', 'Humidity3pm']].mean()
    humidity_weather_df = humidity_weather_df.reset_index()
    humidity_weather_df.head()

    x = humidity_weather_df.loc[:, 'Location']
    y1 = humidity_weather_df['Humidity9am'] 
    y2 = humidity_weather_df['Humidity3pm']

    '''plt.figure(figsize = (15, 8))

    plt.bar(x, y1, color = 'gold', label = 'Humidity at 9am') 
    plt.bar(x, y2, color = 'coral',label = 'Humidity at 3pm')

    plt.xlabel('Location', fontsize = 14)
    plt.ylabel('Humidity', fontsize = 14)
    plt.title('Location-wise observation of Average Humidity', fontsize = 18)
    plt.legend(fontsize = 10, loc = 'best')
    plt.xticks(rotation=80)
    plt.show()'''

    ##
    pressure_weather_df = updated_weather_df.groupby(['Location'])[['Pressure9am', 'Pressure3pm']].mean()
    pressure_weather_df = pressure_weather_df.reset_index()
    pressure_weather_df.head()

    x = pressure_weather_df.loc[:, 'Location']
    y1 = pressure_weather_df['Pressure9am'] 
    y2 = pressure_weather_df['Pressure3pm']

    '''plt.figure(figsize = (15, 8))

    plt.plot(x, y1, marker='o', color = 'purple', label = 'Pressure at 9am') 
    plt.plot(x, y2, marker='o', color = 'darkcyan', label = 'Pressure at 3pm')

    plt.xlabel('Location', fontsize = 14)
    plt.ylabel('Pressure', fontsize = 14)
    plt.title('Location-wise observation of Average Pressure', fontsize = 18)
    plt.legend(fontsize = 10, loc = 'best')
    plt.xticks(rotation=80)
    plt.show()'''


    location_weather_df = updated_weather_df.groupby(['Location'])[['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']].mean()
    location_weather_df = location_weather_df.reset_index()
    location_weather_df.head()

    x = location_weather_df.loc[:, 'Location']
    y1 = location_weather_df['MinTemp'] 
    y2 = location_weather_df['MaxTemp']
    y3 = location_weather_df['Temp9am'] 
    y4 = location_weather_df['Temp3pm']

    '''plt.figure(figsize = (15, 8))

    plt.plot(x, y1, label = 'Minimum Temperature', marker='o', alpha = 0.8) 
    plt.plot(x, y2, label = 'Maximum Temperature', marker='o', alpha = 0.8) 
    plt.plot(x, y3, label = 'Temperature at 9am', marker='o', alpha = 0.8) 
    plt.plot(x, y4, label = 'Temperature at 3pm', marker='o', alpha = 0.8)

    plt.xlabel('Location', fontsize = 14)
    plt.ylabel('Temperature', fontsize = 14)
    plt.title('Location-wise observation of Average Temperature', fontsize = 18)
    plt.legend(fontsize = 10, loc = 'best')
    plt.xticks(rotation=80)
    plt.show()'''

    num_weather_df = updated_weather_df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                                         'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                                         'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                         'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow']]
    dat1=num_weather_df.head()
    data1=[]
    for ss1 in dat1.values:
        data1.append(ss1)

    ##
    column_names = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    one_original_hot = pd.get_dummies(updated_weather_df[column_names])
    one_original_hot.head()

    final_df = pd.concat([num_weather_df, one_original_hot], axis=1)
    final_df.head()

    #Unknown Weather Features
    num_unknown_weather_df = updated_unknown_weather_df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
                                                     'Sunshine', 'WindGustSpeed', 'WindSpeed9am',
                                                     'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                                                     'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                                     'Temp9am', 'Temp3pm', 'RainToday']]
    dat2=num_unknown_weather_df.head()
    data2=[]
    for ss2 in dat2.values:
        data2.append(ss2)
    ##
    column_names = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    one_unknown_hot = pd.get_dummies(updated_unknown_weather_df[column_names])
    one_unknown_hot.head()

    final_unknown_df = pd.concat([num_unknown_weather_df, one_unknown_hot], axis=1)
    final_unknown_df.head()
    ##
    weather_corr = final_df.corr()
    dat3=weather_corr
    #print(len(dat3))#115
    col=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
       'Humidity3pm','WindDir3pm_NNW', 'WindDir3pm_NW', 'WindDir3pm_S', 'WindDir3pm_SE',
       'WindDir3pm_SSE', 'WindDir3pm_SSW', 'WindDir3pm_SW', 'WindDir3pm_W',
       'WindDir3pm_WNW', 'WindDir3pm_WSW']
       
    data3=[]
    i=0
    j=0
    for ss3 in dat3.values:
        dt=[]
        if i<9:
            
            dt.append(col[j])
            dt.append(ss3)
            data3.append(dt)
            j+=1
        elif i>104:
            dt.append(col[j])
            dt.append(ss3)
            data3.append(dt)
            j+=1
        
        
        i+=1
    print(data3)
    ##
    cor_target = abs(weather_corr["RainTomorrow"])

    relevant_train_features = cor_target[cor_target>0.20]
    relevant_train_features
    ##
    final_weather_train_data = final_df[['Rainfall', 'Sunshine', 'WindGustSpeed',
                                     'Humidity9am', 'Humidity3pm', 'Pressure9am',
                                     'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                     'RainToday', 'RainTomorrow']]
    final_weather_train_data.head()
    #
    final_train_corr = final_weather_train_data.corr()
    mask = np.triu(np.ones_like(final_train_corr, dtype=bool))
    '''plt.figure(figsize=(16,10))
    sns.heatmap(final_train_corr,annot=True, mask = mask)
    plt.xticks(rotation=45)
    plt.show()'''
    #
    final_weather_test_data = final_unknown_df[['Rainfall', 'Sunshine', 'WindGustSpeed',
                                            'Humidity9am', 'Humidity3pm', 'Pressure9am',
                                            'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                            'RainToday']]
    final_weather_test_data.head()
    #
    final_test_corr = final_weather_test_data.corr()
    mask = np.triu(np.ones_like(final_test_corr, dtype=bool))
    '''plt.figure(figsize=(16,10))
    sns.heatmap(final_test_corr,annot=True, mask = mask)
    plt.xticks(rotation=45)
    plt.show()'''
    #
    X = final_weather_train_data.drop(['RainTomorrow'], axis=1)
    y = final_weather_train_data['RainTomorrow']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)

    X_test = final_weather_test_data
    print(X_test.shape)

    
    return render_template('feature_extract.html',data1=data1,data2=data2,data3=data3)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
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
    windspeed_weather_df = updated_weather_df.groupby(['Location'])[['WindSpeed9am', 'WindSpeed3pm']].mean()
    windspeed_weather_df = windspeed_weather_df.reset_index()
    windspeed_weather_df.head()

    x = windspeed_weather_df.loc[:, 'Location']
    y1 = windspeed_weather_df['WindSpeed9am'] 
    y2 = windspeed_weather_df['WindSpeed3pm']

    #plt.figure(figsize = (15, 8))

    #plt.plot(x, y1, marker='D', color = 'darkgreen', label = 'WindSpeed at 9am') 
    #plt.plot(x, y2, marker='D', color = 'darkorange', label = 'WindSpeed at 3pm')

    #plt.xlabel('Location', fontsize = 14)
    #plt.ylabel('WindSpeed', fontsize = 14)
    #plt.title('Location-wise observation of Average WindSpeed', fontsize = 18)
    #plt.legend(fontsize = 10, loc = 'best')
    #plt.xticks(rotation=80)
    #plt.show()

    ##
    humidity_weather_df = updated_weather_df.groupby(['Location'])[['Humidity9am', 'Humidity3pm']].mean()
    humidity_weather_df = humidity_weather_df.reset_index()
    humidity_weather_df.head()

    x = humidity_weather_df.loc[:, 'Location']
    y1 = humidity_weather_df['Humidity9am'] 
    y2 = humidity_weather_df['Humidity3pm']

    '''plt.figure(figsize = (15, 8))

    plt.bar(x, y1, color = 'gold', label = 'Humidity at 9am') 
    plt.bar(x, y2, color = 'coral',label = 'Humidity at 3pm')

    plt.xlabel('Location', fontsize = 14)
    plt.ylabel('Humidity', fontsize = 14)
    plt.title('Location-wise observation of Average Humidity', fontsize = 18)
    plt.legend(fontsize = 10, loc = 'best')
    plt.xticks(rotation=80)
    plt.show()'''

    ##
    pressure_weather_df = updated_weather_df.groupby(['Location'])[['Pressure9am', 'Pressure3pm']].mean()
    pressure_weather_df = pressure_weather_df.reset_index()
    pressure_weather_df.head()

    x = pressure_weather_df.loc[:, 'Location']
    y1 = pressure_weather_df['Pressure9am'] 
    y2 = pressure_weather_df['Pressure3pm']

    '''plt.figure(figsize = (15, 8))

    plt.plot(x, y1, marker='o', color = 'purple', label = 'Pressure at 9am') 
    plt.plot(x, y2, marker='o', color = 'darkcyan', label = 'Pressure at 3pm')

    plt.xlabel('Location', fontsize = 14)
    plt.ylabel('Pressure', fontsize = 14)
    plt.title('Location-wise observation of Average Pressure', fontsize = 18)
    plt.legend(fontsize = 10, loc = 'best')
    plt.xticks(rotation=80)
    plt.show()'''


    location_weather_df = updated_weather_df.groupby(['Location'])[['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']].mean()
    location_weather_df = location_weather_df.reset_index()
    location_weather_df.head()

    x = location_weather_df.loc[:, 'Location']
    y1 = location_weather_df['MinTemp'] 
    y2 = location_weather_df['MaxTemp']
    y3 = location_weather_df['Temp9am'] 
    y4 = location_weather_df['Temp3pm']

    '''plt.figure(figsize = (15, 8))

    plt.plot(x, y1, label = 'Minimum Temperature', marker='o', alpha = 0.8) 
    plt.plot(x, y2, label = 'Maximum Temperature', marker='o', alpha = 0.8) 
    plt.plot(x, y3, label = 'Temperature at 9am', marker='o', alpha = 0.8) 
    plt.plot(x, y4, label = 'Temperature at 3pm', marker='o', alpha = 0.8)

    plt.xlabel('Location', fontsize = 14)
    plt.ylabel('Temperature', fontsize = 14)
    plt.title('Location-wise observation of Average Temperature', fontsize = 18)
    plt.legend(fontsize = 10, loc = 'best')
    plt.xticks(rotation=80)
    plt.show()'''

    num_weather_df = updated_weather_df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                                         'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                                         'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                         'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow']]
    num_weather_df.head()
    
    ##
    column_names = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    one_original_hot = pd.get_dummies(updated_weather_df[column_names])
    one_original_hot.head()

    final_df = pd.concat([num_weather_df, one_original_hot], axis=1)
    final_df.head()

    #Unknown Weather Features
    num_unknown_weather_df = updated_unknown_weather_df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
                                                     'Sunshine', 'WindGustSpeed', 'WindSpeed9am',
                                                     'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                                                     'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                                     'Temp9am', 'Temp3pm', 'RainToday']]
    num_unknown_weather_df.head()
    
    ##
    column_names = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    one_unknown_hot = pd.get_dummies(updated_unknown_weather_df[column_names])
    one_unknown_hot.head()

    final_unknown_df = pd.concat([num_unknown_weather_df, one_unknown_hot], axis=1)
    final_unknown_df.head()
    ##
    weather_corr = final_df.corr()
    weather_corr
    
    ##
    cor_target = abs(weather_corr["RainTomorrow"])

    relevant_train_features = cor_target[cor_target>0.20]
    relevant_train_features
    ##
    final_weather_train_data = final_df[['Rainfall', 'Sunshine', 'WindGustSpeed',
                                     'Humidity9am', 'Humidity3pm', 'Pressure9am',
                                     'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                     'RainToday', 'RainTomorrow']]
    final_weather_train_data.head()
    #
    final_train_corr = final_weather_train_data.corr()
    mask = np.triu(np.ones_like(final_train_corr, dtype=bool))
    '''plt.figure(figsize=(16,10))
    sns.heatmap(final_train_corr,annot=True, mask = mask)
    plt.xticks(rotation=45)
    plt.show()'''
    #
    final_weather_test_data = final_unknown_df[['Rainfall', 'Sunshine', 'WindGustSpeed',
                                            'Humidity9am', 'Humidity3pm', 'Pressure9am',
                                            'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                                            'RainToday']]
    final_weather_test_data.head()
    #
    final_test_corr = final_weather_test_data.corr()
    mask = np.triu(np.ones_like(final_test_corr, dtype=bool))
    '''plt.figure(figsize=(16,10))
    sns.heatmap(final_test_corr,annot=True, mask = mask)
    plt.xticks(rotation=45)
    plt.show()'''
    #
    X = final_weather_train_data.drop(['RainTomorrow'], axis=1)
    y = final_weather_train_data['RainTomorrow']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)

    X_test = final_weather_test_data
    print(X_test.shape)

    ##
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)
    log_reg_model.predict(X_val)

    log_reg_model_score = log_reg_model.score(X_val, y_val)
    log_reg_model_accuracy = round(log_reg_model_score*100, 2)
    #print("The classification accuracy of Logistic Regression model is "+ str(log_reg_model_accuracy)+"%")

    y_pred = log_reg_model.predict(X_val)
    cm = confusion_matrix(y_val,y_pred)
    '''axes = sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', linewidths=.5)
    class_labels = ['Not Rain', 'Rain']

    axes.set_xlabel('Predicted', fontsize=12)
    axes.set_ylabel('Actual', fontsize=12)

    xtick_marks = np.arange(len(class_labels)) + 0.5
    ytick_marks = np.arange(len(class_labels)) + 0.5

    axes.set_xticks(xtick_marks)
    axes.set_xticklabels(class_labels, rotation=45)

    axes.set_yticks(ytick_marks)
    axes.set_yticklabels(class_labels, rotation=0)

    axes.set_title('Logistic Regression Confusion Matrix', fontsize=14, pad=20)'''

    #print('=========================================')
    #print()
    #print(classification_report(y_val, y_pred, target_names=class_labels))

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_model.predict(X_val)
    knn_model_score = knn_model.score(X_val, y_val)
    knn_model_accuracy = round(knn_model_score*100, 2)
    print("The classification accuracy of KNN model is "+ str(knn_model_accuracy)+"%")
    y_pred = knn_model.predict(X_val)
    cm = confusion_matrix(y_val,y_pred)
    #axes = sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', linewidths=.5)
    class_labels = ['Not Rain', 'Rain']

    '''axes.set_xlabel('Predicted', fontsize=12)
    axes.set_ylabel('Actual', fontsize=12)

    xtick_marks = np.arange(len(class_labels)) + 0.5
    ytick_marks = np.arange(len(class_labels)) + 0.5

    axes.set_xticks(xtick_marks)
    axes.set_xticklabels(class_labels, rotation=45)

    axes.set_yticks(ytick_marks)
    axes.set_yticklabels(class_labels, rotation=0)

    axes.set_title('KNN Confusion Matrix', fontsize=14, pad=20)
    plt.show()'''

    print('KNN Classification Report')
    print('=========================')
    print()
    print(classification_report(y_val, y_pred, target_names=class_labels))
    value1=classification_report(y_val, y_pred, target_names=class_labels)
    ar1=[['Not Rain','0.86','0.93','0.89','15488'],['Rain','0.66','0.49','0.56','4416'],['micro avg','0.83','0.83','0.83','19904'],['macro avg','0.76','0.71','0.73','19904'],['weighted avg','0.82','0.83','0.82','19904']]
    #####

    dtree_model = DecisionTreeClassifier()
    dtree_model.fit(X_train, y_train)

    dtree_model.predict(X_val)
    dtree_model_score = dtree_model.score(X_val, y_val)
    dtree_model_accuracy = round(dtree_model_score*100, 2)
    print("The classification accuracy of Decision Tree model is "+ str(dtree_model_accuracy)+"%")
    y_pred = dtree_model.predict(X_val)
    cm = confusion_matrix(y_val,y_pred)
    #axes = sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', linewidths=.5)
    class_labels = ['Not Rain', 'Rain']

    '''axes.set_xlabel('Predicted', fontsize=12)
    axes.set_ylabel('Actual', fontsize=12)

    xtick_marks = np.arange(len(class_labels)) + 0.5
    ytick_marks = np.arange(len(class_labels)) + 0.5

    axes.set_xticks(xtick_marks)
    axes.set_xticklabels(class_labels, rotation=45)

    axes.set_yticks(ytick_marks)
    axes.set_yticklabels(class_labels, rotation=0)

    axes.set_title('Decision Tree Confusion Matrix', fontsize=14, pad=20)
    plt.show()'''
    print('Decision Tree Classification Report')
    print('===================================')
    print()
    print(classification_report(y_val, y_pred, target_names=class_labels))
    value2=classification_report(y_val, y_pred, target_names=class_labels)
    ar2=[['Not Rain','0.86','0.85','0.86','15488'],['Rain','0.50','0.52','0.51','4416'],['micro avg','0.78','0.78','0.78','19904'],['macro avg','0.68','0.69','0.68','19904'],['weighted avg','0.78','0.78','0.78','19904']]
    #####
    adaboost_model = AdaBoostClassifier()
    adaboost_model.fit(X_train, y_train)
    adaboost_model.predict(X_val)
    adaboost_model_score = adaboost_model.score(X_val, y_val)
    adaboost_model_accuracy = round(adaboost_model_score*100, 2)
    print("The classification accuracy of XGBoost model is "+ str(adaboost_model_accuracy)+"%")
    y_pred = adaboost_model.predict(X_val)
    cm = confusion_matrix(y_val,y_pred)
    #axes = sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', linewidths=.5)
    class_labels = ['Not Rain', 'Rain']

    '''axes.set_xlabel('Predicted', fontsize=12)
    axes.set_ylabel('Actual', fontsize=12)

    xtick_marks = np.arange(len(class_labels)) + 0.5
    ytick_marks = np.arange(len(class_labels)) + 0.5

    axes.set_xticks(xtick_marks)
    axes.set_xticklabels(class_labels, rotation=45)

    axes.set_yticks(ytick_marks)
    axes.set_yticklabels(class_labels, rotation=0)

    axes.set_title('XGBoost Confusion Matrix', fontsize=14, pad=20)
    plt.show()'''

    print('XGBoost Classification Report')
    print('==============================')
    print()
    print(classification_report(y_val, y_pred, target_names=class_labels))
    value3=classification_report(y_val, y_pred, target_names=class_labels)
    ar3=[['Not Rain','0.87','0.94','0.90','15488'],['Rain','0.72','0.49','0.58','4416'],['micro avg','0.84','0.84','0.84','19904'],['macro avg','0.79','0.72','0.74','19904'],['weighted avg','0.83','0.84','0.83','19904']]
    #######
    
    rforest_model = RandomForestClassifier()
    rforest_model.fit(X_train, y_train)
    rforest_model.predict(X_val)
    rforest_model_score = rforest_model.score(X_val, y_val)
    rforest_model_accuracy = round(rforest_model_score*100, 2)
    print("The classification accuracy of Random Forest model is "+ str(rforest_model_accuracy)+"%")
    y_pred = rforest_model.predict(X_val)
    cm = confusion_matrix(y_val,y_pred)
    #axes = sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', linewidths=.5)
    class_labels = ['Not Rain', 'Rain']

    '''axes.set_xlabel('Predicted', fontsize=12)
    axes.set_ylabel('Actual', fontsize=12)

    xtick_marks = np.arange(len(class_labels)) + 0.5
    ytick_marks = np.arange(len(class_labels)) + 0.5

    axes.set_xticks(xtick_marks)
    axes.set_xticklabels(class_labels, rotation=45)

    axes.set_yticks(ytick_marks)
    axes.set_yticklabels(class_labels, rotation=0)

    axes.set_title('Random Forest Confusion Matrix', fontsize=14, pad=20)
    plt.show()'''
    print('Random Forest Classification Report')
    print('===================================')
    print()
    print(classification_report(y_val, y_pred, target_names=class_labels))
    value4=classification_report(y_val, y_pred, target_names=class_labels)
    ar4=[['Not Rain','0.86','0.94','0.90','15488'],['Rain','0.70','0.47','0.56','4416'],['micro avg','0.84','0.84','0.84','19904'],['macro avg','0.78','0.71','0.73','19904'],['weighted avg','0.83','0.84','0.83','19904']]
    #####
    

    Seq_model = keras.Sequential()
    Seq_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    Seq_model.add(Dense(128, activation='relu'))
    Seq_model.add(Dense(64, activation='relu'))
    Seq_model.add(Dense(32, activation='relu'))
    Seq_model.add(Dense(8, activation='relu'))
    Seq_model.add(Dense(1, activation='sigmoid'))

    Seq_model.compile(loss= "binary_crossentropy" , optimizer="adam", metrics=["accuracy"])
    Seq_model.summary()

    #hist = Seq_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=100)

    Seq_model_score = Seq_model.evaluate(X_val, y_val)
    Seq_model_accuracy = round(Seq_model_score[1]*100, 2)
    print("The classification accuracy of Sequential model is "+ str(Seq_model_accuracy)+"%")

    accuracy_dict = {'Logistic Regrssion' : log_reg_model_accuracy,
                     'K-Nearest Neighbors' : knn_model_accuracy,
                     'Decision Tree' : dtree_model_accuracy,
                     'XGBoost' : adaboost_model_accuracy,
                     'Random Forest' : rforest_model_accuracy,
                     'Deep Sequential' : Seq_model_accuracy}
    print("Classification Accuracy of All Models")
    print('=====================================')
    print()
    acy=[]
    for k, v in accuracy_dict.items():
        print(k,"=",v,"%")
        vv=k+" = "+str(v)+"%"
        acy.append(vv)

    n_estimators = [100, 150, 200]
    max_depth = [10, 15, 20]
    criterion = ['gini', 'entropy']
    bootstrap = [True,False]
    random_state = [10, 20]
    max_features = ['auto', 'sqrt']
    min_samples_split = [1, 2, 3]
    min_samples_leaf = [1, 2, 3]

    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,
                  criterion = criterion, bootstrap = bootstrap,
                  random_state = random_state, max_features = max_features, 
                  min_samples_split = min_samples_split,
                  min_samples_leaf = min_samples_leaf)

    randF = GridSearchCV(rforest_model, hyperF, cv = 3, verbose = 1, n_jobs = -1)
    '''bestF = randF.fit(X_train, y_train)

    print("Results from Random Search ::" )
    print("\nThe best estimator across ALL searched params:\n", randF.best_estimator_)
    print("\nThe best parameters across ALL searched params:\n", randF.best_params_)

    rforest_model = RandomForestClassifier(**randF.best_params_)
    rforest_model.fit(X_train, y_train)

    rforest_model.predict(X_val)
    rforest_model_score = rforest_model.score(X_val, y_val)
    rforest_model_accuracy = round(rforest_model_score*100, 2)
    print("The classification accuracy of Random Forest model is "+ str(rforest_model_accuracy)+"%")

    y_pred = rforest_model.predict(X_val)
    cm = confusion_matrix(y_val,y_pred)
    #axes = sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', linewidths=.5)
    class_labels = ['Not Rain', 'Rain']'''

    '''axes.set_xlabel('Actual', fontsize=12)
    axes.set_ylabel('Predicted', fontsize=12)

    xtick_marks = np.arange(len(class_labels)) + 0.5
    ytick_marks = np.arange(len(class_labels)) + 0.5

    axes.set_xticks(xtick_marks)
    axes.set_xticklabels(class_labels, rotation=45)

    axes.set_yticks(ytick_marks)
    axes.set_yticklabels(class_labels, rotation=0)

    axes.set_title('Random Forest Confusion Matrix', fontsize=14, pad=20)'''

    '''print('Random Forest Classification Report')
    print('===================================')
    print()
    print(classification_report(y_val, y_pred, target_names=class_labels))

    predicted_value = rforest_model.predict(X_test)
    final_prediction_df = pd.DataFrame()
    final_prediction_df = updated_unknown_weather_df
    final_prediction_df["Predict-RainTomorrow"] = predicted_value
    final_prediction_df.to_csv("Final Rain Prediction.csv", index=False)
    dat1=final_prediction_df.head()
    print(dat1)'''

    
    return render_template('classify.html',value1=value1,value2=value2,value3=value3,value4=value4,acy=acy,ar1=ar1,ar2=ar2,ar3=ar3,ar4=ar4)


@app.route('/test_data', methods=['GET', 'POST'])
def test_data():
    act=""
    res=""
    uname=""
    if 'username' in session:
        uname = session['username']

    if uname is None:
        return redirect(url_for('login'))

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM rf_register where uname=%s",(uname,))
    data = mycursor.fetchone()
    name=data[1]
    
    min_temp=""
    max_temp=""
    rain=""
    gust_dir=""
    gust_speed=""
    hum1=""
    hum2=""
    pres1=""
    pres2=""
    temp1=""
    temp2=""
    
    if request.method=='POST':
        min_temp=request.form['min_temp']
        max_temp=request.form['max_temp']
        rain=request.form['rain']
        gust_dir=request.form['gust_dir']
        gust_speed=request.form['gust_speed']
        hum1=request.form['hum1']
        hum2=request.form['hum2']
        pres1=request.form['pres1']
        pres2=request.form['pres2']
        temp1=request.form['temp1']
        temp2=request.form['temp2']

        df = pd.read_csv("static/dataset/weather_test.csv")

        x=0

        mt=float(min_temp)
        mt1=mt-1
        mt2=mt+1

        mx=float(max_temp)
        mx1=mx-1
        mx2=mx+1

        rain1=float(rain)
        r1=rain1-2
        r2=rain1+2
        gs=int(gust_speed)
        gs1=gs-1
        gs2=gs+2

        h1=float(hum1)
        hm1=h1-1
        hm2=h1+1

        h2=float(hum2)
        hm3=h2-1
        hm4=h2+1

        p1=float(pres1)
        pr1=p1-2
        pr2=p1+2

        p2=float(pres2)
        pr3=p2-2
        pr4=p2+2

        t1=float(temp1)
        tm1=t1-2
        tm2=t1+2

        t2=float(temp2)
        tm3=t2-2
        tm4=t2+2

       
        
    
        for rr in df.values:
            if rr[2]>mt1 and rr[2]<=mt2 and rr[3]>mx1 and rr[3]<=mx2 and rr[4]>r1 and rr[4]<=r2 and rr[7]==gust_dir and rr[8]>gs1 and rr[8]<=gs2:
                print("a")
                if rr[13]>hm1 and rr[13]<=hm2 and rr[14]>hm3 and rr[14]<=hm4 and rr[15]>pr1 and rr[15]<=pr2 and rr[16]>pr3 and rr[16]<=pr4:
                    print("b")
                    if rr[19]>tm1 and rr[19]<=tm2 and rr[20]>tm3 and rr[20]<=tm4:
                        print("c")
                        x+=1

        print(x)
        act="1"
        if x>0:
            res="Yes"
        else:
            res="No"
            
    return render_template('test_data.html',act=act,name=name,res=res,min_temp=min_temp,max_temp=max_temp,rain=rain,gust_dir=gust_dir,gust_speed=gust_speed,hum1=hum1,hum2=hum2,pres1=pres1,pres2=pres2,temp1=temp1,temp2=temp2)




@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


