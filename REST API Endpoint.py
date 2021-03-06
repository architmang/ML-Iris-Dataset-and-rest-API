import flask
from flask import Flask,jsonify 
app = Flask(__name__)

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv(r"C:\Users\archi\Desktop\bezdekIris.csv")
cdf=df[["sepal length","sepal width","petal length","petal width"]]
edf=df["Class"]
x=np.asanyarray(cdf)
y=np.asanyarray(edf)

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.4,random_state=3)
Class_tree=DecisionTreeClassifier(criterion='entropy',max_depth=3)
Class_tree.fit(train_x,train_y)

@app.route('/')
def hello_world():
    return jsonify("Hey User, Enter the sepal length , sepal width , petal length , petal width as floating point numbers after slashes after the base URL. Example enter /5.3/3.3/1.0/1.0 after base URL")

@app.route('/<float:a>/<float:b>/<float:c>/<float:d>')
def Classifier2(a,b,c,d):
    result={
        "sepal length":a, 
        "sepal width":b, 
        "petal length":c,
        "petal width":d, 
        "Predicted Class":Class_tree.predict([[a,b,c,d]])[0] 
        }
    return result    

if __name__=="__main__":
    app.run(debug="true") 
