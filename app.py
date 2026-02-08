import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
from flask import Flask ,render_template,request,url_for,app

app = Flask(__name__)

## Load the model 
model=pickle.load(open('boston.pkl','rb'))

@app.route('/')
def home():
  return render_template("index.html")

@app.route('/predict_api',methods=["POST"])
def predict_api():
  return



