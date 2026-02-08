import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
from flask import Flask ,render_template,request,url_for,app,jsonify

app = Flask(__name__)

## Load the model 
model=pickle.load(open('boston.pkl','rb'))

@app.route('/')
def home():
  return render_template("index.html")

@app.route('/predict_api',methods=["POST"])
def predict_api():
  data=request.json['data']

  # list of training columns
  input_df =pd.DataFrame([data], columns=['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT'])

  prediction = model.predict(input_df)

  return jsonify({
         "predictio":float(prediction[0])
  })

#----Html form prediction ------

@app.route("/predict", methods=['POST'])
def predict():
  data = [

        float(request.form["CRIM"]),
        float(request.form["ZN"]),
        float(request.form["INDUS"]),
        float(request.form["NOX"]),
        float(request.form["RM"]),
        float(request.form["AGE"]),
        float(request.form["DIS"]),
        float(request.form["RAD"]),
        float(request.form["TAX"]),
        float(request.form["PTRATIO"]),
        float(request.form["LSTAT"])
  ]

  df = pd.DataFrame([data], columns=[
        'CRIM','ZN','INDUS','NOX','RM','AGE',
        'DIS','RAD','TAX','PTRATIO','LSTAT'
  ])

  prediction = model.predict(df)
  return render_template(
    "index.html",prediction_text=f"prediction price: $ {prediction[0]:.2f}"
  )

if __name__ == "__main__":
  app.run(debug=True)
  



