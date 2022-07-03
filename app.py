import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pickle

# Create flask app
flask_app = Flask(__name__)
model = joblib.load('spam_model.pkl')
model2 = joblib.load('spam_model2.pkl')

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    input_mail = request.form.values()
    message = input_mail.__next__()
    output = model.predict([message])
    
    if(output == 0):
        flag=0
        return render_template("index.html",flag=flag)
        
    else:
        flag=1
        return render_template("failed.html")  
          
    
@flask_app.route("/predict2", methods = ["POST"])
def predict2():
    input_mail = request.form.values()
    message = input_mail.__next__()
    output = model2.predict([message])
    
    if(output == 0):
        return render_template("success.html")
    else:
        return render_template("failed.html")

if __name__ == "__main__":
    flask_app.run(debug=True)