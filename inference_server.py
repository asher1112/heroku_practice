import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from flask import Flask , render_template , request
import pickle
import json


filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1> hello </h1>"


@app.route('/postjson', methods=['GET'])
def postJsonHandler():
    my_dict = {}
    my_list = ['age', 'sex','cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak','slope','ca' , 'thal']
    res = json.loads(request.json)
    for i in my_list:
        values = res[i].values()
        my_dict.update({i : values})

    df = pd.DataFrame(my_dict)
    X = df.to_numpy()
    y_pred = loaded_model.predict(X)

    return str(y_pred)

@app.route('/predict' , methods = ['POST' , 'GET'])
def predict():
    data1 = request.args.get('age')
    data2 = request.args.get('sex')
    data3 = request.args.get('cp')
    data4 = request.args.get('trestbps')
    data5 = request.args.get('chol')
    data6 = request.args.get('fbs')
    data7 = request.args.get('restecg')
    data8 = request.args.get('thalach')
    data9 = request.args.get('exang')
    data10 = request.args.get('oldpeak')
    data11 = request.args.get('slope')
    data12 = request.args.get('ca')
    data13 = request.args.get('thal')

    array = np.array([[data1 ,data2, data3 ,data4 , data5 , data6 , data7 ,data8 ,data9 , data10 ,data11, data12,data13]])
    pred = loaded_model.predict(array)
    return str(pred[0])
if __name__ == '__main__':
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    app.run(debug=True)


class LinearRegression:
    def __init__(self):