from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name___)
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/')
def hello_world():
    return render_template('index.html')
@app.route('./predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=loaded_model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1],2)
    