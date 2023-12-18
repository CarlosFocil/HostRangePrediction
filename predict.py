import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_mi=0.12_depth=None_min_samples_leaf=5.bin'

with open(model_file,'rb') as f_in:
    dv,model = pickle.load(f_in)

app = Flask('strain-classification')

@app.route('/predict_host_range', methods=['POST'])
def predict_host_range():
    strain_profile=request.get_json()

    X = dv.transform([strain_profile])
    y_pred = model.predict_proba(X)[0, 1]
    
    if y_pred >= 0.5:
        prediction = 'Generalist'
    else:
        prediction = 'Specialist'
    
    result = {
        'Generalist probability':float(y_pred),
        'Host range': str(prediction)
    }

    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)