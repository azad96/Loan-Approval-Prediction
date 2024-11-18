import pickle
from flask import Flask, jsonify, request
    
app = Flask(__name__)

model_path = './model.bin'
with open(model_path, 'rb') as f:
    dv, model = pickle.load(f)

@app.route('/predict', methods=["POST"])
def predict():
    client = request.get_json()

    X = dv.transform(client)
    y_pred = model.predict_proba(X)[:, 1]
    approval = y_pred >= 0.5

    result = {
        'get_loan_prob': float(y_pred),
        'get_loan': bool(approval)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)