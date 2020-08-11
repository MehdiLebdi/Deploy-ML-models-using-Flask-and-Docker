from flask import Flask, request, jsonify

app = Flask(__name__)

#Load the model
import pickle
model = pickle.load(open('model.pkl','rb'))
labels ={
  0: "malignant",
  1: "benign"
}

import json
@app.route('/api', methods=['POST'])
def predict():
  # POST request for the data
	data = request.get_json(force=True)
	predict = model.predict(data['feature'])
	return jsonify(predict[0].tolist())

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')