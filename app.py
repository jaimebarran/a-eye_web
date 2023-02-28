from flask import Flask, request, jsonify
from utilities import main

app = Flask(__name__)

# Through post
@app.post('/predict')
def predict():
    req = request.json
    try:
        text = req['text']
    except KeyError:
        return jsonify({'error': 'No text sent'})
        
    if text == 'predict':
        return main()
    else:
        return 'Error: incorrect request!! \n'

# Through get (be in that page; index  in this case)
# @app.get('/predict')
# def predict():
#     return main()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# curl -X POST -H "Content-Type: application/json" -d '{"text":"predict"}' 0.0.0.0:5000/predict
# netstat -nlp | grep 5000
# kill -9 441420