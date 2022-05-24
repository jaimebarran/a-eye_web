from flask import Flask, request
from utilities import main

app = Flask(__name__)

# Through post
@app.post('/predict')
def predict():
    req = request.json
    text = req['text']
    if text == 'predict':
        return main()
    else:
        return 'Error: incorrect request!! \n'

# Through get (be in that page; index  in this case)
# @app.get("/")
# def predict():
#     return main()

if __name__ == '__main__':
    app.run(debug=True)