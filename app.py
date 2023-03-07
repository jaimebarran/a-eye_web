import os
from flask import Flask, request, jsonify
from utilities import main
import logging

app = Flask(__name__)

# Set Flask environment to development (to print console)
os.environ['FLASK_ENV'] = 'development'

# Manage logs
logging.basicConfig(filename='./a-eye_web/app.log', level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(message)s')

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


# ----------------------------------------------------------------------------------------------
# UTILS

# Call flask from bash (other terminal)
# curl -X POST -H "Content-Type: application/json" -d '{"text":"predict"}' 0.0.0.0:5000/predict

# Finish open port
# netstat -nlp | grep 5000
# kill -9 441420

# Launch app.py and save console output
# /home/jaimebarranco/miniconda3/envs/a-eye/bin/python /mnt/sda1/Repos/a-eye/a-eye_web/app.py >> .//a-eye_web//console.log 2>&1

# Run docker with nvidia
# sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
# docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi


