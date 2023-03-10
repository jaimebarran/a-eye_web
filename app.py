import os
from flask import Flask, request, jsonify, render_template, request, redirect, url_for
from utilities import main
import logging
from flask_sqlalchemy import SQLAlchemy

# ----------------------------------------------------------------------------------------------
# FLASK

app = Flask(__name__)

# Set Flask environment to development (to print console)
# export FLASK_APP=./a-eye_web/app.py
os.environ['FLASK_APP'] = './a-eye_web/app.py' # for flask run
os.environ['FLASK_DEBUG'] = 'True' # to print console

# /// = relative path, //// = absolute path
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Manage logs
logging.basicConfig(filename='./a-eye_web/app.log', level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(message)s')


# ----------------------------------------------------------------------------------------------
# ROUTES

@app.route('/')
def home():
    return render_template('base.html', done=False)

@app.route('/predict')
def predict():
    main()
    return render_template('base.html', done=True)


# ----------------------------------------------------------------------------------------------
# MAIN

if __name__ == '__main__':
    db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True to development mode


# ----------------------------------------------------------------------------------------------
# UTILS

# Call flask from bash (other terminal)
# curl -X POST -H "Content-Type: application/json" -d '{"text":"predict"}' 0.0.0.0:5000/predict

# Finish open port
# netstat -nlp | grep 5000
# kill -9 441420

# Launch app.py and save console output
# /home/jaimebarranco/miniconda3/envs/a-eye/bin/python /mnt/sda1/Repos/a-eye/a-eye_web/app.py >> .//a-eye_web//console.log 2>&1

# Complete command: args, logs
# /home/jaimebarranco/miniconda3/envs/a-eye/bin/python /mnt/sda1/Repos/a-eye/a-eye_web/app.py -i /home/jaimebarranco/Desktop/test_inference/input -o /home/jaimebarranco/Desktop/test_inference/output  >> .//a-eye_web//console.log 2>&1

# Run docker with nvidia
# sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
# docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
# echo $SUDO_PWD | sudo -S -s docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi


