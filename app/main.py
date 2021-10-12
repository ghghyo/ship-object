#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:45:40 2021

@author: yabubaker
"""


from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import base64
from io import BytesIO
from PIL import Image
import app.transforms as T
from app.torch_utils import accept_input, get_prediction, get_image

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# A welcome message to test our server
@app.route('/')
@cross_origin()
def index():
    return "<h1>Welcome to our server !!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        #print(request.data)
        #file = request.form.get('number')
        file=request.form['nm']
 
        img=accept_input(int(file))
         
        boxes = get_prediction(img)
        
        return boxes
    
@app.route('/get_images', methods=['POST'])
def get_images():
    file=request.form['nm']
    img=accept_input(int(file))
    image=get_image(img)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({'status': True, 'image': str(img_str)})

@app.route('/get_uploaded', methods=['POST'])
def get_uploaded():
    file=request.form['nm']
    im = Image.open(BytesIO(base64.b64decode(file.split(",")[1])))
    image=get_image(T.ToTensor()(im)[0])
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({'status': True, 'image': str(img_str)})
