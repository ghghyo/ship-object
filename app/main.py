#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:45:40 2021

@author: yabubaker
"""


from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin

from app.torch_utils import accept_input, get_prediction

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
