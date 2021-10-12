#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:48:30 2021

@author: yabubaker
"""

import requests 

resp = requests.post("http://localhost:5000/predict", data={'number': '1'})

print(resp.text)