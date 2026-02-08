#!/usr/bin/env python
# coding: utf-8



from io import BytesIO
from urllib import request
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image



# url = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'

interpreter = tflite.Interpreter('Hairstyle_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'Straight',
    'Curly'
]


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess (url):
    img = download_image(url)

    img = prepare_image(img, target_size=(200,200))



    X = np.array(img, dtype='float32')
    X  = X /255

    X = np.array([X])
    return X


def predict(url):
    
    X = preprocess(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_preds= preds[0].tolist()

    return 'Curly' if float(float_preds[0]) > .5 else 'Straight', float_preds

def lambda_handler(event, context):

    url = event['url']
    out, result = predict(url)
    return out,result

