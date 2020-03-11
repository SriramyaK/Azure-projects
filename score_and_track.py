import json
import torch
import os, base64
import json
from fastai import *
from fastai.vision import *
from shutil import copyfile
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector

def init():
    global model
    
    model_file = Model.get_model_path('./cars.pkl')
    model_dir = os.path.dirname(model_file)
    #print(model_path)
    model = load_learner(model_dir)
    
    global prediction_dc
    prediction_dc = ModelDataCollector("cars-classifier", identifier="predictions", feature_names=["prediction"])

def run(raw_data):
    base64_string = json.loads(raw_data)['data']
    base64_bytes = base64.b64decode(base64_string)
    with open(os.path.join(os.getcwd(),"score.jpg"), 'wb') as f:
        f.write(base64_bytes)
    
    # make prediction
    img = open_image(os.path.join(os.getcwd(),"score.jpg"))
    result = model.predict(img)
    prediction_dc.collect(result)
    return json.dumps({'category':str(result[0]), 'confidence':result[2].data[1].item()})