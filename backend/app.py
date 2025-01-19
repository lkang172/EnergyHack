# from flask import Flask, request
# from flask_cors import CORS
from parse_data import parse_function

# app = Flask(__name__) 
# CORS(app)

dummy_data = '''
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, tqdm


class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, **kwargs):
        super(model, self).__init__()

        self.visualModel = None
        self.audioModel = None
        self.fusionModel = None
        self.fcModel = None

        self.loss_fn = nn.CrossEntropyLoss()

    def createVisualModel(self):
        self.visualModel = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.Flatten()
        )

    def createAudioModel(self):
        self.audioModel = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.Flatten()
        )

    def createFusionModel(self):
        pass

    def createFCModel(self):
        self.fcModel = nn.Sequential(
            nn.Linear(29824, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
'''

def calculate_energy(data):
    parse_function(data)
    # total_flops = 20 * 10**10 #replace with their function, which outputs total flops
    # energy = ((total_flops / 10**9) / float(data['gpu'])) * (float(data['datasetSize']) / float(data['batchSize']))
    # return {"energy": energy}
  
# @app.route('/calculate', methods=['POST']) 
# def get_calculation():
#     data = request.get_json()
    
#     return(calculate_energy(data))


  
# if __name__ == "__main__": 
#     app.run(debug=True)

calculate_energy(dummy_data)