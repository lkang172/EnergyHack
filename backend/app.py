from flask import Flask, request
from flask_cors import CORS
from calculator import Calculator
from parser import parse_function

app = Flask(__name__) 
CORS(app)

dummy_data = '''self.loss_fn = nn.CrossEntropyLoss()

    def createVisualModel(self):
        self.visualModel = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Flatten()
        )

    def createAudioModel(self):
        self.audioModel = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
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
        )'''


def calculate_energy(data):
    parsed_data = parse_function(data)
    calculator = Calculator(parsed_data, data['batchSize'])
    total_flops = calculator.calculate()
    energy = ((total_flops / 10**9) / float(data['gpu'])) * (float(data['datasetSize']) / float(data['batchSize']))
    return {"energy": energy}
  
# @app.route('/calculate', methods=['POST']) 
# def get_calculation():
#     data = request.get_json()
    
#     return(calculate_energy(data))


  
# if __name__ == "__main__": 
#     app.run(debug=True)

calculate_energy(dummy_data)