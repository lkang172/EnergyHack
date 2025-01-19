from flask import Flask, request
from flask_cors import CORS
from calculator import Calculator
from parse_data import parse_function

app = Flask(__name__) 
CORS(app)

def calculate_energy(data):
    parsed_data = parse_function(data['code'])
    calculator = Calculator(parsed_data, data['batchSize'])
    total_flops = calculator.calculate()
    print("Total flops: " + str(total_flops))
    energy = ((total_flops / 10**9) / float(data['gpu'])) * (float(data['datasetSize']) / float(data['batchSize']))
    return {"energy": energy}
  
@app.route('/calculate', methods=['POST']) 
def get_calculation():
    data = request.get_json()
    
    return(calculate_energy(data))


  
if __name__ == "__main__": 
    app.run(debug=True)

