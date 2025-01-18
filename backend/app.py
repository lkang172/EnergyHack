from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__) 
CORS(app)

def calculate_energy(data):
    total_flops = 20 * 10**10
    energy = ((total_flops / 10**9) / float(data['gpu'])) * float(data['iterations'])
    return {"energy": energy}
  
@app.route('/calculate', methods=['POST']) 
def get_calculation():
    data = request.get_json()

    return(calculate_energy(data))


  
if __name__ == "__main__": 
    app.run(debug=True)