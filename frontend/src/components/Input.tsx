import { useState } from "react";
import CodeInput from "./CodeInput";
import gpuData from "../../data/gpu-efficiencies.json";
import carbonData from "../../data/carbon-intensities.json";

export default function Input() {
  const [datasetSize, setDatasetSize] = useState(0);
  const [batchSize, setBatchSize] = useState(0);
  const [gpu, setGpu] = useState("");
  const [code, setCode] = useState();
  const [carbon, setCarbon] = useState(0);
  const [response, setResponse] = useState<any>();
  const carbon_data: Record<string, number> = carbonData;
  const gpu_data: Record<string, number> = gpuData;
  //Cursor parking lot :D
  //  ___________
  //  | | | | | |
  //  | | | | | |
  //  ___________
  const calculate = async () => {
    const exportData = {
      code: code,
      gpu: gpu,
      datasetSize: datasetSize,
      batchSize: batchSize,
    };
    console.log("Export data: ", exportData);
    try {
      const res = await fetch("http://127.0.0.1:5000/calculate", {
        method: "POST",
        headers: {
          "Content-type": "application/json",
        },
        body: JSON.stringify(exportData),
      });
      const response = await res.json();
      setResponse(response);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <>
      <div className="container">
        <div className="container-2">
          <h1>WattsTheMatter</h1>

          <CodeInput onChange={(e: any) => setCode(e.target.value)}></CodeInput>
        </div>
        <div className="container-2">
          <center>
            <br></br>
            <h3>Configuration</h3>
            <input
              type="text"
              className="form-control"
              placeholder="Dataset Size"
              onChange={(e) => setDatasetSize(parseInt(e.target.value))}
            ></input>
            <input
              type="text"
              className="form-control"
              placeholder="Batch Size"
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
            ></input>
            <select
              name="gpus"
              className="btn btn-ptimary"
              onChange={(e) => setGpu(e.target.value)}
            >
              <option>(Select GPU)</option>
              {Object.keys(gpu_data).map((gpu) => (
                <option key={gpu} value={gpu_data[gpu]}>
                  {gpu}
                </option>
              ))}
            </select>
            <select
              name="intensityarea"
              className="btn btn-ptimary"
              onChange={(e) => setCarbon(parseInt(e.target.value))}
            >
              <option>(Select Region)</option>
              {Object.keys(carbon_data).map((country) => (
                <option key={country} value={carbon_data[country]}>
                  {country}
                </option>
              ))}
            </select>
            <br></br>
            <button
              type="button"
              className="btn btn-primary"
              onClick={calculate}
            >
              Calculate
            </button>
            <div className="output">
              {response != null && (
                <>
                  <p>Total energy: {response.energy} watts</p>
                  <p>Total carbon emissions: {response.energy * carbon} gCO</p>
                  <p>
                    That's equal to {response.energy / 1214} days worth of energy for an average American home, or {response.energy / 25000000} Taylor Swift
                    Eras Tours!
                  </p>
                </>
              )}
            </div>
          </center>
        </div>
      </div>
    </>
  );
}
