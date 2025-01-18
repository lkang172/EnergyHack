import "bootstrap";
import { useState } from "react";
import CodeInput from "./CodeInput";

export default function Input() {
  const [iterations, setIterations] = useState(0);
  const [gpu, setGpu] = useState("");
  const [code, setCode] = useState();
  const [response, setResponse] = useState<any>();

  const calculate = async () => {
    const exportData = { code: code, gpu: gpu, iterations: iterations };
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

  //add dataset size and batch size inputs
  return (
    <>
      <div className="container">
        <CodeInput onChange={(e: any) => setCode(e.target.value)}></CodeInput>
        <input
          type="text"
          className="form-control"
          placeholder="Number of iterations"
          onChange={(e) => setIterations(parseInt(e.target.value))}
        ></input>
        <select
          name="gpus"
          className="btn btn-ptimary"
          onChange={(e) => setGpu(e.target.value)}
        >
          <option label="GPU"></option>
          <option label="GeForce GTX 580" value="6.48"></option>
          <option label="GeForce GTX 590" value="6.82"></option>
          <option label="GeForce GTX 680" value="15.85"></option>
          <option label="GeForce GTX 690" value="18.73"></option>
          <option label="Tesla K10" value="20.36"></option>
          <option label="Tesla K20x" value="16.77"></option>
          <option label="GeForce GTX 780" value="16.64"></option>
          <option label="Tesla K40" value="21.45"></option>
          <option label="GeForce GTX 780 TI" value="21.40"></option>
          <option label="GeForce GTX Titan Black" value="22.60"></option>
          <option label="GeForce GTX Titan Z" value="21.65"></option>
          <option label="GeForce GTX 980" value="30.18"></option>
          <option label="Tesla K80" value="27.40"></option>
          <option label="GeForce GTX TITAN X" value="26.76"></option>
          <option label="GeForce GTX 980 Ti" value="24.24"></option>
          <option label="Tesla M60" value="32.17"></option>
          <option label="Tesla M40" value="27.36"></option>
          <option label="GeForce GTX 1080" value="49.28"></option>
          <option label="TITAN X Pascal" value="43.88"></option>
          <option label="GeForce GTX 1080 Ti" value="45.36"></option>
          <option label="TITAN XP" value="48.60"></option>
          <option label="Tesla V100" value="52.33"></option>
          <option label="Tesla T4" value="115.71"></option>
          <option label="GeForce RTX 2080" value="46.84"></option>
          <option label="GeForce RTX 2080 Ti" value="53.80"></option>
          <option label="Nvidia Titan RTX" value="58.25"></option>
          <option label="GeForce RTX 3080" value="93.13"></option>
          <option label="GeForce RTX 3090" value="101.71"></option>
        </select>
        <select name="intensityarea" className="btn btn-ptimary">
          <option value="">Location/Area</option>
        </select>
        <button type="button" className="btn btn-primary" onClick={calculate}>
          Calculate
        </button>
      </div>
      <div className="output">
        {response != null && (
          <>
            <p>Total energy: {response.energy} watts</p>
          </>
        )}
      </div>
    </>
  );
}
