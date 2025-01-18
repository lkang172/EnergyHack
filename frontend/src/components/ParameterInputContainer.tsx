import "bootstrap";
import { useState } from "react";

interface ParametersProps {
    calculate : Function
}

export default function ParameterInputContainer({calculate}: ParametersProps) {
    const [iterations, setIterations] = useState(0);

  return (
    <div>
      <input
        type="text"
        className="form-control"
        placeholder="Number of iterations"
        onChange={(e) => setIterations(parseInt(e.target.value))}
      ></input>
      <select name="gpus" className="btn btn-ptimary">
        <option value="4090">GPUS</option>
      </select>
      <select name="intensityarea" className="btn btn-ptimary">
        <option value="">Area</option>
      </select>
      <button type="button" className="btn btn-primary" onClick={calculate}>
        Calculate
      </button>
    </div>
  );
}
