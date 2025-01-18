import './App.css'
import ParameterInputContainer from './components/ParameterInputContainer'
import { useState } from "react";
import "./App.css";
import CodeInput from "../components/CodeInput";

function App() {
  return (
    <>
      <div className="container">
        <CodeInput></CodeInput>
        <ParameterInputContainer></ParameterInputContainer>
      </div>
    </>
  );
}

export default App;
