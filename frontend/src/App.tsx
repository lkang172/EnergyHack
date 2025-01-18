import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import ParameterInputContainer from './components/ParameterInputContainer'
import { useState } from "react";
import "./App.css";
import CodeInput from "../components/CodeInput";

function App() {
  return (
    <>
      <ParameterInputContainer></ParameterInputContainer>
      <div className="container">
        <CodeInput></CodeInput>
      </div>
    </>
  );
}

export default App;
