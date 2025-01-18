import ParameterInputContainer from "./components/ParameterInputContainer";
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
