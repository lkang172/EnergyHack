import ParameterInputContainer from "./components/ParameterInputContainer";
import "./App.css";
import CodeInput from "./components/CodeInput";

function App() {

  const submit = () => {
    
  };
  return (
    <>
      <div className="container">
        <CodeInput></CodeInput>
        <ParameterInputContainer calculate={}></ParameterInputContainer>
      </div>
    </>
  );


}

export default App;
