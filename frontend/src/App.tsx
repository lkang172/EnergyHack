import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import ParameterInputContainer from './components/ParameterInputContainer'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <ParameterInputContainer></ParameterInputContainer>
    </>
  )
}

export default App
