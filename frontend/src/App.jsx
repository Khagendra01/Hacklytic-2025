import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Api from './Api'
import Auth from './Auth'
function App() {

  return (
    <>
      <Api />
      <Auth />
    </>
  )
}

export default App
