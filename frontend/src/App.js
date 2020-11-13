import Home from "./pages/Home";
import Helmet from "react-helmet";

function App() {
  return (
    <div className="App">
      <Helmet>
        <style>
          {
            "body, html{ background-color: #F5F5F5; max-width: 100%; overflow-x: hidden}"
          }
        </style>
      </Helmet>
      <Home />
    </div>
  );
}

export default App;
