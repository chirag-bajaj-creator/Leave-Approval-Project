import React from "react";
import ReactDOM from "react-dom/client";  // ✅ Use createRoot from react-dom/client
import "./index.css";
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root")); // ✅ Use createRoot
root.render(
  
    <App />

);
