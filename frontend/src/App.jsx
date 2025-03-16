import React, { useState } from "react";
import InvoiceUploader from "./components/InvoiceUploader";
import Results from "./components/Results";
import "./App.css";
import Navbar from "./components/Navbar";
import {Route, Router, Routes} from "react-router-dom"
import Registration from "./components/Registration";

function App() {
  const [invoiceData, setInvoiceData] = useState(null);
  return (
    <div>
      <Navbar />
      {/* <div className="title">
        <h1>Invoice Field Extractor</h1>
      </div>
      <InvoiceUploader setInvoiceData={setInvoiceData} />
      {invoiceData && <Results invoiceData={invoiceData} />}
      <footer className="footer">Adarsh Vinod, CanData.ai, 2025</footer> */}
      <Registration />
    </div>
  );
}

export default App;
