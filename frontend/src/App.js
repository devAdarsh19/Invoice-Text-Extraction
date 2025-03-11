import React, {useState} from 'react'
import InvoiceUploader from './components/InvoiceUploader';
import Results from './components/Results';
import './App.css'

function App () {
    const [invoiceData, setInvoiceData] = useState(null);
    return (
        <div>
            <h1>Invoice Field Extractor</h1>
            <InvoiceUploader setInvoiceData={setInvoiceData} />
            <p>{ invoiceData && (Object.entries(invoiceData).map(([key, value]) => {
                <p><strong>{ key }</strong> { value }</p>
            }) )}</p>
            {invoiceData && <Results invoiceData={invoiceData} />}
        </div>
    );
}

export default App;
