import React from "react";

const Results = ({ invoiceData }) => {
  return (
    <div className="results-container">
      <div className="invoice-results">
        <h2>Invoice Details</h2>
        {invoiceData && (
          <ul>
            {Object.entries(invoiceData).map(([key, value]) => (
              <div className="list-items">
                <li>
                  <strong>{key} : </strong>
                  {value}
                </li>
              </div>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default Results;
