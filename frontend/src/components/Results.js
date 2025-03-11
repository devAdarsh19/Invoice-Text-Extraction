import React from "react";

const Results = ({ invoiceData }) => {
  return (
    <div>
      <h2>Invoice Details</h2>
      {invoiceData ? (
        <ul>
          {Object.entries(invoiceData).map(([key, value]) => {
            <li>
              <strong>{key}</strong> {value}
            </li>;
          })}
        </ul>
      ) : (
        <p>No invoice data found</p>
      )}
    </div>
  );
};

export default Results;
