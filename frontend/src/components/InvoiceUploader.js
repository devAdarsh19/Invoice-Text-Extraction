import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import axios from "axios";

const filetypes = ["PNG", "JPG"]

const InvoiceUploader = ({ setInvoiceData }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (file) => {
    setFile(file);
  };

  const analyzeInvoice = async () => {
    if (!file) {
      return alert("Please select an invoice image file");
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/upload-invoice/",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      if (response.status !== 200) {
        throw new Error(`HTTP Error | Error code : ${response.status}`)
      }

      const data = await response.data;
      console.log(data);
      setInvoiceData(data);
      setLoading(false);
    } catch (error) {
      console.error(`Error: ${error}`);
    }
  };

  return (
    <div>
      <div className="invoice-uploader">
        <FileUploader
          handleChange={handleFileChange}
          name="file"
          types={filetypes}
          error={loading ? "true" : undefined}
        />
        <button onClick={analyzeInvoice} disabled={loading}>
          {loading ? "Analyzing..." : "Upload"}
        </button>
      </div>
    </div>
  );
};

export default InvoiceUploader;
