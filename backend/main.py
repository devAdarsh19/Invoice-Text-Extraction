import os
import cv2
import csv
import shutil
from mistralai import Mistral
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Uploaded files dir
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CSV_DIR = "extractedInfo"
os.makedirs(CSV_DIR, exist_ok=True)

# Load API key
load_dotenv("C:\\Users\\ADMIN\\Desktop\\api_key.env")
mistral_api_key_candata = os.getenv("CANDATA_MISTRAL_API_KEY")

def extract_text(invoice_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    
    result = ocr.ocr(invoice_path, cls=True)
    text = ""
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text += line[1][0]
            
    return text

def analyze_invoice_text(text):
    model = "mistral-large-latest"
    prompt = f"Analyze the given text from an invoice and extract the following information \nInvoice Number, Date, Total Amount and Organization Name\nReturn the extracted information on new lines in the form fieldname : value \nBe careful with Invoice Number since it can also be labeled as Document Number in the document. Do not add any other text in the response other than the required fields\n{text}"
    
    client = Mistral(api_key=mistral_api_key_candata)
    
    completion = client.chat.complete(
        model=model,
        messages=[
            {"role":"user", "content":prompt}
        ]
    )
    
    return completion.choices[0].message.content

@app.post("/upload-invoice/")
def upload_analyze_invoice(file: UploadFile = File(...)):
    
    
    invoice_info_dict = {}
    # Saving file to local directory for access
    image_file = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(image_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    # Now extracting text and analyzing it to return fields
    invoice_text = extract_text(image_file)
    info_text = analyze_invoice_text(invoice_text)
    
    for line in info_text.split("\n"):
        fieldnames = ["Organization Name", "Invoice Number", "Date", "Total Amount"]
        key, *value = line.split(":")
        if key.strip() in fieldnames:
            invoice_info_dict[key.strip()] = ', '.join(value)
        
    # Saving to csv for future reference
    csv_file_dir = f"{CSV_DIR}/extracted_info.csv"
    with open(csv_file_dir, "a") as f:
        fieldnames = ["Organization Name", "Invoice Number", "Date", "Total Amount"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if f.tell() == 0:
            writer.writeheader()
            
        writer.writerow(invoice_info_dict)
        
    return invoice_info_dict





