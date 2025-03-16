import os
import csv
import shutil
from mistralai import Mistral
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from schemas import Token, TokenData, User, UserInDb

SECRET_KEY = "4f43cda95255703b0cdb8b78f7c096d3fc99715cd9961766476da0b9808382cc"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

fake_db = {
    "adarsh0609": {
        "username": "adarsh0609",
        "full_name": "Adarsh Vinod",
        "email": "adarsh@gmail.com",
        # password : adarsh123
        "hashed_password": "$2b$12$11cBpkVMrpI7sXL//.5ex./Xv090EHIOt6lTQq6QHj4P9siiklAy6",
        "disabled": False
    }
}

pwd_context = CryptContext(schemes=['bcrypt'], deprecated="auto")
oauth_2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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

'''
Authentication utility functions
'''

def verify_password(password, hashed_password):
    return pwd_context.verify(password, hashed_password)
    
def get_hashed_password(password: str):
    return pwd_context.hash(password)
    
def get_user(db, username: str):
    if username in db:
        user_data = db[username]
        return UserInDb(**user_data)

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    
    if not verify_password(password, user.hashed_password):
        return False
    
    return user

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, ALGORITHM)
    
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth_2_scheme)):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate" : "Bearer"})
    
    try:
        # payload : dict
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception
        
        token_data = TokenData(username=username)
    except JWTError:
        raise credential_exception
    
    user = get_user(fake_db, token_data.username)
    if user is None:
        raise credential_exception
    
    return user

async def get_current_active_user(current_user: UserInDb = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive User")
    
    return current_user       

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_db, form_data.username, form_data.password)
    
    if not user:
        return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate" : "Bearer"})
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token({"sub": user.username}, access_token_expires)
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/users/me/items")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id" : 1, "owner": current_user}]

'''
Endpoints for text extraction and analysis
'''

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
        key, *value = line.split(":", 1)
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


