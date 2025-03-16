# Invoice Text Extraction

## Libraries
- SpaCy (3.8.4)
- paddlepaddle (3.0.0rc1)
- paddleocr (2.9.1)
- pdf2image (1.17.0)
- opencv-python (4.11.0.86)
- fastapi (0.115.11)
- mistralai (1.5.0)
- uvicorn (0.34.0)


## Creating a virtual environment
- Navigate to your working directory and open the command prompt. Use the command below to create a virtual environment
  ```
  python -m venv <your_virtual_env_name>
  ```
- Activate the virtual environment with (on Windows):
  ```
  <your_virtual_env_name>/Scripts/activate
  ```
  
  On Mac/Linux:
  ```
  source <your_virtual_env_name>/Scripts/activate
  ```
- Install requirements
  ```
  pip install -r requirements.txt
  ```
  
## Setup PaddleOCR for text region detection
- Installing paddlepaddle
  ```
  python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  ```
- Installing paddleocr
  ```
  pip install paddleocr
  ```

## Install FastAPI and Uvicorn
```
pip install fastapi uvicorn
```
## Running the frontend
```
cd frontend
npm start
```

## Running the backend server
```
cd backend
uvicorn main:app  --reload
```


