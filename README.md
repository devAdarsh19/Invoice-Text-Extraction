# Invoice Text Extraction

## Libraries
- SpaCy (3.8.4)
- paddlepaddle (3.0.0rc1)
- paddleocr (2.9.1)
- pdf2image (1.17.0)
- opencv-python (4.11.0.86)


## Creating a virtual environment
- Navigate to your working directory and open the command prompt. Use the command below to create a virtual environment
  ```
  python -m venv <your_virtual_env_name>
  ```
- Activate the virtual environment with:
  ```
  <your_virtual_env_name>/Scripts/activate
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

## Setting up SpaCy for Named Entity Recognition (NER)
- Install SpaCy
  ```
  pip install spacy 
  ```
- Download the English model
  ```
  python -m spacy download en_core_web_<model_size>
  ```
  * model_size = sm - small | md - medium | lg - large

## Datasets from Robust Reading Competition
Challenge - [ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction] (https://rrc.cvc.uab.es/?ch=13&com=introduction)

Datasets obtained from 0325updated.task1train(626p) folder

## Pipeline

### File - text_extractor.ipynb
### Image Preprocessing
- Converting to grayscale
- Using adaptive thresholding to enhance visibility and improve the quality of OCR output

### Optical Character Recognition using PaddleOCR
- Initial OCR run returns individual characters
- First, sort the words by order of appearance in the image vertically, then horizontally
- Pool all the words that appear on the same line, that is, the difference between y-coordinates of words must be less than a pre-defined threshold
![OCR_result](https://github.com/user-attachments/assets/d6073df3-c602-4e25-9c5a-fff21f3797d3)


### Regex for field extraction (Inefficient)
- Extracting store name, date, invoice number, and total amount using regex patterns

### File - nlp_demo.ipynb
### Model training
- Load the English model
- Provide training data in an appropriate format
- Train the model
- Evaluate model performance on evaluation set
- Test on custom data








  
