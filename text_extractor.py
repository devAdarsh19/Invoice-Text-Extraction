from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import re

def merge_ocr_results(ocr_results, x_threshold=30, y_threshold=10):
    
    # Sort OCR results
    ocr_results.sort(key=lambda item: (item[0][0][1], item[0][0][0]))
    
    # print("*************Debugging***********")
    # for idx in range(len(ocr_results)):
    #     res = ocr_results[idx]
    #     for line in res:
    #         print(line)
    
    merged_result = []
    for idx in range(len(ocr_results)):
        res = ocr_results[idx]
        for box, (text, conf) in res:
            _, y1 = box[0]
            x2, _ = box[1]
            
            if not merged_result:
                merged_result.append([box, (text, conf)])
                continue
            
            prev_box, (prev_text, prev_conf) = merged_result[-1]
            prev_x1, prev_y1 = prev_box[0]
            
            # print(f"DEBUG : x1 - prev_x1 = {x1 - prev_x1}")
            # print(f"DEBUG : y1 - prev_y1 = {y1 - prev_y1}")
            
            if abs(y1 - prev_y1) < y_threshold and abs(x2 - prev_x1) > x_threshold:
                merged_txt = prev_text + " " + text
                avg_conf = (conf + prev_conf) / 2
                prev_box[1], prev_box[2] = box[1], box[2]
                merged_result[-1] = [prev_box, (merged_txt, avg_conf)]
            else:
                merged_result.append([box, (text, conf)])
            
    return merged_result

def extract_fields(merged_result: list):
    
    date, receipt_no, total_amt, store_name = None, None, None, None
    
    date_pattern = re.compile(r'\b(?:Date[:\s]*)?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b|Date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', re.IGNORECASE)    
    # date_pattern = re.compile(r'\b(?:Date[:\s]*)?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', re.IGNORECASE)
    total_amount_pattern = re.compile(r'TOTAL(?: AMOUNT|AMT\.?|:)?\s*(?:RM|USD|\$)?\s*(\d+\.\d{2})', re.IGNORECASE)
    receipt_no_pattern = re.compile(r'(?:Receipt No|Invoice No|Invoice#|Inv#|Bill No|Document No|Room No|Doc No).*?(\S+)')
    store_name_keywords = ["HOME", "STORE", "SHOP", "MARKET", "GIFT", "MART", "RETAIL"]
    
    for entry in merged_result:
        text = entry[1][0].strip()
        
        if not date:
            date_match = date_pattern.search(text)
            if date_match:
                date = date_match.group()
                date = date.lstrip('Date')
                
        if not total_amt:
            total_match = total_amount_pattern.search(text)
            if total_match:
                total_amt = total_match.group(1)
        
        # Extract Receipt No.
        if not receipt_no:
            receipt_match = receipt_no_pattern.search(text)
            if receipt_match:
                receipt_no = receipt_match.group(1)
        
        # Extract Store Name
        if not store_name:
            if any(keyword in text.upper() for keyword in store_name_keywords):
                store_name = text
                
    return {
        "Receipt No": receipt_no,
        "Date": date,
        "Total Amount": total_amt,
        "Store Name": store_name
    }
            


if __name__ == "__main__":

    ocr = PaddleOCR(use_angle_cls=True, lang="en")

    img_path = "./train_datasets/X51005230617.jpg"
    image = cv2.imread(img_path)

    result = ocr.ocr(img_path, cls=True)
    merged_result = merge_ocr_results(result)

    for res in merged_result:
        print(res)
        
    extracted_fields = extract_fields(merged_result)
    print(extracted_fields)
    
    boxes = [res[0] for res in merged_result]
    texts = [res[1][0] for res in merged_result]
    scores = [res[1][1] for res in merged_result]
    
    # Draw OCR results on the image
    image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path="C:\Windows\Fonts\Arial.ttf")

    # Convert to OpenCV format
    image_with_boxes = cv2.cvtColor(np.array(image_with_boxes), cv2.COLOR_RGB2BGR)
    
    cv2.imshow("OCR Results", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    