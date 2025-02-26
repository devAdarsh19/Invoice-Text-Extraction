import os
import cv2

train_folder = "./train_datasets"

class_labels = {
    'DATE': 0,
    'DOCUMENT NO': 1,
    'BILL NO.': 2,
    'INVOICE NO.': 3,
    'CASHIER': 4,
    'TOTAL': 5,
}

def convert_to_yolo_format(img_height, img_width, coords):
    x_coord = coords[::2]
    y_coord = coords[1::2]
    
    x_min, y_min = min(x_coord), min(y_coord)
    x_max, y_max = max(x_coord), max(y_coord)
    
    x_center = (x_max + x_min) / (2 * img_width)
    y_center = (y_max + y_min) / (2 * img_height)
    box_width = (x_max + x_min) / img_width
    box_height = (y_max + y_min) / img_height
    
    return x_center, y_center, box_width, box_height
    

for filename in os.listdir(train_folder):
    if filename.endswith(".txt"):
        txt_path = os.path.join(train_folder, filename)
        img_name = filename.replace(".txt", ".jpg")
        img_path = os.path.join(train_folder, img_name)
        
        try:
            img = cv2.imread(img_path)
            
            img_height, img_width, _ = img.shape
        except:
            print("An error occured")
        
        with open(txt_path, 'r') as f:
            lines = f.readlines() 
            
        yolo_annotation_lines = []
        
        for line in lines:
            parts = line.strip().split(',')
            # if parts[-1] == '':
            #     parts.pop()
            print(f"parts : {parts} for {filename}")
            coords = list(map(int, parts[0:8]))
            label = ','.join(parts[8:])
            
            if label not in class_labels.keys():
                print("Class not in class_labels")
                continue
            
            class_id = class_labels[label]
            x_center, y_center, bbox_width, bbox_height = convert_to_yolo_format(img_height, img_width, coords)
            
            yolo_annotation_lines.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")
            
        yolo_txt_path = os.path.join(train_folder, filename)
        with open(yolo_txt_path, 'w') as f:
            f.writelines('\n'.join(yolo_annotation_lines))
            
        print(f"Converted {filename} to YOLO annotation format")
            
            
        