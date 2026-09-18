import os
import cv2
import glob
import csv
import re
from pathlib import Path

# Path Configuration
RAW_DATA_DIR = "C:/Projects/alpr/data/raw"
OUTPUT_CROP_DIR = "C:/Projects/alpr/data/interim/crops"
MANIFEST_DIR = "C:/Projects/alpr/data/manifests"

os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

def normalize_text(text):
    """Uppercase and strip whitespace/hyphens per project rules."""
    return re.sub(r'[\s\-]', '', text).upper()

def parse_annotation_file(txt_path):
    """Parses UFPR and RodoSol annotation files for plate text and corner points."""
    plate_text = ""
    x, y, w, h = 0, 0, 0, 0
    
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
        # Extract plate text
        text_match = re.search(r'plate:\s*([A-Za-z0-9\- ]+)', content, re.IGNORECASE)
        if text_match:
            plate_text = normalize_text(text_match.group(1))
            
        # Extract four corner points: x1,y1 x2,y2 x3,y3 x4,y4
        corner_match = re.search(
            r'corners:\s*(\d+),(\d+)\s+(\d+),(\d+)\s+(\d+),(\d+)\s+(\d+),(\d+)', 
            content
        )
        if corner_match:
            coords = list(map(int, corner_match.groups()))
            xs = coords[0::2]
            ys = coords[1::2]
            
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            
            x, y = xmin, ymin
            w, h = xmax - xmin, ymax - ymin
            
    return plate_text, (x, y, w, h)

def process_dataset(dataset_name, file_pattern, track_id_strategy):
    manifest_rows = []
    image_paths = glob.glob(file_pattern, recursive=True)
    
    for img_path in image_paths:
        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
        
        if not os.path.exists(txt_path):
            continue
            
        plate_text, (x, y, w, h) = parse_annotation_file(txt_path)
        
        if not plate_text or w <= 0 or h <= 0:
            continue
            
        # Assign track IDs to preserve vehicle/track grouping
        if track_id_strategy == 'ufpr':
            track_id = Path(img_path).parent.name  # e.g., 'track0091'
        else:
            track_id = Path(img_path).stem  # RodoSol image name as unique ID

        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        # Safety bounds for crop coordinates
        xmin = max(0, x)
        ymin = max(0, y)
        xmax = min(img_w, x + w)
        ymax = min(img_h, y + h)
        
        crop = img[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue
        
        crop_filename = f"{dataset_name}_{track_id}_{Path(img_path).name}"
        crop_out_path = os.path.join(OUTPUT_CROP_DIR, crop_filename)
        
        cv2.imwrite(crop_out_path, crop)
        
        manifest_rows.append({
            'image_path': f"crops/{crop_filename}",
            'label': plate_text,
            'dataset': dataset_name,
            'track_id': track_id
        })
        
    return manifest_rows

# Execute Extraction
all_manifest_data = []

# Process UFPR-ALPR (.png files)
ufpr_pattern = os.path.join(RAW_DATA_DIR, "ufpr-alpr/**/*.png")
all_manifest_data.extend(process_dataset('ufpr', ufpr_pattern, track_id_strategy='ufpr'))

# Process RodoSol-ALPR (.jpg files)
rodosol_pattern = os.path.join(RAW_DATA_DIR, "rodosol-alpr/**/*.jpg")
all_manifest_data.extend(process_dataset('rodosol', rodosol_pattern, track_id_strategy='rodosol'))

# Export Master Manifest CSV
csv_path = os.path.join(MANIFEST_DIR, 'manifest_master.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image_path', 'label', 'dataset', 'track_id']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_manifest_data)

print(f"Extraction complete! Saved {len(all_manifest_data)} crops to {OUTPUT_CROP_DIR}")
print(f"Master manifest saved to {csv_path}")