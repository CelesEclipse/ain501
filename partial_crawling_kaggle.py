import os
import time
import zipfile
import json
from pathlib import Path

JSON_PATH = Path(r"kaggle.json")

with open(JSON_PATH, 'r') as f:
    data_dict = json.load(f)

os.environ['KAGGLE_USERNAME'] = data_dict["username"]
os.environ['KAGGLE_KEY'] = data_dict["key"]

from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize and authenticate client
api = KaggleApi()
api.authenticate()

# 2. DEFINITIONS
DATASET_SLUG = "kontheeboonmeeprakob/midv500"
TARGET_SUBFOLDER = "midv500/01_alb_id" # change here

all_files = []
page_token = None

print("Fetching dataset file list with rate-limit pacing...")

while True:
    try:
        response = api.dataset_list_files(
            DATASET_SLUG,
            page_size=200,
            page_token=page_token
        )
        all_files.extend(response.files)
        
        if hasattr(response, 'nextPageToken') and response.nextPageToken:
            page_token = response.nextPageToken
            print(f"Retrieved {len(all_files)} files so far. Pausing 2 seconds to respect rate limits...")
            time.sleep(2)  # <--- Essential delay to avoid 429 errors
        else:
            break
            
    except Exception as e:
        # If we still get a 429, pause longer and try again
        if "429" in str(e):
            print("\n[Rate Limit Triggered] Kaggle blocked us. Waiting 15 seconds to cool down...")
            time.sleep(15)
            continue
        else:
            raise e

# 4. FILTER FILES
filtered_files = [f.name for f in all_files if f.name.startswith(TARGET_SUBFOLDER)]
print(f"\nTotal files found in '{TARGET_SUBFOLDER}': {len(filtered_files)}")

if len(filtered_files) == 0 and len(all_files) > 0:
    print("\n[Warning] No files matched the subfolder path.")
    print("Actual dataset paths look like this:")
    for f in all_files[:5]:
        print(f" - {f.name}")
    exit()

# 5. DEFINE OUTPUT & DOWNLOAD WITH RATE-LIMIT PACING
output_dir = Path(r"data")
output_dir.mkdir(parents=True, exist_ok=True)

for fname in filtered_files:
    print(f'\nDownloading: {fname}')
    
    api.dataset_download_file(
        dataset=DATASET_SLUG,
        file_name=fname,
        path=str(output_dir),
        quiet=False
    )
    
    time.sleep(0.5) 
    
    actual_file_path = output_dir / fname

    if actual_file_path.exists():
        if zipfile.is_zipfile(actual_file_path):
            print(f"Extracting zip archive: {actual_file_path.name}")
            with zipfile.ZipFile(actual_file_path, 'r') as zf:
                zf.extractall(output_dir)
            actual_file_path.unlink() 
        else:
            print(f"Successfully downloaded: {actual_file_path.name}")
