import cv2
import numpy as np
import json
from pathlib import Path

img_root = Path("01_alb_id/images")
gt_root  = Path("01_alb_id/ground_truth")
out_img  = Path("data/images")
out_mask = Path("data/masks")

out_img.mkdir(parents=True, exist_ok=True)
out_mask.mkdir(parents=True, exist_ok=True)

for json_path in gt_root.rglob("*.json"):
    with open(json_path) as f:
        data = json.load(f)
    
    if "quad" not in data:
        print(f"Skipping : {json_path.name}")
        continue

    quad = data["quad"]

    rel   = json_path.relative_to(gt_root).with_suffix(".tif")
    img_path = img_root / rel

    if not img_path.exists():
        print(f"Missing: {img_path}")
        continue

    # Read image to get dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Can't read: {img_path}")
        continue

    h, w = img.shape[:2]

    # Generate mask
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = np.array(quad, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Save both with flat unique name
    name = f"{json_path.parent.name}_{json_path.stem}"

    cv2.imwrite(str(out_img  / f"{name}.jpg"), img)   # convert tif → jpg
    cv2.imwrite(str(out_mask / f"{name}.png"), mask)

print("Done")
