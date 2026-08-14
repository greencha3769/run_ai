import os
import re

# Label Studio가 내보낸 labels 폴더
LABEL_DIR = r"C:\Product_Anomaly_Detection\training\Yolo_v8\dataset\images\train"

for filename in os.listdir(LABEL_DIR):
    if not filename.endswith(".txt"):
        continue

    # good_000123 부분 추출
    match = re.search(r"(good_\d+)", filename)

    if match:
        new_name = match.group(1) + ".png"

        old_path = os.path.join(LABEL_DIR, filename)
        new_path = os.path.join(LABEL_DIR, new_name)

        os.rename(old_path, new_path)
        print(f"{filename}  ->  {new_name}")

print("완료")