import os
from PIL import Image

# กำหนดพาธโฟลเดอร์ที่มีไฟล์ .jpg
folder_path = "data/wow4"  # เปลี่ยนเป็นโฟลเดอร์ที่คุณต้องการ

# รับรายการไฟล์ทั้งหมดในโฟลเดอร์ที่มีนามสกุล .jpg
files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

# แปลงไฟล์จาก .jpg เป็น .png
for file_name in files:
    # กำหนดพาธไฟล์ต้นฉบับและไฟล์ใหม่
    jpg_file_path = os.path.join(folder_path, file_name)
    png_file_name = file_name.replace(".jpg", ".png")
    png_file_path = os.path.join(folder_path, png_file_name)

    # เปิดไฟล์ .jpg และแปลงเป็น .png
    with Image.open(jpg_file_path) as img:
        img.save(png_file_path)
        print(f"Converted: {jpg_file_path} -> {png_file_path}")
