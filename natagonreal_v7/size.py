from PIL import Image
import os


def resize_images(input_folder, output_folder, size=(300, 300)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img = img.resize(size)
        img.save(os.path.join(output_folder, filename))
        print(f"Resized and saved {filename}")


# โฟลเดอร์ที่มีรูปภาพต้นฉบับ
input_folder = './wow'

# โฟลเดอร์ที่จะบันทึกรูปภาพที่ปรับขนาดแล้ว
output_folder = './wow4'

# เรียกใช้งานฟังก์ชันเพื่อปรับขนาดรูปทั้งหมดในโฟลเดอร์ input_images
resize_images(input_folder, output_folder, size=(300, 300))
