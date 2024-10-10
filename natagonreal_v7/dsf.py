import os

# กำหนดพาธโฟลเดอร์ที่มีไฟล์
folder_path = "data/wow4"  # เปลี่ยนเป็นโฟลเดอร์ที่คุณต้องการ
new_file_name = "C"  # ชื่อไฟล์ใหม่ที่ต้องการ

# รับรายการไฟล์ทั้งหมดในโฟลเดอร์
files = os.listdir(folder_path)

# เริ่มการนับหมายเลข
counter = 1

# เปลี่ยนชื่อไฟล์
for file_name in files:
    old_file_path = os.path.join(folder_path, file_name)

    # แยกชื่อไฟล์ออกจากนามสกุล (รวมทั้งกรณีที่มีสองนามสกุล)
    file_base, file_extension = os.path.splitext(file_name)

    # ตรวจสอบว่ามีหลายส่วนของนามสกุลหรือไม่
    if file_base.endswith(".jpg") or file_base.endswith(".png"):  # กรณีนามสกุลที่ซับซ้อน
        complex_extension = file_base.split('.')[-1] + file_extension
        file_base = file_base[:-(len(complex_extension) - len(file_extension))]
        new_file_path = os.path.join(folder_path, f"{new_file_name}{counter}.{complex_extension}")
    else:
        new_file_path = os.path.join(folder_path, f"{new_file_name}{counter}{file_extension}")

    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {old_file_path} -> {new_file_path}")

    counter += 1
