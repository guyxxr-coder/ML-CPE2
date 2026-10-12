# ML-09-Object Detection I

โปรเจกต์นี้เป็นการพัฒนาระบบตรวจจับวัตถุและการทำผิดกฎจราจร (Traffic Violation Detection) จำนวน 23 คลาส โดยใช้แนวคิดการทำงานแบบ Fast R-CNN (Region-based CNN)และ R-CNN ฉบับจำลองเพื่อการศึกษา ตัวโมเดลถูกสร้างขึ้นเอง (Custom CNN) ด้วย TensorFlow/Keras และใช้ OpenCV สำหรับการจัดการรูปภาพและการทำ Region Proposal (Sliding Window)

## 1. ภาพรวมของโปรเจกต์ Project Overview


**Features:**
โปรเจกต์นี้ครอบคลุมกระบวนการทำ Machine Learning สำหรับ Object Detection แบบพื้นฐาน ดังนี้:

Training: เทรนโมเดลจำแนกประเภทภาพ (Image Classifier) ขนาด 64x64 พิกเซล

Evaluation: ประเมินประสิทธิภาพด้วยค่า Accuracy, Precision, Recall และ F1-Score

Inference (Image): ดึงกล่อง (Bounding Box) จาก Label มาจำแนกคลาสและแสดงผล

Real-time Detection: ใช้เทคนิค Sliding Window วางกรอบและตรวจจับคลาสแบบเรียลไทม์ผ่านกล้องเว็บแคม

## 2. Dataset

The [dataset](https://www.kaggle.com/datasets/guisahanes/traffic-violation-detection-dataset) is used in YOLO annotation format.

รายชื่อ 23 คลาส **Classes:** 

person, 2. car, 3. truck, 4. bus, 5. motorcycle,

red light, 7. green light, 8. stop sign, 9. no entry, 10. no overtaking,

speed limit 20, 12. speed limit 30, 13. speed limit 40, 14. speed limit 50, 15. speed limit 60,

speed limit 70, 17. speed limit 80, 18. speed limit 100, 19. speed limit 120, 20. no left turn,

no right turn, 22. no stopping, 23. no u-turn



โปรเจกต์นี้ใช้ชุดข้อมูล Traffic Violation Detection ในรูปแบบ YOLO Annotation format

## 3. Project Structure

```text
ML-09-Object-Detection-I/
├── dataset/
│   ├── images/{train,val}/
│   ├── labels/{train,val}/│  
│   └── data.yaml
│
├── 01-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│
├── 02-Fast-R-CNN/
│   ├── train.py
│   ├── evaluate.py
│   ├── image_test.py
│   ├── realtime_test.py
│   └── outputs/
│
├── requirements.txt
└── README.md
```
 ## วิธีการติดตั้ง (Installation)

 แนะนำให้ใช้งานบน Python 3.10 และติดตั้งไลบรารีที่จำเป็นผ่านคำสั่งด้านล่างนี้:


| Script | Purpose |
|---|---|
| `train.py` | ฝึกสอนโมเดลและบันทึกจุดตรวจสอบ (Checkpoints) |
| `evaluate.py` | ประเมินประสิทธิภาพโมเดลด้วยภาพที่มีป้ายกำกับ |
| `image_test.py` | ทดสอบโมเดลที่ฝึกสอนแล้วกับรูปภาพที่เลือก |
| `realtime_test.py` | รันการทำนายแบบเรียลไทม์ผ่านเว็บแคม |
| `outputs/` | จัดเก็บโมเดลที่ฝึกสอนแล้ว, บันทึก Log และผลลัพธ์การทำนาย |



## 4.การตรวจจับวัตถุ Object Detection

**การฝึกสอนโมเดล** — ไปยังโฟลเดอร์สถาปัตยกรรมที่เลือกแล้วรันคำสั่ง

```bash
python train.py --limit 200 --epochs 50
```

**การประเมินผลโมเดล** — ประเมินประสิทธิภาพโมเดลโดยใช้ภาพทดสอบที่มีป้ายกำกับ

```bash
python evaluate.py --limit 200 --split test
```

**การทดสอบกับรูปภาพ** — เลือกภาพทดสอบ รันการทำนาย และบันทึกผลลัพธ์

```bash
python image_test.py
```

**การทดสอบแบบเรียลไทม์** — เปิดกล้องเว็บแคมหมายเลข 0 และทำการทำนายแบบเรียลไทม์ กดปุ่ม **Q** เพื่อออก:
```bash
python realtime_test.py --camera 0 --threshold 0.85 --regions 25
```

สคริปต์ต่างๆ จะต้องอ้างอิงไปยังไดเรกทอรีชุดข้อมูลและจุดตรวจสอบโมเดลที่ถูกต้อง.

**หมายเหตุการใช้งาน:** การใช้งาน R-CNN และ Fast R-CNN ในปัจจุบันเป็นการใช้แบบจำลองการจำแนกประเภทส่วนที่ตัดมา (Crop-classification baselines) อย่างง่าย แทนที่จะเป็นสถาปัตยกรรมตรวจจับวัตถุแบบสมบูรณ์ การใช้งานแบบเต็มรูปแบบจำเป็นต้องมีระบบเสนอแนะภูมิภาค (Region proposals), การถดถอยกรอบพิกัด (Bounding-box regression) และการประมวลผลหลังการตรวจจับ ชุดข้อมูล BDD100K มีคำอธิบายกรอบพิกัดในรูปแบบ YOLO สำหรับการฝึกสอนและการประเมินผล

## 5. References

- [DATASET](https://www.kaggle.com/datasets/guisahanes/traffic-violation-detection-dataset)


