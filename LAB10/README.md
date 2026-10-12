# ML-10-Object Detection II

โปรเจกต์นี้เป็นการนำเสนอการตรวจจับวัตถุด้วย YOLOv8 โดยใช้ชุดข้อมูล Traffic Violation Detection โปรเจกต์นี้ใช้ตรวจจับ ป้ายบนท้องถนน โดยครอบคลุมตั้งแต่การเทรน (Training), การประเมินผล (Evaluation), การอนุมานภาพ (Image inference), และการตรวจจับผ่านเว็บแคมแบบเรียลไทม์ (Real-time webcam detection)
 
## 1. ภาพรวมของโปรเจกต์ (Project Overview)

โปรเจกต์นี้ทำการปรับแต่ง (Fine-tunes) โมเดล **Ultralytics YOLOv8n** ที่ผ่านการเทรนเบื้องต้นมาแล้ว ให้สามารถตรวจจับบริบทของการจราจรและการทำผิดกฎจราจรได้

**คุณสมบัติ (Features):**
- ตรวจจับวัตถุและการทำผิดกฎจราจรทั้งหมด 23 คลาส (เช่น รถยนต์, มอเตอร์ไซค์, ไม่สวมหมวกกันน็อก ฯลฯ)
- ปรับแต่ง (Fine-tuning) โมเดล YOLOv8n ด้วย PyTorch
- ประเมินผลโมเดลโดยใช้ค่า Precision, Recall, mAP50, และ mAP50-95
- รองรับการทดสอบผลลัพธ์ทั้งบนรูปภาพนิ่ง และการตรวจจับผ่านวิดีโอ/กล้องแบบเรียลไทม์ (ปรับขนาดหน้าต่างแสดงผลให้เหมาะสมแล้ว)
- บันทึกโมเดลและประวัติการเทรน (Training logs) อัตโนมัติ

## 2. ชุดข้อมูล Dataset

The [Traffic Violation Detection](https://www.kaggle.com/datasets/guisahanes/traffic-violation-detection-dataset) ในรูปแบบ YOLO Annotation format.

| การแบ่งข้อมูล (Split) | รายละเอียด |
|---|---|
| Training | โฟลเดอร์ `dataset/images/train` |
| Validation | โฟลเดอร์ `dataset/images/val` |
| Classes | จำนวน 23 คลาส |

*(หมายเหตุ: สามารถดูรายชื่อคลาสทั้งหมดได้ในไฟล์ `dataset/data.yaml`)*

Each YOLO label line contains a class ID and normalized bounding-box coordinates.

## 3. โครงสร้างโปรเจกต์ (Project Structure)

```text
ML-10-Object-Detection-II/
├── dataset/
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   │
│   └── data.yaml
├── train.py
├── evaluate.py
├── image_test.py
├── realtime_test.py
├── requirements.txt
└── README.md
```

| Script | Purpose |
|---|---|
| `train.py` | โหลดโมเดล YOLOv8n และทำการเทรนโมเดลด้วยชุดข้อมูล |
| `evaluate.py` | ประเมินประสิทธิภาพของโมเดลด้วยชุดข้อมูล Validation |
| `image_test.py` | สุ่มรูปภาพจากชุด Validation มาทดสอบการตรวจจับและแสดงผล |
| `realtime_test.py` | โหลดโมเดล best.pt มาทดสอบการตรวจจับกับไฟล์วิดีโอหรือกล้องเว็บแคม |


## 4. Object Detection

**Image detection**  สุ่มเลือกรูปภาพทดสอบ, วาดกรอบและป้ายกำกับ, และบันทึกผลลัพธ์ที่ตรวจจับได้:

```bash
python image_test.py
```

**Real-time detection** — เปิดกล้องเว็บแคมดัชนี 0 และแสดงผลภาพที่ถูกตรวจจับ กด Q เพื่อออก:

```bash
python realtime_test.py
```



## 5. เครื่องมือและไลบรารีที่ใช้ (Dependencies)
Python 3.x

Ultralytics (YOLOv8)

OpenCV (cv2)

PyTorch



## 6. References

- [Dataset](https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo?resource=download/)
- [Videotest](https://www.pexels.com/search/videos/?q=traffic%2C%20dashcam%2C%20driving%20city)



