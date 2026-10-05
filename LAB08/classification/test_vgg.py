import json
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from vgg_model import VGGNet, VGG_SMALL_CONFIG, VGG16_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# กำหนดเส้นทางโฟลเดอร์โมเดลที่ต้องการทดสอบ (เช่น VGG_SMALL_30Epochs หรือ VGG16_30Epochs)
EXP_DIR = os.path.join(OUTPUT_DIR, "VGG_SMALL_30Epochs")
N_SAMPLES = 4


def test_vgg(n_samples=N_SAMPLES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # โหลดไฟล์ข้อมูล
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    with open(os.path.join(OUTPUT_DIR, "classes.json")) as f:
        classes = json.load(f)

    # โหลดโมเดล PyTorch
    model = VGGNet(X_test.shape[1:], len(classes), VGG_SMALL_CONFIG).to(device)
    model_path = os.path.join(EXP_DIR, "vgg_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: ไม่พบไฟล์โมเดลที่ {model_path} ให้รัน main.py ก่อนครับ")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # สุ่มเลือกภาพทดสอบ
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # เตรียม Input
    inputs = torch.tensor(X_sample, dtype=torch.uint8).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        if len(classes) == 2:
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            predictions = (probs > 0.5).astype(int)
            confidence = np.where(predictions == 1, probs, 1 - probs)
        else:
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = probs.argmax(axis=1)
            confidence = probs.max(axis=1)

    # วาดรูปผลลัพธ์
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred} ({confidence[i] * 100:.0f}%)\nTrue: {true}", color=color)

        print(f"[{i + 1}] Pred: {pred:<6} True: {true:<6} conf {confidence[i] * 100:5.1f}%  {'OK' if correct else 'WRONG'}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "prediction_sample.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_vgg()