
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FRAMES  = 16
IMG_SIZE    = 224
NUM_CLASSES = 14

CLASS_NAMES = [
    "abuse", "arrest", "arson", "assault", "burglary",
    "explosion", "fighting", "normal", "road_accident",
    "robbery", "shooting", "shoplifting", "stealing", "vandalism"
]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

class ResNeXt50Classifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        backbone        = models.resnext50_32x4d(weights=None)
        self.features   = nn.Sequential(*list(backbone.children())[:-2])
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def load_model(checkpoint_path):
    model = ResNeXt50Classifier().to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    return model

def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

@torch.no_grad()
def predict(video_path, model, top_k=3):
    frames    = extract_frames(video_path)
    tensors   = [transform(image=f)["image"] for f in frames]
    batch     = torch.stack(tensors).to(DEVICE)
    avg_probs = torch.softmax(model(batch), dim=1).mean(0)
    top_vals, top_idxs = torch.topk(avg_probs, k=top_k)
    return {
        "predicted_class" : IDX_TO_CLASS[top_idxs[0].item()],
        "confidence"      : round(top_vals[0].item() * 100, 2),
        "top_k"           : [
            {"class": IDX_TO_CLASS[i.item()], "confidence": round(v.item() * 100, 2)}
            for v, i in zip(top_vals, top_idxs)
        ]
    }

if __name__ == "__main__":
    import sys
    import os

    model      = load_model(r"./resnext50_cctv_anomaly.pth")
    video_path = r"C:\\Users\\rgdar\\OneDrive\\Desktop\\Academics\\ASC\\Sem-6\\NNDL_CASE_STUDY\\Videos\\Assault\\Assault005_x264.mp4"

    # ── Debug: verify file exists & is readable ───────────────────────────
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        print(f"   Absolute path : {os.path.abspath(video_path)}")
        sys.exit(1)
    else:
        print(f"✅ File found    : {os.path.abspath(video_path)}")
        print(f"   Size          : {os.path.getsize(video_path) / (1024*1024):.2f} MB")

    # ── Debug: test OpenCV can open it ────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ OpenCV cannot open this video.")
        print("   Try installing: pip install opencv-python")
    else:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        print(f"✅ OpenCV opened : {total} frames  |  {fps:.1f} FPS")
    cap.release()

    # ── Run prediction ────────────────────────────────────────────────────
    result = predict(video_path, model)
    print(f"\nPrediction : {result['predicted_class'].upper()}")
    print(f"Confidence : {result['confidence']}%")
    for r in result["top_k"]:
        print(f"  {r['class']:<18} {r['confidence']:>6.2f}%")
