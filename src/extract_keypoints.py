import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import math

mp_pose = mp.solutions.pose

def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))

def extract_angles(landmarks):
    lm = landmarks
    def pt(idx):
        return [lm[idx].x, lm[idx].y]

    angles = [
        compute_angle(pt(11), pt(13), pt(15)),
        compute_angle(pt(12), pt(14), pt(16)),
        compute_angle(pt(13), pt(11), pt(23)),
        compute_angle(pt(14), pt(12), pt(24)),
        compute_angle(pt(11), pt(23), pt(25)),
        compute_angle(pt(12), pt(24), pt(26)),
        compute_angle(pt(23), pt(25), pt(27)),
        compute_angle(pt(24), pt(26), pt(28)),
        compute_angle(pt(25), pt(27), pt(29)),
        compute_angle(pt(26), pt(28), pt(30)),
        compute_angle(pt(11), pt(12), pt(24)),
        compute_angle(pt(12), pt(11), pt(23)),
        compute_angle(pt(23), pt(24), pt(26)),
        compute_angle(pt(24), pt(23), pt(25)),
        compute_angle(pt(11), pt(23), pt(24)),
    ]
    return angles

def extract_keypoints_from_dataset(data_dir, output_csv):
    rows = []
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    classes = sorted(os.listdir(data_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]

    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

    for label in tqdm(classes, desc="Classes"):
        class_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(class_dir):
            if not img_file.endswith(valid_exts):
                continue
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)
            if not result.pose_landmarks:
                continue
            lm = result.pose_landmarks.landmark
            raw = []
            for point in lm:
                raw.extend([point.x, point.y, point.z, point.visibility])
            angles = extract_angles(lm)
            rows.append(raw + angles + [label])

    pose.close()
    cols = [f"kp_{i}" for i in range(132)] + [f"angle_{i}" for i in range(15)] + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} samples to {output_csv}")

if __name__ == "__main__":
    extract_keypoints_from_dataset("data/raw/train", "data/processed/train_keypoints.csv")
    extract_keypoints_from_dataset("data/raw/test",  "data/processed/test_keypoints.csv")