import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from pathlib import Path
import mediapipe as mp
from src.utils import preprocess_landmarks

class ISLDataset(Dataset):
    """
    Dataset for ISL signs: sequences of landmark frames labeled by sign.
    Expected dir: data/train/{sign_label}/{video_id}.mp4
    """
    def __init__(self, root_dir: str, sequence_length: int = 30, transform=None):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True)
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for label_dir in self.root_dir.iterdir():
            if label_dir.is_dir():
                for video_path in label_dir.glob("*.mp4"):
                    samples.append((str(video_path), label_dir.name))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        landmarks_seq = []
        frame_count = 0

        while cap.isOpened() and frame_count < self.sequence_length * 2:  # Sample frames
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 2 == 0:  # Every other frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_hands.process(rgb)
                if results.multi_hand_landmarks:
                    lm_data = []
                    for hand in results.multi_hand_landmarks:
                        lm = np.array([[p.x, p.y, p.z] for p in hand.landmark]).flatten()
                        lm_data.append(lm)
                    if len(lm_data) == 1:
                        lm_data.append(np.zeros(63))
                    combined = np.concatenate(lm_data)
                    landmarks_seq.append(preprocess_landmarks(combined))
            frame_count += 1

        cap.release()
        self.mp_hands.close()

        # Pad or truncate sequence
        seq = np.array(landmarks_seq)
        if len(seq) < self.sequence_length:
            seq = np.pad(seq, ((0, self.sequence_length - len(seq)), (0, 0)), 'constant')
        else:
            seq = seq[:self.sequence_length]

        label_idx = hash(label) % 100  # Dummy; use real label encoder
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)

def get_dataloader(root_dir: str, batch_size: int = 32, **kwargs):
    dataset = ISLDataset(root_dir, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)