import tensorflow as tf
import numpy as np
import soundfile as sf
from tensorflow_tts.inference import AutoProcessor, TFAutoModel
import os
import torch
from torch.utils.data
import (Dataset, DataLoader)
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import Wav2Vec2ForSequenceClassification
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1、生成ai语音

#加载预训练模型
processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
wavenet = TFAutoModel.from_pretrained("tensorspeech/tts-wavenet-ljspeech-en")

def synthesize_wavenet(text, output_file):
    # 文本转梅尔频谱
    input_ids = processor.text_to_sequence(text)
    mel_outputs, _, _ = tacotron2.inference(tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                                            tf.convert_to_tensor([len(input_ids)], dtype=tf.int32),
                                            tf.convert_to_tensor([0], dtype=tf.int32))

    # 梅尔频谱转换为音频波形
    audio = wavenet.inference(mel_outputs)[0, :, 0].numpy()

    # 保存音频
    sf.write(output_file, audio, 22050)
    print(f"WaveNet TTS synthesis complete. Output saved to {F:/ntu/sem2/h/sound1}")

# 生成语音
synthesize_wavenet("This is a test sentence using WaveNet.", "wavenet_output.wav")


#2、提取MFCC特征

class AudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir, sample_rate=16000, n_mfcc=40):
        self.real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.wav')]
        self.fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.wav')]
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)  # 真实：0，合成：1
        self.files = self.real_files + self.fake_files
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]

        # 加载音频
        audio, sr = librosa.load(F:/ntu/sem2/h/sound2, sr=self.sample_rate)

        # 提取 MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        return mfcc, torch.tensor(label, dtype=torch.long)

# 数据加载
dataset = AudioDataset("dataset/common_audio", "dataset/generated_audio")
dataloader= DataLoader(dataset, batch_size=32, shuffle=True)

#3、训练RawNet2

class RawNet2(nn.Module):
    def __init__(self, input_dim=40, num_classes=2):
        super(RawNet2, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.mean(x, dim=-1)  # 全局平均池化
        x = self.fc(x)
        return x

model = RawNet2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    for mfccs, labels in dataloader:
        mfccs, labels = mfccs.to(device), labels.to(device)

        # 前向传播
        outputs = model(mfccs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#4、训练Transformer

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=2  # 2 分类（真实 vs 合成）
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):
    for mfccs, labels in dataloader:
        mfccs, labels = mfccs.to(device), labels.to(device)

        # 计算 loss
        outputs = model(input_values=mfccs).logits
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#5、评估模型

model.eval()
true_labels, pred_labels = [], []

with torch.no_grad():
    for mfccs, labels in dataloader:
        mfccs, labels = mfccs.to(device), labels.to(device)

        outputs = model(mfccs)
        preds = torch.argmax(outputs, dim=1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

accuracy = accuracy_score(true_labels, pred_labels)
print(f"Test Accuracy: {accuracy:.4f}")

#6、使用模型进行实时监测

def detect_audio(file_path):
    model.eval()
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

    output = model(mfcc)
    pred = torch.argmax(output, dim=1).item()

    if pred == 1:
        print("This is a synthetic (AI-generated) audio.")
    else:
        print("This is a real human audio.")

detect_audio("test_audio.wav")
