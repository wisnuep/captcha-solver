import streamlit as st
import torch
import gdown
import os
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F

# ── Paksa tema putih ───────────────────────────────────────
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #000000; }
        .stMetric { background-color: #f0f2f6; border-radius: 10px; padding: 10px; }
        .confidence-box { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# ── Model Definition ───────────────────────────────────────
class CaptchaModel(pl.LightningModule):
    def __init__(self, num_classes=6, num_characters=21, input_channels=1, learning_rate=1e-3):
        super(CaptchaModel, self).__init__()
        self.num_classes = num_classes
        self.num_characters = num_characters
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
        self.resnet50 = models.resnet50()
        self.resnet50.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.3),
            nn.Linear(1024, self.num_characters * self.num_classes),
        )

    def forward(self, x):
        return self.resnet50(x)

# ── Karakter ───────────────────────────────────────────────
DECODING_DICT = {
    0:"a", 1:"f", 2:"e", 3:"c", 4:"b", 5:"h", 6:"v", 7:"z",
    8:"2", 9:"x", 10:"g", 11:"m", 12:"r", 13:"u", 14:"p",
    15:"s", 16:"d", 17:"n", 18:"6", 19:"k", 20:"t"
}

# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "best_model.ckpt"
    if not os.path.exists(model_path):
        with st.spinner("⏳ Mendownload model dari Google Drive..."):
            file_id = "1ikhPiNF2HQkwD5sf6qXdziGHZcqmeQlG"
            gdown.download(
                f"https://drive.google.com/uc?export=download&id={file_id}",
                model_path,
                quiet=False
            )
    model = CaptchaModel.load_from_checkpoint(model_path, map_location="cpu")
    model.eval()
    return model

# ── Prediksi + Confidence ──────────────────────────────────
def predict(model, image):
    transform = transforms.Compose([
        transforms.Resize((50, 250)),
        transforms.CenterCrop((50, 250)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.7570), (0.3110)),
    ])
    img_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        out = model(img_tensor)

    # Reshape → (6 posisi, 21 karakter)
    logits = out.reshape(21, 6).T         # (6, 21)
    probs  = F.softmax(logits, dim=1)     # probabilitas tiap posisi

    predicted_indices = probs.argmax(dim=1)  # index karakter terpilih
    predicted_chars   = [DECODING_DICT[i.item()] for i in predicted_indices]
    confidences       = [probs[i, predicted_indices[i]].item() * 100 for i in range(6)]

    result     = "".join(predicted_chars)
    avg_conf   = sum(confidences) / len(confidences)
    return result, confidences, avg_conf

# ── Tampilan ───────────────────────────────────────────────
st.set_page_config(page_title="CAPTCHA Solver", page_icon="🤖", layout="centered")
st.title("🤖 CAPTCHA Solver")
st.write("Upload gambar CAPTCHA, model akan menebak teksnya.")
st.markdown("---")

model = load_model()

uploaded_file = st.file_uploader("📁 Upload gambar CAPTCHA (.jpeg / .png)", type=["jpeg", "jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    with st.spinner("🔍 Sedang menganalisis..."):
        result, confidences, avg_conf = predict(model, image)

    st.markdown("---")

    # Hasil prediksi
    st.success(f"✅ Hasil Prediksi: **{result.upper()}**")
    st.metric(label="Teks CAPTCHA", value=result.upper())

    st.markdown("---")

    # Confidence score keseluruhan
    st.subheader("📊 Confidence Score")
    color = "green" if avg_conf >= 90 else "orange" if avg_conf >= 70 else "red"
    st.markdown(f"**Rata-rata Confidence: :{color}[{avg_conf:.2f}%]**")
    st.progress(avg_conf / 100)

    # Confidence per karakter
    st.markdown("**Detail per Karakter:**")
    cols = st.columns(6)
    for i, (char, conf) in enumerate(zip(result.upper(), confidences)):
        with cols[i]:
            st.metric(label=f"Char {i+1}", value=char)
            color = "🟢" if conf >= 90 else "🟡" if conf >= 70 else "🔴"
            st.caption(f"{color} {conf:.1f}%")
