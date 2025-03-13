import os
import asyncio
import nest_asyncio
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Patch the event loop
nest_asyncio.apply()

# Disable file watcher for PyTorch modules
st.get_option("server.fileWatcherType") == "none"

# ✅ Streamlit Branding
st.set_page_config(page_title="Live Face Recognition", page_icon="👀")

st.title("🔴 Live Face Recognition | CareerUpskillers")
st.write("🚀 Developed by [CareerUpskillers](https://www.careerupskillers.com)")
st.write("📞 Contact: WhatsApp 917975931377")

# ✅ Step 1: Load Face Recognition Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_recognition_model.pth")

# Debug statement
st.write(f"Model Path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found!")
    st.stop()

# Load Model
class FaceRecognitionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Change this to match your model
model = FaceRecognitionModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ Define Image Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Define Webcam Stream Processing
class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            st.write("✅ Frame captured successfully!")  # Debug statement

            # Convert to PIL Image
            pil_img = Image.fromarray(img)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            st.write(f"Image Tensor Shape: {img_tensor.shape}")  # Debug statement

            # Predict Class
            with torch.no_grad():
                output = model(img_tensor)
                st.write(f"Model Output: {output}")  # Debug statement
                predicted_class = torch.argmax(output).item()

            # Label (known or unknown)
            label = "Known" if predicted_class == 0 else "Unknown"
            st.write(f"Prediction: {label}")  # Debug statement

            # Display Label on Image
            cv2.putText(img, f"Prediction: {label}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            st.error(f"❌ Error in transform method: {e}")  # Debug statement
            return frame  # Return the original frame if an error occurs

# ✅ Start Webcam Stream
webrtc_streamer(
    key="face-recognition",
    video_transformer_factory=FaceRecognitionTransformer,
    async_processing=True  # Enable async processing
)
