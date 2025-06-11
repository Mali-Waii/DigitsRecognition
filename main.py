import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --- モデル定義 ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x_reshaped = x.view(x.shape[0], -1)
        h = self.fc1(x_reshaped)
        z = torch.sigmoid(h)
        y_hat = self.fc2(z)
        return y_hat

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルのロード ---
loaded_model = SimpleMLP().to(device)
loaded_model.load_state_dict(torch.load("modelwithBatch.pth", map_location=device))
loaded_model.eval()

st.title("Digit Classification with SimpleMLP")
st.write("画像をアップロードするか、下のキャンバスに手書きで数字を描いて予測できます。")

# --- 画像アップロード ---
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください（PNG, JPG, JPEG）", type=["png", "jpg", "jpeg"])

# --- 手書きキャンバス ---
st.write("または、下のキャンバスに数字を描いてください（黒で太めに描くと認識しやすいです）")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


def preprocess_image(image: Image.Image):
    # グレースケール化・28x28リサイズ・正規化
    image = image.convert("L").resize((28, 28))
    image_np = np.array(image) / 255.0
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image_tensor

def predict(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = loaded_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.cpu().numpy()

image_tensor = None
input_source = None

# 画像アップロード優先
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード画像", use_column_width=False)
    image_tensor = preprocess_image(image)
    input_source = "upload"
elif canvas_result.image_data is not None:
    # Streamlit Draw Canvasの画像データはRGBAなのでPIL Imageに変換
    image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
    st.image(image, caption="手書き画像", use_column_width=False)
    image_tensor = preprocess_image(image)
    input_source = "canvas"

if image_tensor is not None:
    predicted_class, probabilities = predict(image_tensor)
    st.write("**予測されたクラス:**", predicted_class)
    st.write("**各クラスの確率:**", probabilities)

    # Matplotlibで画像と予測結果を表示
    fig, ax = plt.subplots()
    ax.imshow(image_tensor.squeeze().cpu().numpy(), cmap="gray")
    ax.set_title(f"Prediction: {predicted_class}")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("画像がアップロードされていないか、キャンバスに何も描かれていません。")

