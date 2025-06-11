import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

 --- モデル定義 ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # 入力: 784 -> 出力: 128
        self.fc2 = nn.Linear(128, 10)   # 入力: 128 -> 出力: 10（10クラス分類）

    def forward(self, x):
        # x の形状は (batch, 1, 28, 28) を想定。バッチサイズに合わせてリシェイプ
        x_reshaped = x.view(x.shape[0], -1)  # (batch, 784)
        h = self.fc1(x_reshaped)
        z = torch.sigmoid(h)
        y_hat = self.fc2(z)
        return y_hat

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルのロード ---
loaded_model = SimpleMLP().to(device)
# モデルのパラメータをロード（weights_only=True を指定）
loaded_model.load_state_dict(torch.load("modelwithBatch.pth", map_location=device, weights_only=True))
loaded_model.eval()  # 推論モードに設定


# --- セッションステートでアップロード画像を管理 ---
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

def clear_uploaded_file():
    st.session_state.uploaded_file = None

st.title("Digit Classification with SimpleMLP")
st.write("画像をアップロードするか、下のキャンバスに手書きで数字を描いて予測できます。")

# --- 画像アップロード ---
uploaded_file = st.file_uploader(
    "画像ファイルをアップロードしてください（PNG, JPG, JPEG）",
    type=["png", "jpg", "jpeg"],
    key="uploaded_file"
)

# --- 消去ボタン ---
if st.session_state.uploaded_file is not None:
    if st.button("アップロード画像を消去"):
        clear_uploaded_file()
        st.experimental_rerun()  # ページをリロードして反映

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

# --- 以降はアップロード画像 or キャンバス画像の処理 ---
# ...（省略）...
