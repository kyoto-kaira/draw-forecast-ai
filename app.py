from typing import List

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import CanvasResult, st_canvas

from infer import forecast, load_model_and_std
from utils import process_canvas_data


# ページの設定
st.set_page_config(
    page_title="描画予測アプリ",
    layout="wide",
)

st.title("描画予測アプリ")
st.session_state.saved_models = {}

# サイドバーにモデルパラメータのスライドバーを配置
category: str = st.selectbox("category", ["cat", "bus"])
temperature: float = st.slider(
    "temperature", min_value=0.01, max_value=10.0, value=0.5, step=0.1
)
sample_step: int = st.slider("sample step", min_value=1, max_value=10, value=1, step=1)

path_to_weight = f"weights/partial_sketchrnn_{category}_amp.pth"
path_to_std = f"weights/partial_sketchrnn_{category}_amp.json"

# Canvasの設定
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=256,
    height=256,
    drawing_mode="freedraw",
    key="canvas",
)

if type(canvas_result) is CanvasResult:
    strokes: List[List[List[float]]] = process_canvas_data(canvas_result, sample_step)

    # モデルの予測結果を表示
    if st.button("予測を実行"):
        if category not in st.session_state.saved_models:
            model, std = load_model_and_std(path_to_weight, path_to_std)
            st.session_state[category] = {"model": model, "std": std}
        model = st.session_state[category]["model"]
        std = st.session_state[category]["std"]
        prediction_image: Image.Image = forecast(strokes, model, std, temperature)
        st.image(prediction_image)
