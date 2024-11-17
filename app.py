from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

# ページの設定
st.set_page_config(
    page_title="描画予測アプリ",
    layout="wide",
)

st.title("描画予測アプリ")

# サイドバーにモデルパラメータのスライドバーを配置
st.sidebar.header("モデルパラメータ設定")
param1: float = st.sidebar.slider("パラメータ1", 0.0, 1.0, 0.5, 0.01)
param2: int = st.sidebar.slider("パラメータ2", 0, 100, 50, 1)
param3: float = st.sidebar.slider("パラメータ3", 0.0, 10.0, 5.0, 0.1)

# Canvasの設定
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=400,
    height=400,
    drawing_mode="freedraw",
    key="canvas",
)


def process_canvas_data(canvas_data: Optional[Any]) -> List[List[List[float]]]:
    """
    ストロークデータを処理し、JSON構造に変換する。

    Parameters
    ----------
    canvas_data : Optional[Any]
        Canvasから取得したデータ。存在しない場合は空リストを返す。

    Returns
    -------
    List[List[List[float]]]
        各ストロークのx座標、y座標、タイムスタンプのリスト。
    """
    if not canvas_data:
        return []

    strokes_json: List[List[List[float]]] = []
    objects: List[Dict[str, Any]] = canvas_data.json_data.get("objects", [])

    for stroke in objects:
        if stroke.get("type") == "path":
            x_coords: List[float] = []
            y_coords: List[float] = []
            t_coords: List[int] = []
            points: List[List[float]] = stroke.get("path", [])
            for i, point in enumerate(points):
                if len(point) >= 3:
                    x_coords.append(point[1])
                    y_coords.append(point[2])
                    t_coords.append(i)
            strokes_json.append([x_coords, y_coords, t_coords])

    return strokes_json


strokes: List[List[List[float]]] = process_canvas_data(canvas_result)


def dummy_model(
    strokes_json: List[List[List[float]]],
    param1: float,
    param2: int,
    param3: float,
) -> Image.Image:
    """
    ダミーの予測モデル。描画を反転させた画像を返す。

    Parameters
    ----------
    strokes_json : List[List[List[float]]]
        処理されたストロークデータ。
    param1 : float
        モデルパラメータ1。
    param2 : int
        モデルパラメータ2。
    param3 : float
        モデルパラメータ3。

    Returns
    -------
    Image.Image
        予測結果の画像。
    """
    if not strokes_json:
        return Image.new("RGB", (400, 400), color="white")

    # 描画データから画像を再生成
    img = Image.new("RGB", (400, 400), color="white")
    draw = ImageDraw.Draw(img)

    for stroke in strokes_json:
        x, y, _ = stroke
        points = list(zip(x, y))
        if len(points) > 1:
            line_width = max(1, int(param2 / 50))
            draw.line(points, fill="black", width=line_width)  # パラメータ2で線の太さを調整

    return img


# モデルの予測結果を表示
st.subheader("予測結果")
if st.button("予測を実行"):
    prediction_image: Image.Image = dummy_model(strokes, param1, param2, param3)
    st.image(prediction_image, use_container_width=True)
