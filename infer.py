import json
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from src.model import MODEL_CONFIG, SketchRNN


def load_model_and_std(
    path_to_weight: str, path_to_std: str
) -> Tuple[SketchRNN, List[float]]:
    device = torch.device("cpu")

    model = SketchRNN(device=device, **MODEL_CONFIG)
    model.load_state_dict(
        torch.load(path_to_weight, map_location=device, weights_only=True)
    )
    model.eval()

    with open(path_to_std, "r") as f:
        std_xy = json.load(f)["std"]

    return model, std_xy


def forecast(
    strokes_json: List[List[List[float]]],
    model: SketchRNN,
    std_xy: List[float],
    temperature: float,
) -> Image.Image:
    """
    予測結果の画像を生成する。

    Parameters
    ----------
    strokes_json : List[List[List[float]]]
        処理されたストロークデータ。
    model : SketchRNN
        学習済みのモデル。
    std_xy : List[float]
        標準偏差。
    temperature : float
        サンプリング時の温度。

    Returns
    -------
    Image.Image
        予測結果の画像。
    """
    point_init = strokes_json[0][0][0], strokes_json[0][1][0]
    condition = preprocess_strokes(strokes_json, std_xy).to(model.device)
    output = model.cond_sampling(condition=condition, temperature=temperature)
    all_strokes = np.concatenate(
        [condition.cpu().numpy(), output.cpu().numpy()[len(condition) :]]
    )
    img = point_draw(all_strokes, point_init, std_xy)
    img_pil = Image.fromarray(img)

    return img_pil


def preprocess_strokes(
    strokes_json: List[List[List[float]]], std_xy: List[float]
) -> torch.Tensor:
    """
    ストロークデータを前処理する。

    Parameters
    ----------
    strokes_json : List[List[List[float]]]
        処理されたストロークデータ。
    std_xy : List[float]
        標準偏差。

    Returns
    -------
    torch.Tensor
        前処理されたストロークデータ。

    Notes
    -----
    データ形式は、[差分ｘ, 差分y, ペン状態×3]の５次元です。
    - $p_1$: ペンが紙に触れていて、次の点と現在の点の間で線が引かれる場合に1
    - $p_2$: 現在の点の後でペンが紙から離れ、線が引かれない場合に1
    - $p_3$: 描画が終了し、現在の点を含む後続の点が描画されない場合に1

    strokes_jsonの形式は以下の通り。
    [
        [[x^0_0, x^0_1, x^0_2, ..., x^0_n1], [y^0_0, y^0_1, y^0_2, ..., y^0_n1]],
        [[x^1_0, x^1_1, x^1_2, ..., x^1_n2], [y^1_0, y^1_1, y^1_2, ..., y^1_n2]],
        ...
        [[x^m_0, x^m_1, x^m_2, ..., x^m_n3], [y^m_0, y^m_1, y^m_2, ..., y^m_nm]],
    ]
    return
    [
        [dx^0_1, dy^0_1, 1, 0, 0],
        [dx^0_2, dy^0_2, 1, 0, 0],
        ...
        [dx^0_n1, dy^0_n1, 1, 0, 0],
        [dx^1_1, dy^1_1, 0, 1, 0],
        ...
        [dx^m_nm, dy^m_nm, 0, 0, 1],
    ]
    """
    condtion = []
    std_x, std_y = std_xy
    x_prev, y_prev = None, None
    for stroke in strokes_json:
        x = np.array(stroke[0])
        y = np.array(stroke[1])
        for x_i, y_i in zip(x, y):
            if x_prev is None:
                dx = 0
                dy = 0
            else:
                dx = (x_i - x_prev) / std_x
                dy = (y_i - y_prev) / std_y
            condtion.append([dx, dy, 1, 0, 0])
            x_prev = x_i
            y_prev = y_i
        condtion[-1][2] = 0
        condtion[-1][3] = 1

    condtion = np.array(condtion, dtype=np.float32)

    return torch.tensor(condtion, dtype=torch.float32)


def point_draw(
    point: np.ndarray, point_init: List[float], std_xy: List[float]
) -> np.ndarray:
    img = np.full((256, 256), 255, dtype=np.uint8)
    x, y = point_init
    pen = 1

    for p in point:
        dx = p[0] * std_xy[0]
        dy = p[1] * std_xy[1]
        if pen:
            cv2.line(img, (int(x), int(y)), (int(x + dx), int(y + dy)), (0), 1)
        x += dx
        y += dy
        pen = int(p[2])

    return img
