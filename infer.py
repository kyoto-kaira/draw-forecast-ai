from typing import List

from PIL import Image, ImageDraw


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
            draw.line(
                points, fill="black", width=line_width
            )  # パラメータ2で線の太さを調整

    return img
