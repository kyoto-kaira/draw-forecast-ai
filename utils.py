from typing import Any, Dict, List, Optional

def process_canvas_data(canvas_data: Optional[Any], sample_step: int) -> List[List[List[float]]]:
    """
    ストロークデータを処理し、JSON構造に変換する。

    Parameters
    ----------
    canvas_data : Optional[Any]
        Canvasから取得したデータ。存在しない場合は空リストを返す。

    Returns
    -------
    List[List[List[float]]]
        各ストロークのx座標、y座標のリスト。
    """
    if not canvas_data:
        return []

    strokes_json: List[List[List[float]]] = []
    objects: List[Dict[str, Any]] = canvas_data.json_data.get("objects", [])

    for stroke in objects:
        if stroke.get("type") == "path":
            x_coords: List[float] = []
            y_coords: List[float] = []
            points: List[List[float]] = stroke.get("path", [])
            for point in points[::sample_step]:
                if len(point) >= 3:
                    x_coords.append(point[1])
                    y_coords.append(point[2])
            strokes_json.append([x_coords, y_coords])

    return strokes_json
