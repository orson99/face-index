"""
Wrapper sobre insightface: dado el path de una imagen, devuelve una lista
de rostros detectados, cada uno con su bounding box, score y embedding.
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import INSIGHTFACE_MODEL, DET_SIZE, MIN_DET_SCORE

_app = None


def get_app() -> FaceAnalysis:
    """
    Objetivo:
        Cargar el modelo de insightface una sola vez (patrón singleton) y
        reutilizarlo en cada llamada, para no pagar el costo de carga en
        cada foto procesada.

    Argumentos:
        (ninguno)

    Devuelve:
        FaceAnalysis: instancia ya preparada del modelo de detección/embedding.
    """
    global _app
    if _app is None:
        _app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=DET_SIZE)
    return _app


def detect_faces(image_path: str):
    """
    Objetivo:
        Detectar todos los rostros humanos presentes en una imagen y, por
        cada uno, calcular su ubicación (bounding box) y su embedding
        (vector numérico que representa esa cara para poder reconocerla
        más adelante). No modifica ni guarda la imagen en ningún lado.

    Argumentos:
        image_path (str): ruta al archivo de imagen a analizar.

    Devuelve:
        list[dict]: un dict por rostro encontrado, con las claves:
            - "bbox" (tuple): (x, y, ancho, alto) del rostro dentro de la imagen.
            - "det_score" (float): confianza de la detección (0 a 1).
            - "embedding" (np.ndarray): vector de 512 valores que representa el rostro.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    app = get_app()
    faces = app.get(img)

    results = []
    h_img, w_img = img.shape[:2]
    for face in faces:
        if face.det_score < MIN_DET_SCORE:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        # Clampear al tamaño real de la imagen
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)

        results.append({
            "bbox": bbox,
            "det_score": float(face.det_score),
            "embedding": np.asarray(face.normed_embedding, dtype=np.float32),
        })
    return results
