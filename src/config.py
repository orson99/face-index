"""Configuración central del pipeline de reconocimiento facial."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "input")
DB_PATH = os.path.join(BASE_DIR, "data", "face_index.db")

# Extensiones de imagen que el sistema acepta (drag&drop y watcher de carpeta)
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Modelo de insightface (detección + embedding en un solo paso)
# "buffalo_l" es el más preciso; "buffalo_s" es más liviano/rápido.
INSIGHTFACE_MODEL = "buffalo_l"
DET_SIZE = (640, 640)

# Umbral de similitud coseno para considerar que dos rostros son la misma persona.
# 0.0 a 1.0. Empíricamente, ~0.45-0.55 funciona bien con embeddings de insightface (ArcFace).
MATCH_THRESHOLD = 0.50

# Confianza mínima de detección para guardar un rostro (insightface usa 0-1)
MIN_DET_SCORE = 0.5

# Tamaño máximo aceptado por subida, en bytes (16 MB por archivo)
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

# Puerto donde corre la interfaz web de drag&drop
WEB_PORT = 5000
