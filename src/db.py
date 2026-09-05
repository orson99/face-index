"""
Capa de acceso a la base de datos SQLite.
Guarda SOLO coordenadas y embeddings (vectores numéricos) de cada rostro,
nunca la imagen ni el recorte de la cara.
"""
import sqlite3
import numpy as np
from datetime import datetime
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    -- embedding promedio (centroide) de todas las caras asignadas a esta persona
    centroid BLOB NOT NULL,
    faces_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id),
    person_id INTEGER REFERENCES persons(id),
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL,
    bbox_h INTEGER NOT NULL,
    det_score REAL NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
"""


@contextmanager
def get_conn():
    """
    Objetivo:
        Abrir una conexión a la base SQLite y garantizar que se confirme
        (commit) y se cierre correctamente, incluso si ocurre un error.
        Se usa como context manager: `with get_conn() as conn: ...`.

    Argumentos:
        (ninguno)

    Devuelve:
        sqlite3.Connection: conexión abierta a la base de datos.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """
    Objetivo:
        Crear las tablas (photos, persons, faces) si todavía no existen.
        Es seguro llamarla varias veces (usa CREATE TABLE IF NOT EXISTS).

    Argumentos:
        (ninguno)

    Devuelve:
        None
    """
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _vec_to_blob(vec: np.ndarray) -> bytes:
    """
    Objetivo:
        Convertir un vector (embedding) de numpy a bytes, para poder
        guardarlo en una columna BLOB de SQLite.

    Argumentos:
        vec (np.ndarray): vector numérico a convertir (ej. embedding de 512 valores).

    Devuelve:
        bytes: representación binaria del vector en float32.
    """
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    """
    Objetivo:
        Reconstruir un vector numpy a partir de los bytes guardados en SQLite.
        Es la operación inversa de `_vec_to_blob`.

    Argumentos:
        blob (bytes): datos binarios leídos de la base de datos.

    Devuelve:
        np.ndarray: vector numérico reconstruido (float32).
    """
    return np.frombuffer(blob, dtype=np.float32)


def add_photo(conn, path: str, filename: str) -> int:
    """
    Objetivo:
        Registrar una foto procesada en la tabla `photos`. Si esa misma
        ruta ya estaba guardada, no la duplica: devuelve el id existente.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        path (str): ruta absoluta del archivo de la foto (referencia, no se copia la imagen).
        filename (str): nombre del archivo, para mostrarlo en reportes/consultas.

    Devuelve:
        int: id de la foto en la tabla `photos`.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO photos (path, filename, processed_at) VALUES (?, ?, ?)",
        (path, filename, datetime.utcnow().isoformat()),
    )
    if cur.lastrowid:
        return cur.lastrowid
    row = conn.execute("SELECT id FROM photos WHERE path = ?", (path,)).fetchone()
    return row[0]


def add_face(conn, photo_id: int, bbox, det_score: float, embedding: np.ndarray, person_id: int):
    """
    Objetivo:
        Guardar un rostro detectado: su ubicación dentro de la foto (bbox),
        la confianza de la detección y su embedding. No guarda ninguna imagen.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        photo_id (int): id de la foto (tabla `photos`) a la que pertenece el rostro.
        bbox (tuple): coordenadas (x, y, ancho, alto) del rostro dentro de la foto.
        det_score (float): confianza de la detección, entre 0 y 1.
        embedding (np.ndarray): vector numérico que representa el rostro (para reconocimiento).
        person_id (int): id de la persona (tabla `persons`) a la que quedó asignado este rostro.

    Devuelve:
        None
    """
    x, y, w, h = bbox
    conn.execute(
        """INSERT INTO faces (photo_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                               det_score, embedding, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (photo_id, person_id, int(x), int(y), int(w), int(h), float(det_score),
         _vec_to_blob(embedding), datetime.utcnow().isoformat()),
    )


def get_all_persons(conn):
    """
    Objetivo:
        Traer todas las personas ya registradas, con su embedding promedio
        (centroide), para poder compararlas contra un rostro nuevo.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.

    Devuelve:
        list[tuple]: lista de (person_id, centroide (np.ndarray), cantidad_de_rostros).
    """
    rows = conn.execute("SELECT id, centroid, faces_count FROM persons").fetchall()
    return [(pid, _blob_to_vec(blob), count) for pid, blob, count in rows]


def create_person(conn, embedding: np.ndarray) -> int:
    """
    Objetivo:
        Registrar una persona nueva (nunca vista antes) usando el embedding
        de su primer rostro detectado como centroide inicial.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        embedding (np.ndarray): embedding del rostro que originó esta persona.

    Devuelve:
        int: id de la nueva persona en la tabla `persons`.
    """
    cur = conn.execute(
        "INSERT INTO persons (label, centroid, faces_count, created_at) VALUES (?, ?, ?, ?)",
        (None, _vec_to_blob(embedding), 1, datetime.utcnow().isoformat()),
    )
    return cur.lastrowid


def update_person_centroid(conn, person_id: int, new_embedding: np.ndarray):
    """
    Objetivo:
        Actualizar el centroide (embedding promedio) de una persona cuando
        se le suma un rostro nuevo, usando un promedio incremental.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        person_id (int): id de la persona (tabla `persons`) a actualizar.
        new_embedding (np.ndarray): embedding del nuevo rostro asignado a esta persona.

    Devuelve:
        None
    """
    row = conn.execute(
        "SELECT centroid, faces_count FROM persons WHERE id = ?", (person_id,)
    ).fetchone()
    centroid, count = _blob_to_vec(row[0]), row[1]
    new_count = count + 1
    new_centroid = (centroid * count + new_embedding) / new_count
    conn.execute(
        "UPDATE persons SET centroid = ?, faces_count = ? WHERE id = ?",
        (_vec_to_blob(new_centroid), new_count, person_id),
    )
