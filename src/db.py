"""
Capa de acceso a la base de datos SQLite.
Guarda SOLO coordenadas y embeddings (vectores numéricos) de cada rostro,
nunca la imagen ni el recorte de la cara. También lleva el registro de
qué fotos ya se subieron/procesaron, para no repetir trabajo.
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
    stored_filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    original_ext TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TEXT NOT NULL,
    processed_at TEXT
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
CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
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
    conn.row_factory = sqlite3.Row
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


def register_photo(conn, path: str, stored_filename: str, original_filename: str,
                    original_ext: str, content_hash: str):
    """
    Objetivo:
        Dar de alta una foto nueva en estado 'pending', ANTES de procesarla.
        Es el punto de entrada común tanto para la interfaz web (drag&drop)
        como para el watcher de carpeta. Si la foto ya existía (mismo path
        o mismo contenido/hash — por ejemplo la misma imagen subida dos
        veces con nombres distintos), no crea un duplicado: devuelve la
        fila ya existente.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        path (str): ruta absoluta donde quedó guardado el archivo (con el nombre nuevo).
        stored_filename (str): nombre único que el sistema le asignó al archivo (uuid + extensión).
        original_filename (str): nombre del archivo tal como lo subió/tenía el usuario.
        original_ext (str): extensión original del archivo (informativa).
        content_hash (str): hash SHA-256 del contenido del archivo, para detectar duplicados.

    Devuelve:
        tuple[int, bool]: (id de la foto, True si se creó una fila nueva /
        False si ya existía una foto igual y se reutilizó).
    """
    existing = conn.execute(
        "SELECT id FROM photos WHERE path = ? OR content_hash = ?",
        (path, content_hash),
    ).fetchone()
    if existing:
        return existing["id"], False

    cur = conn.execute(
        """INSERT INTO photos (path, stored_filename, original_filename, original_ext,
                                content_hash, status, created_at)
           VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
        (path, stored_filename, original_filename, original_ext, content_hash,
         datetime.utcnow().isoformat()),
    )
    return cur.lastrowid, True


def get_photo(conn, photo_id: int):
    """
    Objetivo:
        Consultar el estado actual de una foto (para que la interfaz web
        pueda mostrar en vivo si ya terminó de procesarse).

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        photo_id (int): id de la foto a consultar.

    Devuelve:
        sqlite3.Row | None: fila con todos los campos de `photos`, o None si no existe.
    """
    return conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()


def claim_next_pending(conn):
    """
    Objetivo:
        Tomar la foto pendiente más antigua y marcarla como 'processing' de
        forma atómica (con un UPDATE condicionado al estado actual), para
        que si en algún momento hay más de un proceso mirando la cola, dos
        workers nunca terminen procesando la misma foto dos veces.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.

    Devuelve:
        sqlite3.Row | None: la fila de la foto ya marcada como 'processing',
        o None si no hay ninguna foto pendiente.
    """
    row = conn.execute(
        "SELECT id, path FROM photos WHERE status = 'pending' ORDER BY id LIMIT 1"
    ).fetchone()
    if row is None:
        return None

    cur = conn.execute(
        "UPDATE photos SET status = 'processing' WHERE id = ? AND status = 'pending'",
        (row["id"],),
    )
    if cur.rowcount == 0:
        # Otro proceso se la llevó justo antes: no es nuestra, no la tocamos.
        return None
    return row


def mark_done(conn, photo_id: int):
    """
    Objetivo:
        Marcar una foto como procesada con éxito.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        photo_id (int): id de la foto a actualizar.

    Devuelve:
        None
    """
    conn.execute(
        "UPDATE photos SET status = 'done', processed_at = ? WHERE id = ?",
        (datetime.utcnow().isoformat(), photo_id),
    )


def mark_error(conn, photo_id: int, message: str):
    """
    Objetivo:
        Marcar una foto como fallida y guardar el motivo, para poder
        diagnosticarla después sin volver a intentarla en loop infinito.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        photo_id (int): id de la foto a actualizar.
        message (str): descripción breve del error ocurrido.

    Devuelve:
        None
    """
    conn.execute(
        "UPDATE photos SET status = 'error', error_message = ?, processed_at = ? WHERE id = ?",
        (str(message), datetime.utcnow().isoformat(), photo_id),
    )


def get_faces_for_photo(conn, photo_id: int):
    """
    Objetivo:
        Traer los rostros ya guardados de una foto (sus coordenadas y a
        qué persona quedaron asignados), para mostrarlos en la interfaz.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        photo_id (int): id de la foto.

    Devuelve:
        list[sqlite3.Row]: una fila por rostro, con bbox, det_score y person_id.
    """
    return conn.execute(
        """SELECT id, person_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score
           FROM faces WHERE photo_id = ? ORDER BY id""",
        (photo_id,),
    ).fetchall()


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
    return [(row["id"], _blob_to_vec(row["centroid"]), row["faces_count"]) for row in rows]


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
    centroid, count = _blob_to_vec(row["centroid"]), row["faces_count"]
    new_count = count + 1
    new_centroid = (centroid * count + new_embedding) / new_count
    conn.execute(
        "UPDATE persons SET centroid = ?, faces_count = ? WHERE id = ?",
        (_vec_to_blob(new_centroid), new_count, person_id),
    )


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
