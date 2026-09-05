"""
Motor de procesamiento: dado el registro de una foto ya guardada en disco
y dada de alta en la base (status='pending'), la detecta, reconoce y
guarda coordenadas + embedding. Lo usan tanto la interfaz web como el
watcher de carpeta — es el único lugar donde vive esta lógica.
"""
import os
import uuid

import db
import detector
import matcher
import hashing
from config import INPUT_DIR


def build_stored_name(original_filename: str) -> tuple:
    """
    Objetivo:
        Generar el nombre nuevo y único con el que el sistema va a guardar
        una imagen en disco, separando así el nombre "de sistema" del
        nombre original que traía el archivo.

    Argumentos:
        original_filename (str): nombre del archivo tal como llegó (subida o drop).

    Devuelve:
        tuple[str, str]: (nombre_guardado, extensión_original), donde
        nombre_guardado es del tipo "<uuid4>.<extensión>".
    """
    original_ext = os.path.splitext(original_filename)[1].lower().lstrip(".")
    stored_filename = f"{uuid.uuid4().hex}.{original_ext}" if original_ext else uuid.uuid4().hex
    return stored_filename, original_ext


def ingest_bytes(file_bytes: bytes, original_filename: str):
    """
    Objetivo:
        Recibir el contenido de un archivo subido (por ejemplo, desde el
        endpoint de drag&drop), calcular su huella de contenido, y si es
        una imagen nueva, guardarla en `input/` con un nombre único y darla
        de alta en la base con status='pending'. Si el contenido ya existía
        (misma imagen subida antes, aunque con otro nombre), no la vuelve a
        guardar ni a registrar: devuelve la foto existente.

    Argumentos:
        file_bytes (bytes): contenido binario del archivo subido.
        original_filename (str): nombre original del archivo, tal como lo subió el usuario.

    Devuelve:
        tuple[int, bool]: (id de la foto, True si es una foto nueva que hay
        que procesar / False si ya existía y se reutilizó tal cual estaba).
    """
    content_hash = hashing.sha256_of_bytes(file_bytes)

    with db.get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM photos WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if existing:
            return existing["id"], False

    stored_filename, original_ext = build_stored_name(original_filename)
    dest_path = os.path.join(INPUT_DIR, stored_filename)
    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    with db.get_conn() as conn:
        photo_id, is_new = db.register_photo(
            conn,
            path=os.path.abspath(dest_path),
            stored_filename=stored_filename,
            original_filename=original_filename,
            original_ext=original_ext,
            content_hash=content_hash,
        )
    return photo_id, is_new


def register_existing_file(path: str):
    """
    Objetivo:
        Dar de alta en la base un archivo que apareció directamente en
        `input/` sin pasar por la interfaz web (por ejemplo, alguien copió
        una foto a mano en esa carpeta). Cumple el mismo contrato que
        `ingest_bytes`, pero partiendo de un archivo que ya está en disco.

    Argumentos:
        path (str): ruta del archivo ya presente en `input/`.

    Devuelve:
        tuple[int, bool]: (id de la foto, True si se registró como nueva /
        False si ya existía un registro para ese contenido).
    """
    original_filename = os.path.basename(path)
    content_hash = hashing.sha256_of_file(path)
    stored_filename, original_ext = build_stored_name(original_filename)

    with db.get_conn() as conn:
        photo_id, is_new = db.register_photo(
            conn,
            path=os.path.abspath(path),
            stored_filename=os.path.basename(path),
            original_filename=original_filename,
            original_ext=original_ext,
            content_hash=content_hash,
        )
    return photo_id, is_new


def process_photo_row(photo_id: int, path: str):
    """
    Objetivo:
        Ejecutar el procesamiento real de UNA foto ya reclamada (detectar
        rostros, resolver identidad de cada uno, guardar coordenadas y
        embeddings) y dejar registrado si terminó bien o con error. No
        vuelve a tocar el estado 'pending' ni decide qué foto procesar
        (eso lo hace `db.claim_next_pending`).

    Argumentos:
        photo_id (int): id de la foto (ya en estado 'processing').
        path (str): ruta en disco de la foto a procesar.

    Devuelve:
        int: cantidad de rostros detectados (0 si no encontró ninguno).
    """
    try:
        faces = detector.detect_faces(path)
        with db.get_conn() as conn:
            for face in faces:
                person_id = matcher.match_or_create_person(conn, face["embedding"])
                db.add_face(
                    conn,
                    photo_id=photo_id,
                    bbox=face["bbox"],
                    det_score=face["det_score"],
                    embedding=face["embedding"],
                    person_id=person_id,
                )
            db.mark_done(conn, photo_id)
        print(f"[foto {photo_id}] {len(faces)} rostro(s) detectado(s)")
        return len(faces)
    except Exception as exc:
        with db.get_conn() as conn:
            db.mark_error(conn, photo_id, str(exc))
        print(f"[foto {photo_id}] ERROR: {exc}")
        return 0


def process_pending_queue():
    """
    Objetivo:
        Vaciar la cola de fotos pendientes: mientras haya alguna con
        status='pending', la reclama de forma atómica y la procesa. La
        usan tanto el watcher (después de detectar un archivo nuevo) como
        la interfaz web (en un hilo de fondo, después de recibir una subida).

    Argumentos:
        (ninguno)

    Devuelve:
        int: cantidad de fotos procesadas en esta pasada.
    """
    processed = 0
    while True:
        with db.get_conn() as conn:
            claimed = db.claim_next_pending(conn)
        if claimed is None:
            break
        process_photo_row(claimed["id"], claimed["path"])
        processed += 1
    return processed
