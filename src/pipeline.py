"""
Procesa una foto: detecta rostros, calcula embeddings, resuelve identidad
y guarda coordenadas + embedding en la base. Nunca escribe imágenes nuevas.
"""
import os

import db
import detector
import matcher


def process_photo(path: str) -> int:
    """
    Objetivo:
        Orquestar el procesamiento completo de una foto: detectar sus
        rostros, para cada uno resolver a qué persona pertenece (o crear
        una nueva), y guardar todo (foto, persona, coordenadas, embedding)
        en la base de datos.

    Argumentos:
        path (str): ruta al archivo de la foto a procesar.

    Devuelve:
        int: cantidad de rostros detectados en la foto.
    """
    faces = detector.detect_faces(path)

    with db.get_conn() as conn:
        photo_id = db.add_photo(conn, path=os.path.abspath(path), filename=os.path.basename(path))
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

    print(f"[{os.path.basename(path)}] {len(faces)} rostro(s) detectado(s)")
    return len(faces)
