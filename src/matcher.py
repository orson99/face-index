"""
Lógica de reconocimiento: dado el embedding de un rostro nuevo, decide si
corresponde a una persona ya vista (y actualiza su centroide) o si es una
persona nueva.
"""
import numpy as np

from config import MATCH_THRESHOLD
import db


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Objetivo:
        Calcular qué tan parecidos son dos embeddings faciales, usando
        similitud coseno. Cuanto más cerca de 1.0, más probable que sean
        la misma persona; cerca de 0 (o negativo), más distintos.

    Argumentos:
        a (np.ndarray): primer embedding a comparar.
        b (np.ndarray): segundo embedding a comparar.

    Devuelve:
        float: similitud coseno entre -1.0 y 1.0.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def match_or_create_person(conn, embedding: np.ndarray) -> int:
    """
    Objetivo:
        Decidir a qué persona pertenece un rostro nuevo. Compara su
        embedding contra el centroide de cada persona ya conocida; si la
        mejor similitud supera MATCH_THRESHOLD, lo asigna a esa persona
        (y actualiza su centroide). Si ninguna coincide lo suficiente,
        registra una persona nueva.

    Argumentos:
        conn (sqlite3.Connection): conexión abierta a la base de datos.
        embedding (np.ndarray): embedding del rostro recién detectado.

    Devuelve:
        int: id de la persona (ya existente o recién creada) a la que
        quedó asignado el rostro.
    """
    persons = db.get_all_persons(conn)

    best_id, best_sim = None, -1.0
    for person_id, centroid, _count in persons:
        sim = cosine_similarity(embedding, centroid)
        if sim > best_sim:
            best_id, best_sim = person_id, sim

    if best_id is not None and best_sim >= MATCH_THRESHOLD:
        db.update_person_centroid(conn, best_id, embedding)
        return best_id

    return db.create_person(conn, embedding)
