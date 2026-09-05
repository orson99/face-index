"""
Utilidad simple para inspeccionar lo que hay guardado, sin abrir ninguna
imagen. Correr con: python query_db.py
"""
import db


def main():
    """
    Objetivo:
        Imprimir por consola un resumen de lo guardado en la base: fotos
        procesadas, personas reconocidas, y el detalle de cada rostro
        (a qué foto y persona pertenece, sus coordenadas y su score).
        Es solo de lectura, para verificar el pipeline sin ver imágenes.

    Argumentos:
        (ninguno)

    Devuelve:
        None
    """
    with db.get_conn() as conn:
        print("== Fotos procesadas ==")
        for row in conn.execute("SELECT id, filename, processed_at FROM photos ORDER BY id"):
            print(row)

        print("\n== Personas detectadas ==")
        for row in conn.execute("SELECT id, faces_count, created_at FROM persons ORDER BY id"):
            print(row)

        print("\n== Rostros (foto, persona, bbox, score) ==")
        q = """
        SELECT f.id, p.filename, f.person_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.det_score
        FROM faces f JOIN photos p ON f.photo_id = p.id
        ORDER BY f.id
        """
        for row in conn.execute(q):
            print(row)


if __name__ == "__main__":
    main()
