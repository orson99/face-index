"""
Interfaz web: permite arrastrar y soltar varias imágenes para que el
sistema las registre y las procese (detección + reconocimiento de
rostros). Correr con:

    python app.py

Un solo hilo de fondo va vaciando la cola de fotos pendientes, para no
tener más de un proceso de detección corriendo a la vez (el modelo de
reconocimiento no es seguro para usarlo desde varios hilos al mismo tiempo).
"""
import os
import threading
import time

from flask import Flask, request, jsonify, render_template

import db
import pipeline
from config import VALID_EXTS, MAX_UPLOAD_BYTES, WEB_PORT, INPUT_DIR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * 50  # margen para varias fotos por request


def _background_worker():
    """
    Objetivo:
        Correr en un hilo aparte, para siempre, vaciando la cola de fotos
        pendientes apenas aparecen. Es el único hilo que ejecuta el
        detector, para no correr el modelo de reconocimiento facial desde
        varios hilos a la vez.

    Argumentos:
        (ninguno)

    Devuelve:
        None (nunca termina; es un hilo daemon)
    """
    while True:
        processed = pipeline.process_pending_queue()
        if processed == 0:
            time.sleep(0.5)


def _is_allowed(filename: str) -> bool:
    """
    Objetivo:
        Verificar que el nombre de archivo subido tenga una extensión de
        imagen soportada, antes de aceptarlo.

    Argumentos:
        filename (str): nombre del archivo tal como llegó en la subida.

    Devuelve:
        bool: True si la extensión está permitida.
    """
    return filename.lower().endswith(VALID_EXTS)


@app.route("/")
def index():
    """
    Objetivo:
        Servir la página principal con el área de arrastrar y soltar.

    Argumentos:
        (ninguno)

    Devuelve:
        str: HTML renderizado de templates/index.html.
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Objetivo:
        Recibir una o varias imágenes soltadas en la interfaz, registrar
        cada una en la base (con nombre único de sistema, separado del
        nombre original) y devolver sus ids para que el navegador pueda
        empezar a consultar el estado de cada una. NO procesa acá mismo:
        solo hace la ingesta rápida; el hilo de fondo se encarga de
        detectarlas después.

    Argumentos:
        (ninguno explícito; toma los archivos de request.files, bajo la
        clave "files", uno o varios)

    Devuelve:
        flask.Response: JSON con la lista de fotos recibidas, cada una
        como {id, original_filename, is_new} o {original_filename, error}.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    results = []
    for file in files:
        filename = file.filename or ""
        if not _is_allowed(filename):
            results.append({"original_filename": filename, "error": "Extensión no soportada"})
            continue

        file_bytes = file.read()
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            results.append({"original_filename": filename, "error": "El archivo supera el tamaño máximo permitido"})
            continue

        photo_id, is_new = pipeline.ingest_bytes(file_bytes, filename)
        results.append({"id": photo_id, "original_filename": filename, "is_new": is_new})

    return jsonify({"photos": results})


@app.route("/api/photos/<int:photo_id>")
def photo_status(photo_id):
    """
    Objetivo:
        Informar el estado actual de una foto (pending/processing/done/error)
        y, si ya terminó, cuántos rostros encontró y a qué personas quedaron
        asignados. Lo consulta el navegador en vivo (polling) después de subir.

    Argumentos:
        photo_id (int): id de la foto, tomado de la URL.

    Devuelve:
        flask.Response: JSON con el estado de la foto y, si corresponde,
        la lista de rostros encontrados (coordenadas, score, persona).
    """
    with db.get_conn() as conn:
        photo = db.get_photo(conn, photo_id)
        if photo is None:
            return jsonify({"error": "Foto no encontrada"}), 404

        faces = []
        if photo["status"] == "done":
            faces = [
                {
                    "person_id": f["person_id"],
                    "bbox": [f["bbox_x"], f["bbox_y"], f["bbox_w"], f["bbox_h"]],
                    "det_score": f["det_score"],
                }
                for f in db.get_faces_for_photo(conn, photo_id)
            ]

    return jsonify({
        "id": photo["id"],
        "original_filename": photo["original_filename"],
        "status": photo["status"],
        "error_message": photo["error_message"],
        "faces_count": len(faces),
        "faces": faces,
    })


def main():
    """
    Objetivo:
        Punto de entrada de la interfaz web: prepara la carpeta de entrada
        y la base de datos, arranca el hilo de fondo que procesa la cola,
        y levanta el servidor Flask.

    Argumentos:
        (ninguno)

    Devuelve:
        None
    """
    os.makedirs(INPUT_DIR, exist_ok=True)
    db.init_db()

    worker = threading.Thread(target=_background_worker, daemon=True)
    worker.start()

    app.run(host="0.0.0.0", port=WEB_PORT, debug=False)


if __name__ == "__main__":
    main()
