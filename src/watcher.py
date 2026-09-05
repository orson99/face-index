"""
Vigila la carpeta `input/` y procesa automáticamente cada foto nueva que
aparezca ahí puesta a mano (fuera de la interfaz web). Correr con:

    python watcher.py

Es un modo de ingesta alternativo al drag&drop: comparte exactamente el
mismo motor de procesamiento (pipeline.py) y la misma base de datos, así
que puede convivir con la interfaz web sin pisarse.
"""
import os
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import db
from config import INPUT_DIR, VALID_EXTS
from pipeline import register_existing_file, process_pending_queue


def is_image(path: str) -> bool:
    """
    Objetivo:
        Determinar si un archivo tiene una extensión de imagen soportada,
        para decidir si el watcher debe procesarlo.

    Argumentos:
        path (str): ruta o nombre de archivo a evaluar.

    Devuelve:
        bool: True si la extensión está en VALID_EXTS, False en caso contrario.
    """
    return path.lower().endswith(VALID_EXTS)


class NewPhotoHandler(FileSystemEventHandler):
    """
    Objetivo:
        Reaccionar a los eventos del sistema de archivos dentro de
        `input/` (archivo creado o movido/renombrado) y disparar el
        registro + procesamiento de la foto correspondiente.
    """

    def on_created(self, event):
        """
        Objetivo:
            Manejar el evento "se creó un archivo nuevo" en la carpeta
            vigilada. Si es una imagen, espera a que termine de copiarse
            y la registra/procesa.

        Argumentos:
            event (watchdog.events.FileSystemEvent): evento emitido por
                watchdog, con `src_path` (ruta del archivo) e `is_directory`.

        Devuelve:
            None
        """
        if event.is_directory or not is_image(event.src_path):
            return
        self._wait_until_stable(event.src_path)
        self._register_and_process(event.src_path)

    def on_moved(self, event):
        """
        Objetivo:
            Manejar el evento "se movió o renombró un archivo" hacia la
            carpeta vigilada (por ejemplo, al copiar y luego renombrar).

        Argumentos:
            event (watchdog.events.FileSystemEvent): evento emitido por
                watchdog, con `dest_path` (ruta final del archivo) e
                `is_directory`.

        Devuelve:
            None
        """
        if event.is_directory or not is_image(event.dest_path):
            return
        self._register_and_process(event.dest_path)

    def _wait_until_stable(self, path, checks=3, interval=0.5):
        """
        Objetivo:
            Esperar a que un archivo termine de copiarse/subirse antes de
            procesarlo, verificando que su tamaño deje de cambiar durante
            varias mediciones seguidas. Evita leer una foto a medio copiar.

        Argumentos:
            path (str): ruta del archivo a monitorear.
            checks (int): cantidad de mediciones consecutivas con el mismo
                tamaño necesarias para considerarlo "estable".
            interval (float): segundos de espera entre cada medición.

        Devuelve:
            None
        """
        last_size = -1
        stable_checks = 0
        while stable_checks < checks:
            try:
                size = os.path.getsize(path)
            except OSError:
                time.sleep(interval)
                continue
            if size == last_size:
                stable_checks += 1
            else:
                stable_checks = 0
                last_size = size
            time.sleep(interval)

    def _register_and_process(self, path):
        """
        Objetivo:
            Dar de alta el archivo (si su contenido no estaba ya registrado)
            y vaciar la cola de pendientes, registrando en consola cualquier
            error sin interrumpir al watcher.

        Argumentos:
            path (str): ruta de la foto a registrar y procesar.

        Devuelve:
            None
        """
        try:
            register_existing_file(path)
            process_pending_queue()
        except Exception as exc:
            print(f"[ERROR] No se pudo procesar {path}: {exc}")


def backfill_existing():
    """
    Objetivo:
        Registrar y procesar las fotos que ya estaban en `input/` antes de
        arrancar el watcher (por ejemplo, si se copiaron varias a mano
        mientras el watcher estaba apagado).

    Argumentos:
        (ninguno)

    Devuelve:
        None
    """
    for name in os.listdir(INPUT_DIR):
        path = os.path.join(INPUT_DIR, name)
        if os.path.isfile(path) and is_image(path):
            try:
                register_existing_file(path)
            except Exception as exc:
                print(f"[ERROR] No se pudo registrar {path}: {exc}")
    process_pending_queue()


def main():
    """
    Objetivo:
        Punto de entrada del watcher: inicializa la base de datos, registra
        y procesa lo que ya hubiera en `input/`, y después queda vigilando
        la carpeta en segundo plano hasta que se lo interrumpa (Ctrl+C).

    Argumentos:
        (ninguno)

    Devuelve:
        None
    """
    os.makedirs(INPUT_DIR, exist_ok=True)
    db.init_db()

    print(f"Procesando fotos existentes en {INPUT_DIR} ...")
    backfill_existing()

    print(f"Vigilando {INPUT_DIR} (Ctrl+C para salir)...")
    handler = NewPhotoHandler()
    observer = Observer()
    observer.schedule(handler, INPUT_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
