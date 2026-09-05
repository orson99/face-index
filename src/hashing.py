"""
Utilidad para calcular la huella (hash) del contenido de un archivo, y así
poder detectar cuando dos subidas son en realidad la misma imagen aunque
tengan nombres distintos.
"""
import hashlib


def sha256_of_bytes(data: bytes) -> str:
    """
    Objetivo:
        Calcular el hash SHA-256 de un contenido binario ya cargado en
        memoria (por ejemplo, un archivo recién subido por la interfaz web,
        antes de guardarlo en disco).

    Argumentos:
        data (bytes): contenido binario del archivo.

    Devuelve:
        str: hash SHA-256 en hexadecimal.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Objetivo:
        Calcular el hash SHA-256 de un archivo ya guardado en disco,
        leyéndolo en bloques para no cargarlo entero en memoria (útil para
        el watcher, que recibe archivos que ya están en la carpeta input/).

    Argumentos:
        path (str): ruta del archivo a leer.
        chunk_size (int): tamaño de cada bloque de lectura, en bytes.

    Devuelve:
        str: hash SHA-256 en hexadecimal.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
