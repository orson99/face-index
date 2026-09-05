# face-index

Sistema que detecta rostros humanos en fotos, reconoce si el mismo rostro
ya apareció antes, y guarda únicamente coordenadas + un vector numérico
(embedding) por rostro en una base SQLite. **Nunca guarda la imagen ni el
recorte de la cara.**

Tiene dos formas de ingresar fotos, que comparten el mismo motor de
procesamiento y la misma base de datos:

- **Interfaz web (drag&drop)** — `app.py`: arrastrás una o varias fotos
  al navegador.
- **Watcher de carpeta** — `watcher.py`: copiás fotos a mano en `input/`.

Podés usar una, la otra, o las dos al mismo tiempo.

## Estructura

```
face-index/
  input/              # fotos guardadas con su nombre único de sistema
  data/
    face_index.db      # se crea solo al primer uso
  src/
    config.py           # rutas y parámetros (umbral de reconocimiento, etc.)
    db.py                # esquema y acceso a SQLite
    hashing.py            # huella (SHA-256) del contenido de cada foto
    detector.py            # detección de rostros + embedding (insightface)
    matcher.py              # decide si un rostro es una persona ya vista o nueva
    pipeline.py               # ingesta + motor de procesamiento (usado por app.py y watcher.py)
    app.py                     # interfaz web con drag&drop
    templates/index.html        # página de la interfaz web
    watcher.py                   # ingesta alternativa: vigila input/
    query_db.py                   # inspeccionar lo guardado por consola
  requirements.txt
```

## Instalación (Windows, PowerShell)

```powershell
cd C:\Users\Win11\Desktop\DEV_0\face-index
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

La primera vez que corra, `insightface` descarga el modelo `buffalo_l`
(~350 MB) a `%USERPROFILE%\.insightface\models`. Necesita internet esa
primera vez; después funciona sin conexión.

## Uso — interfaz web (drag&drop)

```powershell
cd src
python app.py
```

Abrí `http://localhost:5000` y arrastrá las fotos. Cada una se muestra en
pantalla y se va actualizando sola (pendiente → procesando → resultado)
a medida que el sistema la analiza.

## Uso — watcher de carpeta (alternativa)

```powershell
cd src
python watcher.py
```

Dejalo corriendo. Cualquier foto que copies a mano en `input/` se procesa
sola.

## Consultar lo guardado

```powershell
python query_db.py
```

## Cómo se maneja cada foto subida

1. **Nombre de sistema, separado del original.** Al recibir una foto se le
   asigna un nombre único (`<uuid>.ext`) con el que se guarda en `input/`.
   El nombre y la extensión originales quedan guardados aparte, en la
   base (`original_filename`, `original_ext`), solo como referencia.
2. **Detección de duplicados por contenido, no por nombre.** Antes de
   guardar el archivo se calcula un hash SHA-256 de su contenido. Si esa
   misma imagen ya se había subido antes (aunque con otro nombre), no se
   vuelve a guardar ni a procesar.
3. **Estado por foto.** Cada foto queda con `status`: `pending` (recién
   ingresada), `processing` (un worker la está analizando), `done`
   (terminada, con sus rostros ya guardados) o `error` (con el motivo en
   `error_message`). Un solo hilo de fondo vacía la cola de pendientes,
   así el modelo de reconocimiento nunca corre desde dos lugares a la vez.

## Qué se guarda en la base (`data/face_index.db`)

- `photos`: `path`, `stored_filename`, `original_filename`, `original_ext`,
  `content_hash`, `status`, `error_message` y fechas — nunca la imagen.
- `persons`: una fila por persona distinta reconocida, con su embedding
  promedio (centroide).
- `faces`: una fila por rostro detectado, con `bbox_x, bbox_y, bbox_w,
  bbox_h` (su ubicación dentro de la foto), el score de confianza, a qué
  `person_id` quedó asignado, y su embedding (vector de 512 números).

## Ajustar la sensibilidad del reconocimiento

En `src/config.py`, `MATCH_THRESHOLD` (0.50 por defecto) controla qué tan
parecidos tienen que ser dos rostros para considerarlos la misma persona.
Subirlo (ej. 0.6) es más estricto (menos falsos positivos, puede duplicar
personas); bajarlo (ej. 0.4) es más laxo (puede juntar personas distintas).

## Nota sobre el intento anterior

Ya existía una carpeta `recfac/` en la misma `DEV_0` con una app Flask que
detectaba rostros con Haar cascades y guardaba miniaturas como archivos
JPG. Este proyecto es una versión distinta y separada: usa un detector más
preciso con reconocimiento real (embeddings), separa nombre original de
nombre de sistema, evita duplicados por contenido, y no persiste ninguna
imagen — solo números en SQLite.
