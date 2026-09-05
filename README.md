# face-index

Sistema que vigila una carpeta, detecta rostros humanos en las fotos que
aparecen ahí, reconoce si el mismo rostro ya apareció en otra foto, y
guarda únicamente coordenadas + un vector numérico (embedding) por rostro
en una base SQLite. **Nunca guarda la imagen ni el recorte de la cara.**

## Estructura

```
face-index/
  input/            # <- subí las fotos acá, el watcher las procesa solo
  data/
    face_index.db   # se crea solo al primer uso
  src/
    config.py        # rutas y parámetros (umbral de reconocimiento, etc.)
    db.py             # esquema y acceso a SQLite
    detector.py       # detección de rostros + embedding (insightface)
    matcher.py        # decide si un rostro es una persona ya vista o nueva
    pipeline.py        # orquesta: detectar -> reconocer -> guardar
    watcher.py          # proceso en segundo plano que vigila input/
    query_db.py          # inspeccionar lo guardado por consola
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

## Uso

```powershell
cd src
python watcher.py
```

Dejalo corriendo. Cualquier foto que copies o subas a `input/` se procesa
sola: detecta las caras, calcula su posición (bounding box) y decide si
es una persona ya vista antes (comparando embeddings) o una nueva.

Para revisar qué se guardó:

```powershell
python query_db.py
```

## Qué se guarda en la base (`data/face_index.db`)

- `photos`: referencia a la foto original (path/nombre), no la imagen.
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

Ya existía una carpeta `recfac/` con una app Flask que detectaba rostros
con Haar cascades y guardaba miniaturas como archivos JPG. Este proyecto
es una versión distinta y separada: usa un detector más preciso con
reconocimiento real (embeddings), corre como watcher en vez de app web, y
no persiste ninguna imagen — solo números en SQLite.
