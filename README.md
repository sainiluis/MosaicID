# MosaicID

Aplicación de escritorio para Windows que reconoce rostros en vivo y encuentra a una persona en una
biblioteca de fotografías. Trabajo Práctico Integrador de Visión Artificial.

**Todo ocurre en tu computadora.** No hay servidor, cuenta, nube, telemetría ni APIs externas. Las caras,
los embeddings y las fotos nunca salen del equipo.

## Qué hace

- **Reconocimiento en vivo** con la webcam (también USB o DroidCam), varias caras a la vez.
- **Dos motores** sobre el mismo embedding facial:
  - **SVM**: clasificador supervisado; hay que reentrenarlo al agregar personas.
  - **Distancia**: similitud coseno contra referencias guardadas; una persona nueva se reconoce al instante.
- **Comparar motores**: la misma cara y el mismo embedding, con las dos decisiones lado a lado.
- **Galería inteligente**: indexa una carpeta de fotos y busca todas las fotos donde aparece alguien.

Pipeline: `cámara/imagen → YuNet (caja + 5 landmarks) → SFace.alignCrop() → SFace.feature() (128 valores) → SVM o Distancia`.

## Requisitos

- Windows 10/11, Python 3.10+ (probado con 3.12)
- Dependencias: ver `requirements.txt` (OpenCV 5.1, PySide6 6.9, NumPy, scikit-learn, joblib, Pillow)
- Modelos ONNX oficiales de OpenCV Zoo en `models/`:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`

## Instalación

```bat
cd desktop
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Los modelos ya vienen en `models/`. Si faltaran, se descargan **una sola vez** de
`https://github.com/opencv/opencv_zoo` (carpetas `models/face_detection_yunet` y `models/face_recognition_sface`).
Después la app funciona sin conexión.

Verificar el entorno:

```bat
.venv\Scripts\python diagnostics.py
```

## Ejecutar

```bat
.venv\Scripts\python main.py
```

## Uso

1. **Cámara por IP (por defecto)**: la app se conecta directo a DroidCam en el celular
   (`http://192.168.0.244:4747/video`), sin el cliente de DroidCam en la PC. Abrí DroidCam en el celular y
   listo. La IP y el puerto se cambian con el botón ⚙ junto al selector o en Configuración → Cámara. DroidCam
   admite **una conexión a la vez**: si el cliente de la PC (u otra herramienta) está conectado, MosaicID no
   puede conectarse, y viceversa. MosaicID suelta la conexión al salir de las vistas con cámara.
   **Elegir otra cámara**: debajo del video (Inicio, Reconocimiento, Registro y Buscar en galería) hay un
   selector con las cámaras por nombre; el cambio se aplica al instante y queda guardado. El botón ↻ vuelve a
   abrir la cámara y actualiza la lista (útil después de conectar el celular a DroidCam o Camo). La lista
   también se actualiza sola al conectar o desconectar una cámara. Si la imagen llega negra, la app lo avisa.
2. **Registrar una persona**: Personas → *Registrar persona*. Son 5 pasos:
   - Identidad: escribir el nombre.
   - Preparación: la app verifica que haya un solo rostro, nítido y bien iluminado.
   - Captura: 8 muestras guiadas (frente, giros leves, expresión, cerca, lejos, frontal, libre). Cada una usa
     el rostro completo y solo se guarda si pasa el control de calidad.
   - Revisión: se puede repetir cualquier muestra.
   - Finalizar: Distancia queda disponible enseguida; el SVM queda pendiente.
3. **Entrenar el SVM**: Personas → *Entrenar clasificador* (requiere al menos 2 personas con 3 muestras cada una).
4. **Reconocer**: Reconocimiento → elegir SVM, Distancia o Comparar.
5. **Indexar fotos**: Galería → *Agregar carpeta* → elegir el tipo de fotos de esa carpeta (real, simulada, generada,
   prueba). Las subcarpetas llamadas `real`, `synthetic`, `generated` o `test` toman ese tipo automáticamente. La
   indexación es incremental: solo procesa archivos nuevos o modificados.
6. **Buscar**: Galería → *Buscar persona* → elegir una persona registrada o escanear un rostro con la cámara.
   El filtro **Solo reales** permite evaluar únicamente sobre fotografías reales.

## Dónde se guarda todo (junto a la app)

| Carpeta | Contenido |
|---|---|
| `data/mosaicid.db` | Base SQLite: personas, muestras, embeddings, fotos, caras indexadas |
| `data/samples/` | Recortes alineados (112×112) de cada muestra |
| `data/svm/` | Modelo SVM (`joblib`) + metadata JSON |
| `config/settings.json` | Todos los umbrales y parámetros |
| `cache/thumbnails/` | Miniaturas de la galería |
| `logs/mosaicid.log` | Registro de eventos (sin datos por frame) |

Las fotos originales nunca se modifican.

## Umbrales (calibrados con fotos reales del grupo)

- Distancia y búsqueda en galería: similitud coseno ≥ **0.363**.
- SVM: margen de rechazo **0.40 sobre el azar**. Con K personas entrenadas el umbral efectivo es
  `1/K + 0.40·(1 − 1/K)` (2 personas → 0.70, 6 → 0.50). Con pocas personas el SVM distingue peor a un
  desconocido; es una limitación propia de un clasificador cerrado y se ve bien en *Comparar motores*.

Detalle y metodología: `calibration/CALIBRACION.md`.

## Tests

```bat
.venv\Scripts\python -m pytest -q tests
```

Cubren serialización de embeddings, matching por distancia, umbrales, base de datos, estado DIRTY del SVM,
construcción del dataset, estado del índice de imágenes, configuración, tracker y una integración con los
modelos reales sobre fotos simuladas.

Pruebas con fotos reales (opcionales, sin tocar tus datos):

```bat
.venv\Scripts\python tools\e2e_real.py "<carpeta de fotos>"
.venv\Scripts\python tools\gui_smoke.py "<carpeta para capturas>"
```

## Empaquetar (.exe)

```bat
build_exe.bat
```

Genera `dist\MosaicID\MosaicID.exe` (carpeta portable). Los modelos y assets van dentro del paquete. `data/`,
`config/`, `cache/` y `logs/` se crean junto al `.exe` la primera vez que se ejecuta.

## Desarrollo sin webcam

La variable `MOSAICID_FAKE_CAMERA=<imagen o carpeta>` reemplaza la cámara por imágenes fijas que se repiten en
bucle. Es solo para pruebas.

Para retomar el desarrollo, el estado técnico completo está en `docs/PROJECT_STATE.md`.
