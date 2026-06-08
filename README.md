# 🦾 NutriBot — Brazo Robótico Asistente para Alimentación

> Sistema autónomo de asistencia para la alimentación de personas con movilidad reducida.
> Combina visión por computadora, aprendizaje por imitación (Behavior Cloning) y control serial de un brazo robótico de 5 ejes.

---

## Tabla de contenidos

1. [Descripción general](#descripción-general)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Arquitectura del sistema](#arquitectura-del-sistema)
4. [Requisitos](#requisitos)
5. [Instalación](#instalación)
6. [Configuración inicial](#configuración-inicial)
7. [Flujo de uso](#flujo-de-uso)
8. [Descripción de módulos](#descripción-de-módulos)
9. [Entrenamiento del modelo MLP](#entrenamiento-del-modelo-mlp)
10. [Calibración](#calibración)
11. [Recolección de demostraciones](#recolección-de-demostraciones)
12. [Hardware](#hardware)
13. [Solución de problemas](#solución-de-problemas)

---

## Descripción general

NutriBot es un sistema de asistencia robótica para la alimentación autónoma. El brazo puede operar en dos modos:

- **Modo Sólido** — detecta trozos de comida en un plato mediante YOLOv8, predice los movimientos necesarios con una red neuronal entrenada por imitación (Behavior Cloning) y entrega el alimento acercando el tenedor a la boca del usuario, detectada con MediaPipe Face Mesh.
- **Modo Líquido (Sopa)** — toma una cuchara, mide el nivel de sopa con segmentación YOLO + análisis de color HSV, realiza cucharadas repetidas y entrega la sopa adaptándose a la posición de la boca del usuario.

La interfaz gráfica (Tkinter) permite al usuario iniciar modos, pausar, detener y monitorear el estado del sistema en tiempo real, con soporte de comandos de voz.

---

## Estructura del proyecto

```
CODIGOS_FINAL/
├── calibracion_sopa/                   # Calibración del plato de sopa
│   ├── calibracion_plato.pkl           # Datos de referencia HSV del plato
│   ├── calibracion_plato_ref.png       # Imagen de referencia de calibración
│   └── calibracion_plato_stats.json    # Estadísticas del plato (tolerancias)
│
├── auto_brazo_completo_PRESENTACION.py # Módulo principal: robot, MLP, YOLO, MediaPipe
├── auto_brazo_cuchara_definitivo.py    # Módulo autónomo de modo Sopa (standalone)
├── brazo_robotico_ASISTENTE.ino        # Firmware Arduino del brazo
├── calibrar_plato_sopa_avanzado.py     # Herramienta de calibración del plato
├── collect_demos_corners.py            # Recolector de demostraciones (esquinas)
├── collect_demos_cuadricula.py         # Recolector de demostraciones (cuadrícula)
├── interfaz_nutribot.py                # Interfaz gráfica principal (Tkinter)
├── orquestador.py                      # Orquestador unificado de modos y fases
├── train_mlp.py                        # Entrenamiento del MLP de Behavior Cloning
│
├── calibracion.pkl                     # Calibración activa del plato de sopa
├── calibracion_cuadricula.json         # Cuadrícula de calibración (modo sólido)
├── estado.json                         # Bus de estado compartido entre procesos
├── estado.json.lock                    # Archivo de bloqueo para acceso concurrente
├── modelo_bc.pt                        # Pesos del modelo MLP entrenado
├── yolov8n-seg.pt                      # YOLOv8 Nano segmentación (nivel de sopa)
└── yolov8x-worldv2.pt                  # YOLOv8 XL World v2 (detección de comida)
```

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────┐
│                  interfaz_nutribot.py                   │
│  Tkinter GUI · Voz (SpeechRecognition) · TTS (pyttsx3) │
│  CAM1 (YOLO detección) · CAM2 (MediaPipe face)         │
└────────────────────┬────────────────────────────────────┘
                     │ estado.json (bus IPC)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    orquestador.py                       │
│  Máquina de estados · Control de fases · ArUco verify  │
│  Modo SOLIDO · Modo LIQUIDO (Sopa) · Home/Pausa/Stop   │
└────────┬───────────────────────────┬────────────────────┘
         │                           │
         ▼                           ▼
┌────────────────┐         ┌──────────────────────┐
│  MLPPredictor  │         │  _MediaPipeProxy     │
│  modelo_bc.pt  │         │  Worker subprocess   │
│  (cx,cy)→pasos │         │  distancia boca (cm) │
└────────┬───────┘         └──────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              RobotInterface (Serial COM3)               │
│         brazo_robotico_ASISTENTE.ino (Arduino)          │
│     base · hombro · codo · muñeca (gripper) · rotación │
└─────────────────────────────────────────────────────────┘
```

### Bus de estado (`estado.json`)

Todos los módulos se comunican a través de un archivo JSON compartido que actúa como bus de mensajes entre procesos:

| Campo | Descripción |
|---|---|
| `fase_actual` | Fase en curso: `REPOSO`, `DETECCION`, `AGARRE`, `ENTREGA`, etc. |
| `comando` | Comando activo: `INICIAR_CICLO`, `INICIAR_CICLO_LIQUIDO`, `DETENER`, `PAUSA` |
| `objetivo` | Posición normalizada del trozo detectado: `{cx_norm, cy_norm, listo}` |
| `voz` | Texto a sintetizar por TTS |
| `cam1_owner` | Propietario de la cámara 1: `interface` o `cuchara` |
| `cam2_owner` | Propietario de la cámara 2: `interface` u `orquestador` |
| `modo_liquido_activo` | Booleano — indica si el modo sopa está activo |
| `yolo_activo` | Booleano — indica si la detección YOLO está corriendo |

---

## Requisitos

### Software

- Python 3.11+
- Arduino IDE (para flashear el firmware)

### Dependencias Python

```
torch torchvision          # Red neuronal MLP
ultralytics                # YOLOv8 (YOLO World v2 + YOLOv8n-seg)
mediapipe                  # Detección de rostro y distancia
opencv-python              # Captura de cámara, ArUco, HSV
numpy
Pillow                     # Tkinter image display
SpeechRecognition pyaudio  # Reconocimiento de voz (opcional)
pyttsx3                    # Síntesis de voz TTS (opcional)
pyserial                   # Comunicación serial con Arduino
```

Instalación de dependencias:

```bash
pip install torch torchvision ultralytics mediapipe opencv-python numpy Pillow pyserial
pip install SpeechRecognition pyaudio pyttsx3   # Módulos de voz (opcionales)
```

### Hardware

- Brazo robótico de 5 ejes con motores paso a paso + servos
- Arduino (con firmware `brazo_robotico_ASISTENTE.ino`)
- 2 cámaras USB:
  - **CAM1 (índice 2-3)** — vista cenital del plato (YOLO + calibración sopa)
  - **CAM2 (índice 1)** — vista frontal del usuario (MediaPipe Face Mesh)
- Marcador ArUco ID=1 (DICT_4X4_50) fijado a la cuchara

---

## Instalación

```bash
# 1. Clona el repositorio
git clone <url-del-repo>
cd CODIGOS_FINAL

# 2. Instala dependencias
pip install torch torchvision ultralytics mediapipe opencv-python numpy Pillow pyserial

# 3. Descarga los modelos YOLO (si no están incluidos)
#    yolov8n-seg.pt  →  https://github.com/ultralytics/assets/releases
#    yolov8x-worldv2.pt  →  https://github.com/ultralytics/assets/releases

# 4. Flashea el Arduino
#    Abre brazo_robotico_ASISTENTE.ino en Arduino IDE
#    Selecciona la placa y puerto COM correcto → Upload

# 5. Ajusta las rutas en los scripts (ver Configuración inicial)
```

---

## Configuración inicial

Antes de ejecutar, edita las siguientes constantes en cada archivo según tu entorno:

### `interfaz_nutribot.py` y `auto_brazo_completo_PRESENTACION.py`

```python
MODELO_PT   = r"ruta\a\modelo_bc.pt"
CALIB_JSON  = r"ruta\a\calibracion_cuadricula.json"
SERIAL_PORT = "COM3"          # Puerto del Arduino
CAM1_INDEX  = 2               # Índice de la cámara del plato
CAM2_INDEX  = 1               # Índice de la cámara frontal
```

### `orquestador.py`

```python
ARCHIVO_CALIB_SOPA = r"ruta\a\calibracion.pkl"
MODELO_YOLO_SEG    = r"ruta\a\yolov8n-seg.pt"
CAMARA_SOPA_INDEX      = 2
CAMARA_MEDIAPIPE_INDEX = 1
```

---

## Flujo de uso

### Inicio normal (interfaz gráfica)

```bash
python interfaz_nutribot.py
```

La interfaz lanza automáticamente el orquestador como subproceso. Desde la GUI puedes:

1. Ver la imagen de las cámaras en tiempo real
2. Pulsar **"Quiero comer"** (o decirlo por voz) → inicia **Modo Sólido**
3. Pulsar **"Quiero sopa"** → inicia **Modo Líquido**
4. **Pausar** / **Detener** en cualquier momento

### Inicio directo (orquestador)

```bash
# Modo normal con hardware real
python orquestador.py --puerto COM3 --cam1 2 --cam2 1

# Modo simulación (sin Arduino ni cámaras)
python orquestador.py --sim

# Parámetros de distancia personalizados
python orquestador.py --threshold 20 --eating 13 --far 25
```

#### Parámetros del orquestador

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `--puerto` | `COM3` | Puerto serial del Arduino |
| `--cam1` | `3` | Índice cámara del plato |
| `--cam2` | `1` | Índice cámara frontal |
| `--threshold` | `20.0` | Distancia (cm) de alerta de proximidad |
| `--eating` | `15.0` | Distancia (cm) para iniciar espera de ingesta |
| `--far` | `20.0` | Distancia (cm) para confirmar alejamiento |
| `--sim` | `False` | Activa modo simulación sin hardware |
| `--modelo` | `""` | Ruta alternativa al modelo `.pt` |
| `--calib` | `""` | Ruta alternativa al `.json` de cuadrícula |

---

## Descripción de módulos

### `interfaz_nutribot.py`
Interfaz gráfica principal en Tkinter. Gestiona:
- Visualización en tiempo real de CAM1 y CAM2
- Detección YOLO de alimentos en CAM1 (publica `objetivo` en `estado.json`)
- Reconocimiento de voz ("quiero comer", "quiero sopa", "para", "pausa")
- Síntesis de voz TTS para mensajes al usuario
- Control de botones: Iniciar / Pausar / Detener

### `orquestador.py`
Máquina de estados central. Coordina:
- Modo REPOSO → espera comandos
- Modo SÓLIDO → ciclo de agarre: HOME → DETECCIÓN → PREDICCIÓN MLP → AGARRE → VERIFICACIÓN → ENTREGA
- Modo LÍQUIDO → toma de cuchara (verificación ArUco) → bucle de cucharadas → entrega adaptativa
- Gestión compartida de cámaras entre interfaz y orquestador (campo `cam_owner`)
- Worker MediaPipe en subproceso separado para detección de distancia boca

### `auto_brazo_completo_PRESENTACION.py`
Módulo núcleo que implementa:
- `BrazoMLP` — arquitectura de la red neuronal (2→128→128→64→3)
- `MLPPredictor` — carga `modelo_bc.pt` y predice pasos de articulaciones
- `Cuadricula` — interpreta `calibracion_cuadricula.json` para mapeo de celdas
- `RobotInterface` — gestión de conexión serial y envío de comandos al Arduino

### `auto_brazo_cuchara_definitivo.py`
Módulo standalone para el modo sopa. Puede ejecutarse de forma independiente del orquestador. Incluye su propia lógica de comunicación serial, detección ArUco para verificar agarre de la cuchara y entrega adaptativa con MediaPipe.

### `train_mlp.py`
Entrena el modelo de Behavior Cloning:
- Lee `demonstrations_clean.pkl` generado por los scripts de recolección
- Arquitectura: MLP con capas `[128, 128, 64]`, LayerNorm, Dropout, Huber Loss
- Normalización estándar de las salidas (pasos por articulación)
- Guarda `modelo_bc.pt` con pesos + parámetros de normalización

### `calibrar_plato_sopa_avanzado.py`
Herramienta interactiva para calibrar el color HSV del plato de sopa. Genera `calibracion.pkl` con la referencia de color y tolerancias usadas por el detector de nivel.

### `collect_demos_corners.py` / `collect_demos_cuadricula.py`
Scripts de recolección de demostraciones humanas. El operador mueve el brazo manualmente hacia distintos trozos de comida mientras el script registra la posición visual (cx_norm, cy_norm) y los pasos de cada articulación. Los datos se guardan en `demonstrations_clean.pkl`.

### `brazo_robotico_ASISTENTE.ino`
Firmware Arduino. Recibe comandos por puerto serial a 115200 baudios:
- `BASE <pasos>` — mueve el eje de base
- `HOMBRO <pasos>` — mueve el hombro
- `CODO <pasos>` — mueve el codo
- `GRIPPER <ángulo>` — controla la pinza (servo)
- `GIRO <pasos>` — mueve el eje de rotación

---

## Entrenamiento del modelo MLP

El modelo de Behavior Cloning aprende a mapear la posición normalizada de un trozo de comida en la imagen `(cx_norm, cy_norm)` a los pasos netos de cada articulación `(Δbase, Δcodo, Δhombro)`.

```bash
# Entrenamiento con parámetros por defecto
python train_mlp.py

# Con parámetros personalizados
python train_mlp.py --epochs 5000 --lr 0.0003 --batch 64

# Especificando rutas
python train_mlp.py --dataset demos/demonstrations_clean.pkl --output modelo_bc.pt
```

#### Parámetros de entrenamiento

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `--epochs` | `3000` | Número de épocas |
| `--lr` | `0.001` | Learning rate (Adam) |
| `--batch` | `32` | Tamaño de batch |
| `--dataset` | ruta absoluta | Ruta al `.pkl` de demostraciones |
| `--output` | ruta absoluta | Ruta de salida del `.pt` |

#### Arquitectura

```
Entrada:  (cx_norm, cy_norm)          — 2 valores en [0, 1]
          ↓
          Linear(2→128) + LayerNorm + ReLU + Dropout(0.05)
          Linear(128→128) + LayerNorm + ReLU + Dropout(0.05)
          Linear(128→64) + LayerNorm + ReLU + Dropout(0.05)
          ↓
          Linear(64→3)
          ↓
Salida:   (Δbase, Δcodo, Δhombro)     — pasos netos por articulación
```

El archivo `.pt` guardado incluye:
- Pesos de la red (`model_state_dict`)
- Parámetros de normalización (`y_mean`, `y_std`)
- Metadatos de arquitectura (`hidden`, `input_dim`, `output_dim`)

---

## Calibración

### Calibración de cuadrícula (modo sólido)

La cuadrícula mapea posiciones de la imagen a celdas físicas del plato. Se genera con `collect_demos_cuadricula.py` y se guarda en `calibracion_cuadricula.json` (cuadrícula de 12×9 por defecto).

### Calibración del plato de sopa

```bash
python calibrar_plato_sopa_avanzado.py
```

El script abre la cámara del plato, permite seleccionar la región de referencia interactivamente y guarda:
- `calibracion_sopa/calibracion_plato.pkl` — datos HSV de referencia
- `calibracion_sopa/calibracion_plato_ref.png` — imagen de referencia
- `calibracion_sopa/calibracion_plato_stats.json` — tolerancias por canal

También genera `calibracion.pkl` en el directorio raíz para uso del orquestador.

---

## Recolección de demostraciones

Para re-entrenar el modelo con nuevos datos:

```bash
# Cuadrícula uniforme (recomendado para cobertura completa del plato)
python collect_demos_cuadricula.py

# Esquinas y puntos clave
python collect_demos_corners.py
```

Los scripts guían al operador para mover el brazo a cada punto de la cuadrícula/esquinas, registrando automáticamente los movimientos. Las demostraciones se acumulan en `demos/demonstrations_clean.pkl`.

---

## Hardware

### Conexiones esperadas

| Componente | Conexión |
|---|---|
| Arduino | USB → Puerto `COM3` (configurable) |
| Cámara plato (CAM1) | USB → índice `2` o `3` (configurable) |
| Cámara frontal (CAM2) | USB → índice `1` (configurable) |
| Marcador ArUco | Pegado a la cuchara (ID=1, DICT_4X4_50) |

### Ejes del brazo

| Eje | Comando Arduino | Dirección positiva |
|---|---|---|
| Base | `BASE` | Rotación horaria |
| Hombro | `HOMBRO` | Elevación |
| Codo | `CODO` | Extensión |
| Muñeca / Gripper | `GRIPPER` | Cierre de pinza |
| Rotación | `GIRO` | Rotación de muñeca |

### Posición HOME

```python
HOME_POSITION = {"base": 0, "hombro": 400, "codo": 400, "muneca": 0, "rotacion": 0}
```

---

## Solución de problemas

**El Arduino no responde**
- Verifica que el puerto COM en el script coincida con el del Administrador de dispositivos
- Comprueba que el baudrate es `115200` tanto en el `.ino` como en el script Python
- Asegúrate de que el Arduino IDE no tiene el monitor serial abierto

**Las cámaras no se abren**
- Ajusta `CAM1_INDEX` / `CAM2_INDEX` probando índices `0`, `1`, `2`, `3`
- En Windows, usa `cv2.CAP_DSHOW` (ya configurado en el código)
- Cierra cualquier otra aplicación que use las cámaras

**YOLO no detecta alimentos**
- Baja `YOLO_CONF` de `0.20` a `0.10` para mayor sensibilidad
- Asegúrate de que `yolov8x-worldv2.pt` está en el directorio correcto
- Verifica que la iluminación del plato es uniforme

**Error al cargar `modelo_bc.pt`**
- Verifica que la ruta en `MODELO_PT` es correcta
- Si el modelo no existe, recoge demostraciones y entrena con `train_mlp.py`

**El ArUco no detecta la cuchara**
- Asegúrate de que el marcador ID=1 del diccionario `DICT_4X4_50` está impreso y plano
- Aumenta la iluminación sobre el área de agarre
- Ajusta `minMarkerPerimeterRate` en `ArucoDetector` si el marcador es muy pequeño

**MediaPipe no detecta el rostro**
- Verifica que `CAM2_INDEX` apunta a la cámara frontal
- Comprueba que hay buena iluminación en la cara del usuario
- El worker MediaPipe se lanza como subproceso; revisa los logs `[MP-PROXY]` en consola

---

## Licencia

Proyecto académico desarrollado para asistencia en la alimentación de personas con movilidad reducida.