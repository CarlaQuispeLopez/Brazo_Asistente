# Brazo Robótico Asistente – Módulo IA
### Jesús Adrián Ovando & Carla Andrea Quispe López

---

## Estructura del Proyecto

```
brazo_robotico/
├── config.py           ← Toda la configuración (pines, modelos, hiperparámetros)
├── robot_interface.py  ← Comunicación serial con Arduino/RAMPS 1.4
├── vision.py           ← Pipeline de visión: YOLO + Depth Anything
├── collect_demos.py    ← PASO 1: Grabar demostraciones humanas
├── train_bc.py         ← PASO 2: Entrenar política por imitación (BC)
├── train_rl.py         ← PASO 3: Refinar con Deep RL (PPO)
├── requirements.txt
└── demos/
    └── demonstrations.pkl   ← generado por collect_demos.py
└── models/
    ├── bc_policy.pth        ← generado por train_bc.py
    └── ppo_policy/          ← generado por train_rl.py
```

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) GPU con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Pipeline Completo – Paso a Paso

### PASO 0 – Configuración
Edita `config.py`:
- `SERIAL_PORT` → tu puerto USB del Arduino (ej: `"COM5"` o `"/dev/ttyUSB0"`)
- `YOLO_MODEL_PATH` → ruta al modelo YOLO entrenado en alimentos
- `FOOD_CLASSES` → lista de clases de alimentos que quieres detectar
- `GRIPPER_CLOSE` / `GRIPPER_OPEN` → ángulos del servo según tu gripper

### PASO 1 – Grabar Demostraciones (Aprendizaje por Imitación)
```bash
# Con hardware real:
python collect_demos.py

# Sin hardware (solo prueba el pipeline visual):
python collect_demos.py --sim

# Ver estadísticas de demos ya grabadas:
python collect_demos.py --analyze
```

**Controles del teclado durante la grabación:**
| Tecla | Acción |
|-------|--------|
| W / S | Hombro arriba / abajo |
| A / D | Base izquierda / derecha |
| Q / E | Codo sube / baja |
| R / F | Muñeca arriba / abajo |
| Z / C | Rotación izquierda / derecha |
| G | Cerrar gripper (agarre) |
| O | Abrir gripper |
| N | Nuevo episodio |
| H | Ir a HOME |
| ESC | Guardar y salir |

**¿Cuántas demos grabar?**
- Mínimo recomendado: **20-30 episodios exitosos**
- Cada episodio: mover el brazo desde la posición inicial hasta agarrar el alimento
- Variedad: graba desde diferentes posiciones del plato (izquierda, centro, derecha, cerca, lejos)

### PASO 2 – Entrenar Behavioral Cloning
```bash
# Entrenamiento estándar:
python train_bc.py

# Con más épocas:
python train_bc.py --epochs 200

# Solo con episodios exitosos:
python train_bc.py --only_success

# Evaluar modelo ya entrenado:
python train_bc.py --eval
```

El modelo aprende: **estado del mundo → acción que debe ejecutar**
(supervisión directa, como el modelo "aprende a imitar" al operador)

### PASO 3 – Refinar con Deep RL (PPO)
```bash
# Entrenamiento PPO inicializado con BC (recomendado):
python train_rl.py

# Sin hardware real:
python train_rl.py --sim

# Más pasos de entrenamiento:
python train_rl.py --steps 500000

# Sin inicialización BC (desde cero):
python train_rl.py --no_bc_init

# Continuar desde checkpoint:
python train_rl.py --resume models/ppo_policy/ppo_brazo_50000_steps.zip

# Ejecutar política entrenada:
python train_rl.py --run models/ppo_policy/ppo_final.zip
```

**Ver curvas de entrenamiento en TensorBoard:**
```bash
tensorboard --logdir models/ppo_policy/tb_logs
```

---

## Concepto Clave: "Yo te enseño, tú mejoras"

```
FASE 1 – Demos humanas
  Operador mueve el brazo + YOLO ve la comida
  → Se graban pares (estado_visual, acción_humana)

FASE 2 – Behavioral Cloning (IL)
  Red neuronal aprende: estado → acción
  (supervisión directa desde las demos)
  → Modelo BC: sabe hacer movimientos básicos

FASE 3 – Deep RL (PPO)
  Parte de los pesos del modelo BC
  → Explora posiciones NUEVAS de alimentos
  → Recibe recompensa: +10 si agarra, -0.01 por paso
  → Mejora la política para generalizarse a cualquier posición
```

---

## Espacio de Estado y Acción

**Estado (8 dimensiones):**
```
[food_cx_norm,    # posición horizontal del alimento en la imagen [0,1]
 food_cy_norm,    # posición vertical del alimento [0,1]
 food_depth_norm, # profundidad estimada por Depth Anything [0,1]
 j0_norm,         # posición articulación base [-1,1]
 j1_norm,         # posición hombro [-1,1]
 j2_norm,         # posición codo [-1,1]
 j3_norm,         # posición muñeca [-1,1]
 j4_norm]         # posición rotación [-1,1]
```

**Acciones (11 discretas):**
```
0  = base+      6  = muneca+
1  = base-      7  = muneca-
2  = hombro+    8  = rotacion+
3  = hombro-    9  = rotacion-
4  = codo+      10 = cerrar_gripper
5  = codo-
```

---

## Función de Recompensa

| Evento | Recompensa |
|--------|-----------|
| Cada paso | -0.01 |
| Acercarse al alimento | +proporcional |
| Centrar alimento en cámara | +0.1 max |
| **Agarre exitoso** | **+10.0** |
| Límite de articulación | -1.0 |
| Gripper cerrado lejos del alimento | -1.5 |

---

## Firmware Arduino

En `robot_interface.py` (al final del archivo) encontrarás el esqueleto
del firmware de Arduino como referencia (`ARDUINO_FIRMWARE_HINT`).
Completa el parsing de comandos y la lógica de movimiento según tu setup.

**Protocolo serial:**
- PC → Arduino: `"MOVE base 200 1 800\n"` (eje, pasos, dirección, velocidad_µs)
- Arduino → PC: `"OK\n"` seguido de `"DONE\n"` al terminar

---

## Notas Importantes

1. **Calibración**: Ajusta `JOINT_LIMITS` en `config.py` según los límites físicos
   reales de tu brazo para evitar daños mecánicos.

2. **Depth Anything**: El modelo retorna profundidad *relativa*, no absoluta.
   Calibra `DEPTH_MAX_CM` midiendo con una regla en tu escenario real.

3. **YOLO personalizado**: Para mejores resultados, fine-tunea YOLOv8 con fotos
   de los alimentos bolivianos que usarás (ensalada, frutas, etc.).

4. **Seguridad**: El código tiene límites de articulación implementados en
   `robot_interface.py`. Vérificalos antes de la primera ejecución.
