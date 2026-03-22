# ============================================================
# config.py — Configuración central del brazo robótico NutriBot
# v2: gripper stepper (sube/baja garra), sin rotacion, HOME real
# ============================================================

# ---------- Puerto Serial ----------
SERIAL_PORT    = "COM3"
SERIAL_BAUD    = 115200
SERIAL_TIMEOUT = 2

# ---------- Cámara ----------
CAMERA_INDEX = 0
FRAME_W      = 640
FRAME_H      = 480

# ---------- YOLO ----------
YOLO_MODEL_PATH = "yolov8m-world.pt"
YOLO_CONF_THR   = 0.35
YOLO_IOU_THR    = 0.3

FOOD_CLASSES = [
    "apple", "pear", "banana", "grape", "strawberry", "watermelon",
    "orange slice", "mango", "kiwi", "peach", "cherry",
    "tomato", "cherry tomato", "carrot piece", "broccoli floret",
    "cucumber slice", "bell pepper", "lettuce piece", "mushroom",
    "potato chunk", "corn kernel", "spinach",
    "chicken piece", "beef piece", "pork piece", "meatball", "nugget",
    "shrimp", "fish piece", "boiled egg", "tofu cube",
    "pasta piece", "rice ball", "bread piece", "dumpling",
    "food piece", "fruit piece", "vegetable piece", "meat piece",
]

# ---------- Depth Anything ----------
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Base-hf"

# Normalización de profundidad por frame (valores raw del modelo).
# El modelo devuelve profundidad RELATIVA, no métrica.
# depth_norm se calcula por percentil dentro del frame capturado.
# Estos límites son de referencia para detectar si algo está "cerca" o "lejos"
# en la escena del plato.
#
# Calibración realizada (calibrate_depth.py):
#   10 cm → raw 180.5214  (distancias 10-15 cm son inciertas)
#   15 cm → raw 231.8959  (idem)
#   20 cm → raw  82.7459
#   ~20 cm → raw 209.8783  (etiquetado 30cm, medido ~20cm)
#   ~24 cm → raw 211.1597  (etiquetado 40cm, medido ~24cm)
#
# NOTA: Los valores raw no son monotónicos con la distancia porque
# Depth Anything produce profundidad relativa a la escena completa.
# Se usa normalización por percentil del frame; no se convierte a cm.
#
# Rango observado de raw en el plato de comida:
DEPTH_RAW_MIN = 80.0    # objeto lejos del gripper  (referencia)
DEPTH_RAW_MAX = 235.0   # objeto cerca del gripper   (referencia)

# ---------- HOME — posición funcional de captura ----------
# Cuando el brazo está en HOME, la cámara enfoca bien el plato.
# Se toma la foto ANTES de cualquier movimiento.
HOME_POSITION = {
    "base":    0,
    "hombro":  100,
    "codo":    1000,
    "gripper": 0,   # gripper stepper (sube/baja la garra)
}

# ---------- Motores — ejes activos ----------
# base    → Arduino: BASE N
# hombro  → Arduino: HOMBRO N
# codo    → Arduino: CODO N
# gripper → Arduino: GRIPPER N   (sube/baja la garra, antes "muneca")
# GIRO no se usa (descartado del proyecto)
MOTOR_AXES = {
    "base":    {"cmd": "BASE"},
    "hombro":  {"cmd": "HOMBRO"},
    "codo":    {"cmd": "CODO"},
    "gripper": {"cmd": "GRIPPER"},
}

# Pinza (servo) — comandos seriales
PINZA_OPEN_CMD  = "PINZA ABRIR"
PINZA_CLOSE_CMD = "PINZA CERRAR"
PINZA_ANGLE_CMD = "PINZA"       # + " N" donde N ∈ [0, 90]

# ---------- Límites de articulación (en pasos) ----------
JOINT_LIMITS = {
    "base":    (-2300, 1000),
    "hombro":  (-2400, 2400),
    "codo":    (-2000, 2000),
    "gripper": (-400,   400),   # sube/baja la garra
}

# ---------- Movimiento ----------
STEPS_PER_REV   = 200
MICROSTEP       = 8
STEPS_FULL      = STEPS_PER_REV * MICROSTEP   # 1600 pasos/vuelta
DEFAULT_SPEED   = 800
MAX_JOINT_STEPS = 3200

# ---------- Gripper y agarre ----------
# Tiempo de estabilización mecánica tras cierre de pinza.
GRIPPER_STABILIZE_SECS = 1.0

# Pasos de hombro hacia arriba para confirmar que el agarre fue exitoso.
# El brazo levanta el hombro; si la pinza tiene algo, lo levanta.
# El operador o la lógica del RL verifica visualmente / por éxito de movimiento.
LIFT_SUCCESS_STEPS = 200

# ---------- Espacio de Estado y Acción ----------
# Estado (8 dimensiones):
#   [food_cx_norm, food_cy_norm, food_depth_norm,    ← desde snapshot
#    j_base, j_hombro, j_codo, j_gripper,            ← posiciones articulares
#    pinza_state]                                     ← 0=abierta, 1=cerrada
STATE_DIM  = 8

# Acciones (9 discretas):
#   0  base+        4  codo+
#   1  base-        5  codo-
#   2  hombro+      6  gripper+   (sube garra)
#   3  hombro-      7  gripper-   (baja garra)
#                   8  cerrar_pinza
ACTION_DIM   = 9
ACTION_STEPS = 100

# ---------- Recompensa ----------
REWARD_GRASP_SUCCESS = 10.0
REWARD_STEP_PENALTY  = -0.01
REWARD_DIST_SCALE    = 1.0
REWARD_COLLISION     = -5.0
MAX_EPISODE_STEPS    = 200

# Profundidad mínima normalizada para intentar agarre.
# Si food_depth_norm < este valor, el agarre es "prematuro" y se penaliza.
GRASP_MIN_DEPTH_NORM = 0.6   # ajustar empíricamente

# ---------- Aprendizaje por Imitación (BC) ----------
DEMO_DIR      = "demos"
DEMO_FILE     = "demos/demonstrations.pkl"
BC_MODEL_PATH = "models/bc_policy.pth"
BC_EPOCHS     = 100
BC_BATCH_SIZE = 64
BC_LR         = 1e-3
BC_HIDDEN_DIM = 256

# ---------- Aprendizaje por Refuerzo (PPO) ----------
RL_MODEL_PATH    = "models/ppo_policy"
RL_TOTAL_STEPS   = 200_000
RL_LEARNING_RATE = 3e-4
RL_N_STEPS       = 512
RL_BATCH_SIZE    = 64
RL_N_EPOCHS      = 10
RL_GAMMA         = 0.99
RL_ENT_COEF      = 0.01