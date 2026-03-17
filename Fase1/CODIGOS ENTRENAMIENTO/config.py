# ============================================================
# config.py — Configuración central del brazo robótico NutriBot
# ============================================================

# ---------- Puerto Serial ----------
SERIAL_PORT    = "COM3"
SERIAL_BAUD    = 115200
SERIAL_TIMEOUT = 2

# ---------- Cámara ----------
CAMERA_INDEX = 0
FRAME_W      = 640
FRAME_H      = 480
FPS_TARGET   = 30

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
DEPTH_MIN_CM     = 5.0
DEPTH_MAX_CM     = 60.0
DEPTH_EVERY_N_FRAMES = 6

# ---------- Calibración Depth Anything (raw → centímetros) ----------
# Medir con: python calibrate_depth.py
# Colocar objeto a cada distancia y registrar valor raw del modelo.
# Los puntos deben estar ordenados de mayor raw (más cerca) a menor raw (más lejos).
DEPTH_CALIB_RAW_POINTS = [0.95, 0.80, 0.60, 0.42, 0.28, 0.18]  # REEMPLAZAR
DEPTH_CALIB_CM_POINTS  = [10.0, 15.0, 20.0, 30.0, 40.0, 50.0]  # REEMPLAZAR

# ---------- Zona alcanzable del brazo en el frame ----------
# Medir con: python calibrate_reachable_zone.py
# Mover gripper a sus extremos y registrar coordenadas de píxel.
REACHABLE_ZONE_PX      = (160, 100, 480, 420)   # (x1, y1, x2, y2) REEMPLAZAR
REACHABLE_DEPTH_MIN_CM = 10.0                    # REEMPLAZAR
REACHABLE_DEPTH_MAX_CM = 45.0                    # REEMPLAZAR

# ---------- Motores — pines RAMPS 1.4 ----------
MOTOR_AXES = {
    "base":     {"step": 54, "dir": 55, "en": 38},
    "hombro":   {"step": 60, "dir": 61, "en": 56},
    "codo":     {"step": 46, "dir": 48, "en": 62},
    "muneca":   {"step": 26, "dir": 28, "en": 24},
    "rotacion": {"step": 36, "dir": 34, "en": 30},
}
SERVO_PIN     = 9
GRIPPER_OPEN  = 180
GRIPPER_CLOSE = 60

# ---------- Movimiento ----------
STEPS_PER_REV  = 200
MICROSTEP      = 8
STEPS_FULL     = STEPS_PER_REV * MICROSTEP   # 1600 pasos/vuelta
DEFAULT_SPEED  = 800
MAX_JOINT_STEPS = 3200

JOINT_LIMITS = {
    "base":     (-2300, 1000),
    "hombro":   (-2400, 2400),
    "codo":     (-2000, 2000),
    "muneca":   (-400,  400),
    "rotacion": (-400,  400),
}

# ---------- Fases del pipeline ----------
# SEARCH   → RL activo, YOLO + Depth activos, MediaPipe inactivo
# GRASP    → RL activo, YOLO + Depth activos, MediaPipe inactivo
# DELIVERY → RL detenido, controlador proporcional activo, MediaPipe activo

# Umbral de pasos del hombro para habilitar la fase DELIVERY.
# Medir con: python calibrate_shoulder_threshold.py
# Mover hombro hasta que el gripper esté a la altura de la boca de la persona.
DELIVERY_SHOULDER_STEPS = 1500    # REEMPLAZAR

# Tiempo de estabilización mecánica tras cierre del gripper (segundos).
# Medir con: python calibrate_stabilization.py
GRIPPER_STABILIZE_SECS = 1.5     # REEMPLAZAR

# Ratio de aumento de profundidad para confirmar agarre.
# Si después de cerrar el gripper la profundidad aumenta más de este porcentaje,
# se considera que el alimento ya no está en el plato (fue agarrado).
GRASP_DEPTH_LOSS_RATIO = 0.25

# Distancia gripper-boca (cm) para considerar la entrega completa y abrir el gripper.
MOUTH_PROXIMITY_CM = 12.0

# ---------- Controlador proporcional para DELIVERY ----------
# Ganancias del controlador P que guía el brazo hacia la boca.
# Ajustar empíricamente: aumentar si el brazo reacciona lento, bajar si oscila.
DELIVERY_KP_BASE    = 0.8   # ganancia base para corregir posición horizontal
DELIVERY_KP_HOMBRO  = 0.6   # ganancia hombro para corregir altura
DELIVERY_KP_CODO    = 0.4   # ganancia codo para corregir profundidad
DELIVERY_DEAD_ZONE  = 0.05  # zona muerta normalizada (no actuar si error < esto)
DELIVERY_STEPS_BASE = 50    # pasos por corrección discreta de base
DELIVERY_STEPS_HOMBRO = 50  # pasos por corrección discreta de hombro
DELIVERY_STEPS_CODO   = 50  # pasos por corrección discreta de codo

# ---------- MediaPipe ----------
FACE_CONF_THRESHOLD  = 0.6
FACE_TRACK_THRESHOLD = 0.6
MOUTH_UPPER_LIP_IDX  = 13
MOUTH_LOWER_LIP_IDX  = 14

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
RL_N_ENVS        = 1
RL_LEARNING_RATE = 3e-4
RL_N_STEPS       = 512
RL_BATCH_SIZE    = 64
RL_N_EPOCHS      = 10
RL_GAMMA         = 0.99
RL_ENT_COEF      = 0.01

# ---------- Espacio de Estado y Acción ----------
# El agente RL solo opera en fases SEARCH y GRASP.
# Estado: [food_cx_norm, food_cy_norm, food_depth_norm, j0..j4]
STATE_DIM  = 8
ACTION_DIM = 11
ACTION_STEPS = 100

# ---------- Recompensa ----------
REWARD_GRASP_SUCCESS = 10.0
REWARD_STEP_PENALTY  = -0.01
REWARD_DIST_SCALE    = 1.0
REWARD_COLLISION     = -5.0
MAX_EPISODE_STEPS    = 200