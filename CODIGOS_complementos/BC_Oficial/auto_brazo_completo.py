"""
auto_brazo_completo.py  —  Bucle Automático Completo con MediaPipe
===================================================================
Brazo robótico asistente de alimentación — SIN necesidad de teclas.

Autores : Jesús Adrián Ovando / Carla Andrea Quispe López
Hardware: Arduino Mega 2560 + RAMPS 1.4 + NEMA17 + A4988

FLUJO POR CICLO:
  [1]  HOME
  [2]  Gripper CERRADO (posición de captura)
  [3]  Cámara 1 (cam2) abre → YOLO detecta trozo automáticamente → cierra
  [4]  Gripper ABRE → MLP predice → mueve Base→Codo→Hombro
  [5]  Gripper CIERRA (agarre con gripper)
  [6]  HOME (Hombro→Codo→Base)
  [7]  Cámara 2 (cam3) abre con MediaPipe
  [8]  Brazo acerca: CODO+600 → HOMBRO+500 → alternando con chequeo de distancia
  [9]  Alarma (dist < THRESHOLD_CM) → brazo se DETIENE
  [10] Espera que persona coma (dist < EATING_CM)
  [11] Espera que persona aleje cara (dist > FAR_CM) → brazo vuelve a HOME
  [12] HOME → Gripper ABRE → Gripper CIERRA → vuelve a [1]
  [FIN] Si no hay más trozos detectados → programa termina

USO:
  python auto_brazo_completo.py
  python auto_brazo_completo.py --sim          # Simulación sin hardware
  python auto_brazo_completo.py --cam1 2 --cam2 3
  python auto_brazo_completo.py --threshold 15 --eating 8 --far 25
"""

import json
import time
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp

# ═══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — Ajusta estas rutas y parámetros
# ═══════════════════════════════════════════════════════════════

MODELO_PT  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\modelo_bc.pt"
CALIB_JSON = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\calibracion_cuadricula.json"

SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200

CAMERA_1_INDEX = 2   # Cámara montada en el brazo (YOLO + cuadrícula)
CAMERA_2_INDEX = 3   # Cámara fija apuntando a la persona (MediaPipe)

# ── Posición HOME ─────────────────────────────────────────────
HOME_POSITION = {"base": 0, "hombro": 400, "codo": 400, "muneca": 0, "rotacion": 0}
JOINT_LIMITS  = {ax: (-3200, 3200) for ax in HOME_POSITION}
AXIS_CMD      = {
    "base":     "BASE",
    "hombro":   "HOMBRO",
    "codo":     "CODO",
    "muneca":   "GRIPPER",
    "rotacion": "GIRO",
}

PINZA_MIN = 0    # Gripper cerrado
PINZA_MAX = 90   # Gripper abierto

GRID_COLS = 12
GRID_ROWS = 9

# ── Captura automática (Cámara 1) ────────────────────────────
WARMUP_FRAMES  = 15   # Frames descartados al inicio (estabilización)
DETECT_FRAMES  = 60   # Máximo frames para buscar detección
MIN_CONF_YOLO  = 0.20 # Confianza mínima para aceptar detección

# ── Distancias MediaPipe (cm) ─────────────────────────────────
THRESHOLD_CM = 15.0   # Distancia de ALARMA: brazo se detiene
EATING_CM    = 12.0   # Distancia "persona comiendo" (se acerca a menos de 12cm)
FAR_CM       = 12.0   # Distancia "persona alejada" (vuelve a más de 12cm → terminó)

# ── Secuencia de aproximación hacia la cara ───────────────────
# Alternancia CODO/HOMBRO hasta que suene la alarma
APPROACH_SEQ = [
    ("codo",   +800),
    ("hombro", +500),
]
MAX_APPROACH_MOVES = 20  # Seguridad: máximo movimientos antes de forzar parada

# ── YOLO — clases de alimentos ────────────────────────────────
YOLO_CONF = 0.20
YOLO_CLASSES = [
    "apple","pear","peach","plum","apricot","cherry","strawberry","raspberry",
    "blueberry","blackberry","grape","watermelon","melon","cantaloupe","honeydew",
    "banana","mango","papaya","pineapple chunk","kiwi","orange slice","mandarin",
    "lemon slice","lime slice","fig","date","lychee","guava","passion fruit",
    "dragon fruit","star fruit","persimmon","pomegranate seed","tomato",
    "cherry tomato","carrot piece","broccoli floret","cauliflower","lettuce piece",
    "cucumber slice","zucchini","eggplant","bell pepper","corn kernel","pea",
    "green bean","asparagus","artichoke","celery piece","beet","radish","turnip",
    "potato chunk","sweet potato","mushroom","onion piece","leek","spinach","kale",
    "cabbage piece","brussels sprout","bok choy","chicken piece","beef piece",
    "pork piece","lamb piece","turkey piece","sausage slice","meatball","nugget",
    "shrimp","fish piece","salmon chunk","tuna piece","squid piece","octopus piece",
    "crab meat","boiled egg","fried egg piece","omelette piece","tofu cube",
    "tempeh piece","pasta piece","noodle","gnocchi","dumpling","rice ball",
    "bread piece","crouton","cheese cube","mozzarella","ham piece","olive",
    "pickle slice","sun-dried tomato","chickpea","lentil","bean",
    "food piece","fruit piece","vegetable piece","meat piece",
]
COLORES = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,255,0),(100,255,100),(255,150,50),(50,200,255),(200,50,255),
]


# ═══════════════════════════════════════════════════════════════
#  MLP — Arquitectura idéntica a train_mlp.py
# ═══════════════════════════════════════════════════════════════

class BrazoMLP(nn.Module):
    def __init__(self, hidden, dropout=0.05):
        super().__init__()
        capas = []
        in_dim = 2
        for h in hidden:
            capas += [nn.Linear(in_dim, h), nn.LayerNorm(h),
                      nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        capas.append(nn.Linear(in_dim, 3))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


class MLPPredictor:
    """Carga modelo_bc.pt y predice (base, codo, hombro) en pasos."""

    def __init__(self, pt_path, simulate=False):
        if simulate:
            self.output_names = ["base", "codo", "hombro"]
            self.y_mean       = np.zeros(3, dtype=np.float32)
            self.y_std        = np.ones(3,  dtype=np.float32)
            self.model        = None
            print("[MLP] Modo SIMULADO — predicciones ficticias.")
            return

        print(f"[MLP] Cargando modelo: {pt_path}")
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

        self.y_mean       = np.array(ckpt["y_mean"], dtype=np.float32)
        self.y_std        = np.array(ckpt["y_std"],  dtype=np.float32)
        self.output_names = ckpt.get("output_names", ["base", "codo", "hombro"])

        hidden  = ckpt["hidden"]
        dropout = ckpt.get("dropout", 0.05)
        self.model = BrazoMLP(hidden, dropout)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        n_params  = ckpt.get("n_params", "?")
        trained   = ckpt.get("trained_at", "?")
        print(f"[MLP] Arq: 2 → {hidden} → 3 | Params: {n_params} | Entrend: {trained}")
        print(f"[MLP] Salidas: {self.output_names}")
        print(f"[MLP] y_mean={self.y_mean.tolist()}  y_std={self.y_std.tolist()}\n")

    def predecir(self, cx_norm, cy_norm):
        """Devuelve dict con pasos predichos para cada articulación."""
        if self.model is None:
            # Modo simulado: valores de ejemplo
            return {"base": 50, "codo": 200, "hombro": 150}
        x = torch.tensor([[cx_norm, cy_norm]], dtype=torch.float32)
        with torch.no_grad():
            y_norm = self.model(x).numpy()[0]
        y_raw = y_norm * self.y_std + self.y_mean
        return {name: int(round(float(v)))
                for name, v in zip(self.output_names, y_raw)}


# ═══════════════════════════════════════════════════════════════
#  CUADRÍCULA — Perspectiva + zona activa
# ═══════════════════════════════════════════════════════════════

class Cuadricula:
    COLOR_LINEA = (0, 255, 255)

    def __init__(self, json_path):
        self.cols = GRID_COLS
        self.rows = GRID_ROWS
        self.M = self.M_inv = None
        try:
            with open(json_path, "r") as f:
                d = json.load(f)
            self.cols = d["grid_cols"]
            self.rows = d["grid_rows"]
            src = np.float32([[0,0],[self.cols,0],[self.cols,self.rows],[0,self.rows]])
            dst = np.float32([tuple(p) for p in d["puntos"]])
            self.M     = cv2.getPerspectiveTransform(src, dst)
            self.M_inv = np.linalg.inv(self.M)
            print(f"[Cuadricula] {self.cols}×{self.rows} | calibración cargada.")
        except Exception as e:
            print(f"[Cuadricula] AVISO: {e} — sin calibración de perspectiva.")

    @property
    def disponible(self):
        return self.M is not None

    def _t(self, pts):
        return cv2.perspectiveTransform(
            np.float32(pts).reshape(-1, 1, 2), self.M).reshape(-1, 2)

    def poligono_px(self):
        if not self.disponible: return None
        return self._t([[0,0],[self.cols,0],
                        [self.cols,self.rows],[0,self.rows]]).astype(np.int32)

    def punto_dentro(self, px, py):
        poly = self.poligono_px()
        if poly is None: return True   # Sin calibración: todo vale
        return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0

    def info_celda(self, px, py):
        if not self.disponible:
            return {"celda": None, "fila": None, "columna": None}
        gx, gy = cv2.perspectiveTransform(
            np.float32([[[px, py]]]), self.M_inv)[0][0]
        if 0 <= gx <= self.cols and 0 <= gy <= self.rows:
            col = min(int(gx), self.cols-1)
            row = min(int(gy), self.rows-1)
            return {"celda": row*self.cols+col+1,
                    "fila": row+1, "columna": col+1}
        return {"celda": None, "fila": None, "columna": None}

    def dibujar(self, frame, highlight_celda=None):
        """Dibuja cuadrícula con perspectiva sobre el frame."""
        if not self.disponible: return frame
        # Fondo oscuro semitransparente
        ov = frame.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                num = r*self.cols + c + 1
                pts = self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
                col = (0, 80, 180) if num == highlight_celda else (0, 40, 40)
                cv2.fillPoly(ov, [pts], col)
        cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)
        # Celda objetivo en naranja
        if highlight_celda:
            r = (highlight_celda-1)//self.cols
            c = (highlight_celda-1) % self.cols
            pts = self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
            cv2.fillPoly(frame, [pts], (0, 140, 255))
        # Líneas de la cuadrícula
        for c in range(self.cols+1):
            p1 = self._t([[c, 0]]).astype(int)[0]
            p2 = self._t([[c, self.rows]]).astype(int)[0]
            grosor = 2 if c in (0, self.cols) else 1
            cv2.line(frame, tuple(p1), tuple(p2), self.COLOR_LINEA, grosor)
        for r in range(self.rows+1):
            p1 = self._t([[0, r]]).astype(int)[0]
            p2 = self._t([[self.cols, r]]).astype(int)[0]
            grosor = 2 if r in (0, self.rows) else 1
            cv2.line(frame, tuple(p1), tuple(p2), self.COLOR_LINEA, grosor)
        # Números de celda
        for r in range(self.rows):
            for c in range(self.cols):
                num = r*self.cols + c + 1
                cx = self._t([[c+0.5, r+0.5]])[0].astype(int)
                txt = str(num)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                col = (255,255,255) if num == highlight_celda else (255,255,0)
                cv2.putText(frame, txt,
                            (int(cx[0])-tw//2, int(cx[1])+th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
        return frame


# ═══════════════════════════════════════════════════════════════
#  ROBOT INTERFACE
# ═══════════════════════════════════════════════════════════════

class RobotInterface:
    def __init__(self, simulate=False):
        self.simulate = simulate
        self._pos     = dict(HOME_POSITION)
        self._pinza   = PINZA_MIN
        self._ser     = None

    def __enter__(self):
        if not self.simulate:
            import serial
            print(f"[Robot] Conectando {SERIAL_PORT} @ {SERIAL_BAUD}...")
            self._ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=15)
            time.sleep(2)
            self._ser.flushInput()
            # Esperar señal READY del Arduino
            deadline = time.time() + 20
            while time.time() < deadline:
                line = self._ser.readline().decode(errors="ignore").strip()
                if line == "READY":
                    break
            print(f"[Robot] Arduino listo.")
        else:
            print("[Robot] Modo SIMULADO — sin hardware.")
        return self

    def __exit__(self, *_):
        if self._ser and self._ser.is_open:
            self._ser.close()
        print("[Robot] Desconectado.")

    def _send(self, cmd):
        """Envía comando al Arduino y espera OK/ERR."""
        if self.simulate:
            print(f"    [SIM] ↑ {cmd}")
            time.sleep(0.12)
            return "OK"
        self._ser.flushInput()
        self._ser.write((cmd + "\n").encode())
        deadline = time.time() + 20
        while time.time() < deadline:
            line = self._ser.readline().decode(errors="ignore").strip()
            if line == "OK":  return "OK"
            if line == "ERR": return "ERR"
        print(f"    [TIMEOUT] {cmd}")
        return "TIMEOUT"

    def set_gripper(self, angulo, label=""):
        angulo = int(np.clip(angulo, PINZA_MIN, PINZA_MAX))
        etq    = f" ({label})" if label else ""
        print(f"  Gripper → {angulo}°{etq}")
        resp = self._send(f"PINZA {angulo}")
        if resp == "OK":
            self._pinza = angulo
        return resp

    def go_home(self):
        """Vuelve a HOME en orden seguro: Hombro → Codo → Muñeca → Rotación → Base."""
        print("  → HOME...")
        orden = ["hombro", "codo", "muneca", "rotacion", "base"]
        for ax in orden:
            tgt  = HOME_POSITION[ax]
            cur  = self._pos[ax]
            diff = tgt - cur
            if diff == 0:
                continue
            print(f"    {ax:10s}: {cur:+5d} → {tgt:+5d}  (Δ{diff:+d})")
            resp = self._send(f"{AXIS_CMD[ax]} {diff}")
            if resp == "OK":
                self._pos[ax] = tgt
        print("  → HOME OK.")

    def mover_eje(self, eje, pasos):
        """
        Mueve un eje de forma relativa (pasos positivo o negativo).
        Respeta los límites articulares.
        """
        cur  = self._pos[eje]
        new  = int(np.clip(cur + pasos, *JOINT_LIMITS[eje]))
        diff = new - cur
        if diff == 0:
            print(f"    {eje}: ya en límite ({cur}).")
            return "OK"
        print(f"    {eje:10s}: {cur:+5d} → {new:+5d}  (Δ{diff:+d})")
        resp = self._send(f"{AXIS_CMD[eje]} {diff}")
        if resp == "OK":
            self._pos[eje] = new
        return resp

    def ejecutar_prediccion(self, pred):
        """Ejecuta Base → Codo → Hombro según predicción del MLP."""
        orden_ejes = [("base","base"), ("codo","codo"), ("hombro","hombro")]
        for nombre, eje in orden_ejes:
            val = pred.get(nombre, 0)
            if val == 0:
                continue
            self.mover_eje(eje, val)
            time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════
#  MEDIAPIPE — Funciones de distancia y visualización
# ═══════════════════════════════════════════════════════════════

mp_face_mesh = mp.solutions.face_mesh

def _estimar_distancia(landmarks, shape):
    """
    Estima distancia (cm) al rostro.
    Usa separación ocular cuando la cara está lejos, nariz-barbilla cuando está cerca.
    """
    h, w = shape[:2]
    # Ojos
    le = landmarks.landmark[33]
    re = landmarks.landmark[263]
    dx = (re.x - le.x) * w
    dy = (re.y - le.y) * h
    eye_dist = np.sqrt(dx*dx + dy*dy)
    # Nariz a barbilla
    nose = landmarks.landmark[1]
    chin = landmarks.landmark[152]
    nc_dx = (chin.x - nose.x) * w
    nc_dy = (chin.y - nose.y) * h
    nose_chin = np.sqrt(nc_dx*nc_dx + nc_dy*nc_dy)
    # Elegir la métrica más confiable
    if eye_dist > 30:
        dist_cm = 5000.0 / eye_dist
    elif nose_chin > 0:
        dist_cm = 2500.0 / nose_chin
    else:
        dist_cm = 50.0
    return float(np.clip(dist_cm, 3.0, 65.0))

def _boca_centro(landmarks, shape):
    """Devuelve (cx, cy) del centro de la boca en píxeles."""
    h, w = shape[:2]
    ul = landmarks.landmark[13]
    ll = landmarks.landmark[14]
    cx = int((ul.x + ll.x) / 2 * w)
    cy = int((ul.y + ll.y) / 2 * h)
    return cx, cy

def _beep():
    """Emite pitido de alarma (Windows + fallback ASCII)."""
    try:
        import winsound
        winsound.Beep(1000, 200)
    except Exception:
        print("\a", end="", flush=True)

def _dibujar_hud_cam2(frame, dist, fase, threshold, eating_cm, far_cm):
    """Dibuja la información de estado en la ventana de cámara 2."""
    h, w = frame.shape[:2]

    # Barra de estado de fase
    colores_fase = {
        "aproximando": (0, 200, 255),
        "alarma":      (0, 0, 255),
        "comiendo":    (0, 255, 100),
        "alejando":    (100, 255, 200),
    }
    color_fase = colores_fase.get(fase, (200, 200, 200))

    # Borde de color según fase
    if fase == "alarma":
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 5)
    elif fase == "comiendo":
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,255,100), 3)

    # Distancia
    if dist is not None:
        color_dist = (0,0,255) if dist < threshold else (0,255,0)
        cv2.putText(frame, f"Distancia: {dist:.1f} cm", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_dist, 2)
    else:
        cv2.putText(frame, "Buscando rostro...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

    # Umbrales
    cv2.putText(frame, f"Umbral alarma: {threshold}cm  Comiendo: <{eating_cm}cm  Lejos: >{far_cm}cm",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

    # Mensaje de fase
    mensajes = {
        "aproximando": "APROXIMANDO AL ROSTRO...",
        "alarma":      "ALARMA — BRAZO DETENIDO",
        "comiendo":    "PERSONA COMIENDO — Esperando...",
        "alejando":    "Esperando que persona se aleje...",
    }
    msg = mensajes.get(fase, fase.upper())
    cv2.putText(frame, msg, (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_fase, 2)

    return frame


# ═══════════════════════════════════════════════════════════════
#  CAPTURA AUTOMÁTICA — Cámara 1 (YOLO)
# ═══════════════════════════════════════════════════════════════

def captura_automatica(cuadricula, yolo_model, simulate=False):
    """
    Abre cámara 1, estabiliza y detecta automáticamente el mejor trozo.

    Returns:
        dict con {cx_norm, cy_norm, celda_info, conf}  si hay detección
        None si no se detecta nada después de DETECT_FRAMES frames
    """
    if simulate:
        print("  [SIM] Detección ficticia: celda=15, cx=0.50, cy=0.45, conf=0.85")
        time.sleep(1.5)
        return {
            "cx_norm": 0.50,
            "cy_norm": 0.45,
            "celda_info": {"celda": 15, "fila": 2, "columna": 3},
            "conf": 0.85,
        }

    print(f"  Abriendo cámara 1 (índice {CAMERA_1_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_1_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"  ✗ ERROR: No se pudo abrir cámara {CAMERA_1_INDEX}")
        return None

    ventana = f"AUTO-DETECCIÓN — Cámara {CAMERA_1_INDEX}"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, 960, 580)
    cv2.moveWindow(ventana, 50, 50)

    # Warmup: descartar frames inestables
    print(f"  Estabilizando cámara ({WARMUP_FRAMES} frames warmup)...")
    for _ in range(WARMUP_FRAMES):
        cap.read()
        cv2.waitKey(1)

    mejor     = None
    frame_idx = 0
    print(f"  Esperando trozo de comida... (cámara abierta indefinidamente)")
    print(f"  La cámara se cerrará sola al detectar un trozo en la cuadrícula.")
    print(f"  Presiona ESC o Q en la ventana para salir del programa.")

    while True:   # ← Espera indefinida hasta encontrar un trozo
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(30)
            continue

        frame_idx += 1
        h, w = frame.shape[:2]

        # Inferencia YOLO
        results    = yolo_model.predict(frame, conf=YOLO_CONF, iou=0.25, verbose=False)[0]
        candidatos = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx_px = (x1 + x2) / 2
            cy_px = (y1 + y2) / 2

            # Solo trozos dentro de la cuadrícula
            if not cuadricula.punto_dentro(cx_px, cy_px):
                continue
            ci = cuadricula.info_celda(cx_px, cy_px)
            if ci["celda"] is None:
                continue
            conf = float(box.conf[0])
            if conf < MIN_CONF_YOLO:
                continue

            cls = int(box.cls[0])
            candidatos.append({
                "bbox":      (x1, y1, x2, y2),
                "cx_px":     cx_px,
                "cy_px":     cy_px,
                "cx_norm":   cx_px / w,
                "cy_norm":   cy_px / h,
                "conf":      conf,
                "celda_info": ci,
                "color":     COLORES[cls % len(COLORES)],
            })

        candidatos.sort(key=lambda x: x["conf"], reverse=True)

        # ── Preview en ventana ────────────────────────────────
        preview = frame.copy()
        hl = candidatos[0]["celda_info"]["celda"] if candidatos else None
        cuadricula.dibujar(preview, highlight_celda=hl)

        for c in candidatos:
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            cv2.rectangle(preview, (x1,y1), (x2,y2), c["color"], 2)
            cv2.circle(preview, (int(c["cx_px"]), int(c["cy_px"])), 8, (0,0,255), -1)
            ci_c = c["celda_info"]
            cv2.putText(preview,
                        f"C{ci_c['celda']} {c['conf']:.2f}",
                        (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if candidatos:
            estado = f"Frame {frame_idx} | Trozos en malla: {len(candidatos)} | OBJETIVO DETECTADO"
            color_estado = (0, 255, 0)
        else:
            estado = f"Frame {frame_idx} | Sin trozos en malla — esperando comida..."
            color_estado = (0, 165, 255)   # Naranja: esperando

        cv2.putText(preview, estado, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, color_estado, 2)
        cv2.putText(preview, "Deteccion AUTOMATICA | ESC/Q = salir programa",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1)
        cv2.imshow(ventana, preview)

        # Permitir salida manual con ESC o Q
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            print("  Salida manual por el usuario (ESC/Q).")
            cap.release()
            cv2.destroyWindow(ventana)
            cv2.waitKey(1)
            return "SALIR"   # Señal especial para terminar el programa

        # ── Primera detección válida: confirmar y salir ───────
        if candidatos:
            mejor = candidatos[0]
            # Mostrar confirmación visual durante 1 segundo
            conf_frame = preview.copy()
            x1, y1, x2, y2 = [int(v) for v in mejor["bbox"]]
            cv2.rectangle(conf_frame, (x1-5,y1-5), (x2+5,y2+5), (0,255,255), 4)
            ci_m = mejor["celda_info"]
            cv2.putText(conf_frame,
                        f"OBJETIVO: Celda {ci_m['celda']}  conf={mejor['conf']:.2f}  cx={mejor['cx_norm']:.3f}",
                        (10, h-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
            cv2.imshow(ventana, conf_frame)
            cv2.waitKey(900)   # 0.9 s de confirmación visual
            break

    cap.release()
    cv2.destroyWindow(ventana)
    cv2.destroyAllWindows()   # asegura que no queden ventanas residuales
    # Espera explícita: Windows necesita tiempo para soltar el driver USB
    # antes de que otra cámara pueda abrirse correctamente
    for _ in range(10):
        cv2.waitKey(50)
    time.sleep(2.0)           # 2 segundos de margen para liberar el bus USB

    if mejor:
        ci = mejor["celda_info"]
        print(f"  ✔ Objetivo: Celda {ci['celda']}  "
              f"cx={mejor['cx_norm']:.4f}  cy={mejor['cy_norm']:.4f}  "
              f"conf={mejor['conf']:.3f}")
    else:
        print("  ✗ Sin trozo en cuadrícula — plato vacío.")

    return mejor


# ═══════════════════════════════════════════════════════════════
#  FASE DE ENTREGA — Cámara 2 + MediaPipe + Aproximación
# ═══════════════════════════════════════════════════════════════

def fase_entrega(robot, simulate=False,
                 threshold_cm=THRESHOLD_CM,
                 eating_cm=EATING_CM,
                 far_cm=FAR_CM):
    """
    Abre cámara 2, acerca el brazo y espera que la persona coma.

    Fases internas:
      A) Aproximación: mueve CODO/HOMBRO alternando hasta que dist < threshold_cm
      B) Alarma:       brazo quieto, esperando que persona se acerque a comer
      C) Comiendo:     esperando que dist baje por debajo de eating_cm
      D) Terminó:      esperando que dist suba por encima de far_cm
    """
    if simulate:
        print("  [SIM] Fase entrega simulada...")
        for s, msg in [(1,"Aproximando..."), (1,"Alarma!"), (1,"Comiendo..."), (1,"Alejado.")]:
            print(f"    {msg}")
            time.sleep(s)
        return

    print(f"\n  Abriendo cámara 2 (índice {CAMERA_2_INDEX}) con MediaPipe...")
    print(f"  Esperando que el bus USB quede libre...")
    time.sleep(1.5)   # pausa extra por si cámara 1 aún no liberó el driver

    # CAP_DSHOW es obligatorio en Windows para que la cámara devuelva frames reales
    cap2 = cv2.VideoCapture(CAMERA_2_INDEX, cv2.CAP_DSHOW)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap2.set(cv2.CAP_PROP_FPS, 30)

    if not cap2.isOpened():
        print(f"  ✗ ERROR: No se pudo abrir cámara {CAMERA_2_INDEX}.")
        print(f"     Ejecuta detectar_camaras.py para verificar el índice correcto.")
        return

    # Warmup: descartar primeros frames — Windows necesita varios ciclos para estabilizar
    print(f"  Estabilizando cámara 2 (20 frames warmup)...")
    frames_ok = 0
    for i in range(40):                      # intentar hasta 40 lecturas
        ret, frame_test = cap2.read()
        if ret and frame_test is not None:
            frames_ok += 1
        cv2.waitKey(30)
        if frames_ok >= 20:                  # con 20 frames buenos ya está estable
            break

    if frames_ok == 0:
        print(f"  ✗ ERROR: Cámara {CAMERA_2_INDEX} se abre pero NO devuelve frames.")
        print(f"     Prueba otro índice o revisa que la cámara no esté en uso.")
        cap2.release()
        return

    print(f"  ✔ Cámara 2 lista ({frames_ok} frames de warmup OK).")

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ventana2 = f"MediaPipe Seguridad — Camara {CAMERA_2_INDEX}"
    cv2.namedWindow(ventana2, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana2, 700, 540)
    cv2.moveWindow(ventana2, 980, 50)

    # ── Helper: leer distancia actual ──────────────────────────
    def leer_distancia_y_mostrar(fase):
        """
        Captura frame de cam2, procesa MediaPipe, muestra HUD.
        Reintenta hasta 5 veces si el frame falla.
        Returns: (distancia_cm o None, frame_anotado o None)
        """
        frame = None
        for _ in range(5):               # reintentos ante frames corruptos
            ret, f = cap2.read()
            if ret and f is not None:
                frame = f
                break
            cv2.waitKey(10)

        if frame is None:
            # Mostrar pantalla de aviso en lugar de imagen negra
            aviso = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(aviso, f"Camara {CAMERA_2_INDEX} sin señal",
                        (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,200), 2)
            cv2.putText(aviso, "Verificando conexion...",
                        (150, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
            cv2.imshow(ventana2, aviso)
            cv2.waitKey(1)
            return None, aviso

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        dist = None
        if res.multi_face_landmarks:
            lm   = res.multi_face_landmarks[0]
            dist = _estimar_distancia(lm, frame.shape)
            # Punto de boca
            bx, by = _boca_centro(lm, frame.shape)
            cv2.circle(frame, (bx, by), 12, (0, 255, 255), -1)
            cv2.circle(frame, (bx, by), 12, (255, 255, 255), 2)
            # Ojos
            h_f, w_f = frame.shape[:2]
            le = lm.landmark[33]
            re = lm.landmark[263]
            cv2.circle(frame, (int(le.x*w_f), int(le.y*h_f)), 5, (255,0,0), -1)
            cv2.circle(frame, (int(re.x*w_f), int(re.y*h_f)), 5, (255,0,0), -1)

        frame = _dibujar_hud_cam2(frame, dist, fase,
                                  threshold_cm, eating_cm, far_cm)
        cv2.imshow(ventana2, frame)
        cv2.waitKey(1)
        return dist, frame

    # ══════════════════════════════════════════════════
    #  FASE A: Aproximación hasta alarma
    # ══════════════════════════════════════════════════
    print(f"\n  ─── FASE A: Aproximación al rostro ───")
    print(f"  (CODO+600 → HOMBRO+500 → alternando hasta dist < {threshold_cm}cm)")

    approach_idx    = 0
    last_beep_time  = 0
    alarm_triggered = False

    while approach_idx < MAX_APPROACH_MOVES:
        # Leer distancia antes de mover
        dist, _ = leer_distancia_y_mostrar("aproximando")

        if dist is None:
            # ⚠️ Sin rostro: el brazo NO se mueve, espera hasta detectar cara
            print(f"  Sin rostro detectado — brazo quieto, esperando cara...")
            time.sleep(0.1)
            continue

        if dist < threshold_cm:
            # ¡ALARMA! Brazo se detiene
            alarm_triggered = True
            if time.time() - last_beep_time > 1.5:
                _beep()
                last_beep_time = time.time()
                print(f"  🔴 ALARMA — dist={dist:.1f}cm < {threshold_cm}cm — BRAZO DETENIDO")
            break

        # Rostro detectado y en zona segura: ejecutar movimiento de aproximación
        eje, pasos = APPROACH_SEQ[approach_idx % len(APPROACH_SEQ)]
        print(f"  dist={dist:.1f}cm — moviendo {eje} +{pasos} (paso {approach_idx+1})")
        robot.mover_eje(eje, pasos)
        approach_idx += 1
        time.sleep(0.25)  # Pequeña pausa tras mover para estabilizar cámara

    if approach_idx >= MAX_APPROACH_MOVES and not alarm_triggered:
        print(f"  AVISO: Se alcanzó el límite de {MAX_APPROACH_MOVES} movimientos sin alarma.")
        print(f"  Continuando con el flujo de entrega...")

    _beep()  # Confirmar parada

    # ══════════════════════════════════════════════════
    #  FASE B: Brazo quieto — esperar que persona se acerque a comer
    # ══════════════════════════════════════════════════
    print(f"\n  ─── FASE B: Brazo quieto — esperando acercamiento ───")
    print(f"  (Esperando dist < {eating_cm}cm para confirmar inicio de comida)")

    eating_started = False
    while not eating_started:
        dist, _ = leer_distancia_y_mostrar("alarma")
        if time.time() - last_beep_time > 2.0:
            _beep()
            last_beep_time = time.time()
        if dist is not None and dist < eating_cm:
            print(f"  ✔ Persona comiendo (dist={dist:.1f}cm < {eating_cm}cm)")
            eating_started = True

    # ══════════════════════════════════════════════════
    #  FASE C: Persona comiendo — esperar que aleje cara
    # ══════════════════════════════════════════════════
    print(f"\n  ─── FASE C: Persona comiendo — esperando alejamiento ───")
    print(f"  (Esperando dist > {far_cm}cm para liberar brazo y volver a HOME)")

    eating_done = False
    while not eating_done:
        dist, _ = leer_distancia_y_mostrar("comiendo")
        # Termina si: persona está lejos O cara no detectada (se fue)
        if dist is None or dist > far_cm:
            if dist is None:
                print("  ✔ Rostro perdido — se asume que terminó de comer.")
            else:
                print(f"  ✔ Persona alejada (dist={dist:.1f}cm > {far_cm}cm) — liberando brazo.")
            eating_done = True

    # Cerrar cámara 2
    cap2.release()
    face_mesh.close()
    cv2.destroyWindow(ventana2)
    cv2.waitKey(1)
    print("  Cámara 2 cerrada.")


# ═══════════════════════════════════════════════════════════════
#  MAIN — Bucle principal automático
# ═══════════════════════════════════════════════════════════════

def main(simulate=False, threshold_cm=THRESHOLD_CM,
         eating_cm=EATING_CM, far_cm=FAR_CM):

    print("=" * 68)
    print("  BRAZO ROBÓTICO ASISTENTE — MODO COMPLETAMENTE AUTOMÁTICO")
    print("  Sin teclas. Bucle hasta que no haya más trozos en el plato.")
    print("=" * 68)
    print(f"  Cámara 1 (YOLO)    : índice {CAMERA_1_INDEX}")
    print(f"  Cámara 2 (MediaPipe): índice {CAMERA_2_INDEX}")
    print(f"  Umbral alarma       : {threshold_cm} cm")
    print(f"  Distancia comiendo  : < {eating_cm} cm")
    print(f"  Distancia alejado   : > {far_cm} cm")
    print(f"  Serial              : {SERIAL_PORT} @ {SERIAL_BAUD}")
    print(f"  Simulación          : {'SÍ' if simulate else 'NO'}")
    print("=" * 68)

    # ── Cargar recursos ───────────────────────────────────────
    cuadricula = Cuadricula(CALIB_JSON)
    predictor  = MLPPredictor(MODELO_PT, simulate=simulate)

    yolo_model = None
    if not simulate:
        from ultralytics import YOLOWorld
        print("\n[YOLO] Cargando YOLOWorld...")
        yolo_model = YOLOWorld("yolov8m-world.pt")
        yolo_model.set_classes(YOLO_CLASSES)
        print("[YOLO] Listo.\n")

    # ── Bucle principal ───────────────────────────────────────
    ciclo = 0

    with RobotInterface(simulate=simulate) as robot:
        continuar = True
        while continuar:
            ciclo += 1
            print(f"\n{'═'*68}")
            print(f"  CICLO {ciclo}")
            print(f"{'═'*68}")

            # ── [1] HOME ──────────────────────────────────────
            print(f"\n[1] HOME...")
            robot.go_home()

            # ── [2] Gripper cerrado (posición de captura) ─────
            print(f"\n[2] Gripper CERRADO (posición de captura)...")
            robot.set_gripper(PINZA_MIN, "captura")
            time.sleep(0.4)

            # ── [3] Detección automática con Cámara 1 ─────────
            print(f"\n[3] Detección automática de trozo objetivo...")
            objetivo = captura_automatica(cuadricula, yolo_model, simulate=simulate)

            if objetivo == "SALIR":
                print("\n  Salida manual del usuario (ESC/Q). FIN del programa.")
                continuar = False
                break
            if objetivo is None:
                # Seguridad: no debería ocurrir (la cámara espera indefinidamente)
                print("\n  WARN: captura devolvió None inesperado. Reintentando...")
                continue

            # ── [4] Predicción MLP ────────────────────────────
            ci   = objetivo["celda_info"]
            pred = predictor.predecir(objetivo["cx_norm"], objetivo["cy_norm"])

            print(f"\n[4] Predicción MLP:")
            print(f"    Entrada  : cx={objetivo['cx_norm']:.4f}  cy={objetivo['cy_norm']:.4f}")
            print(f"    Celda    : {ci['celda']} (Fila {ci['fila']}, Col {ci['columna']})")
            print(f"    Base     : {pred.get('base',0):+d} pasos")
            print(f"    Codo     : {pred.get('codo',0):+d} pasos")
            print(f"    Hombro   : {pred.get('hombro',0):+d} pasos")

            # ── [5] Gripper abre (pre-agarre) ─────────────────
            print(f"\n[5] Gripper ABIERTO (pre-agarre)...")
            robot.set_gripper(PINZA_MAX, "pre-agarre")
            time.sleep(0.5)

            # ── [6] Movimientos MLP: Base → Codo → Hombro ─────
            print(f"\n[6] Ejecutando movimientos MLP (Base → Codo → Hombro)...")
            robot.ejecutar_prediccion(pred)

            # ── [7] Gripper cierra: AGARRE ────────────────────
            print(f"\n[7] Gripper CERRADO — AGARRE con gripper...")
            robot.set_gripper(PINZA_MIN, "agarrando")
            time.sleep(0.8)

            # ── [8] HOME con comida agarrada ──────────────────
            print(f"\n[8] HOME (con comida en gripper) — Hombro→Codo→Base...")
            robot.go_home()
            time.sleep(0.3)

            # ── [9] Entrega con MediaPipe + aproximación ───────
            print(f"\n[9] Entrega — Cámara 2 + MediaPipe + Aproximación...")
            fase_entrega(
                robot,
                simulate=simulate,
                threshold_cm=threshold_cm,
                eating_cm=eating_cm,
                far_cm=far_cm,
            )

            # ── [10] HOME final ───────────────────────────────
            print(f"\n[10] HOME final (después de entrega)...")
            robot.go_home()

            # ── [11] Soltar y resetear gripper ────────────────
            print(f"\n[11] Gripper ABIERTO (soltando)...")
            robot.set_gripper(PINZA_MAX, "soltando")
            time.sleep(0.8)

            print(f"\n[12] Gripper CERRADO (listo para siguiente ciclo)...")
            robot.set_gripper(PINZA_MIN, "listo-siguiente")
            time.sleep(0.4)

            print(f"\n  ✔ Ciclo {ciclo} completado. Iniciando siguiente ciclo...\n")
            time.sleep(0.5)

    cv2.destroyAllWindows()
    ciclos_exitosos = ciclo - (1 if not continuar and ciclo > 0 else 0)
    print(f"\n{'='*68}")
    print(f"  Programa terminado. Trozos entregados: ~{ciclos_exitosos}")
    print(f"{'='*68}\n")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brazo robótico asistente — bucle automático completo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Simulación sin hardware real (no abre serial ni cámaras).",
    )
    parser.add_argument(
        "--modelo", default=MODELO_PT, metavar="RUTA",
        help="Ruta al archivo modelo_bc.pt",
    )
    parser.add_argument(
        "--calib", default=CALIB_JSON, metavar="RUTA",
        help="Ruta al archivo calibracion_cuadricula.json",
    )
    parser.add_argument(
        "--cam1", type=int, default=CAMERA_1_INDEX, metavar="IDX",
        help=f"Índice de cámara 1 — YOLO (default: {CAMERA_1_INDEX})",
    )
    parser.add_argument(
        "--cam2", type=int, default=CAMERA_2_INDEX, metavar="IDX",
        help=f"Índice de cámara 2 — MediaPipe (default: {CAMERA_2_INDEX})",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD_CM, metavar="CM",
        help=f"Distancia de alarma en cm (default: {THRESHOLD_CM})",
    )
    parser.add_argument(
        "--eating", type=float, default=EATING_CM, metavar="CM",
        help=f"Distancia 'persona comiendo' en cm (default: {EATING_CM})",
    )
    parser.add_argument(
        "--far", type=float, default=FAR_CM, metavar="CM",
        help=f"Distancia 'persona alejada' en cm (default: {FAR_CM})",
    )

    args = parser.parse_args()

    # Aplicar argumentos a variables globales
    MODELO_PT        = args.modelo
    CALIB_JSON       = args.calib
    CAMERA_1_INDEX   = args.cam1
    CAMERA_2_INDEX   = args.cam2

    try:
        main(
            simulate=args.sim,
            threshold_cm=args.threshold,
            eating_cm=args.eating,
            far_cm=args.far,
        )
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n\n[!] Interrumpido por el usuario (Ctrl+C). Programa terminado.")
        sys.exit(0)