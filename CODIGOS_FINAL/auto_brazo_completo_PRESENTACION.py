import json
import time
import sys
import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp

MODELO_PT  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_FINAL\modelo_bc.pt"
CALIB_JSON = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_FINAL\calibracion_cuadricula.json"

SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

CAMERA_1_INDEX = 3
CAMERA_2_INDEX = 1

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
CAM2_FRAME_TMP = os.path.join(DIR_BASE, "nutribot_cam2_frame.npy")
CAM2_FRAME_TMP_WRITE = os.path.join(DIR_BASE, "nutribot_cam2_frame.tmp.npy")

HOME_POSITION = {"base": 0, "hombro": 400, "codo": 400, "muneca": 0, "rotacion": 0}
JOINT_LIMITS  = {ax: (-3200, 3200) for ax in HOME_POSITION}
AXIS_CMD      = {
    "base":     "BASE",
    "hombro":   "HOMBRO",
    "codo":     "CODO",
    "muneca":   "GRIPPER",
    "rotacion": "GIRO",
}

PINZA_MIN = 0
PINZA_MAX = 90

GRID_COLS = 12
GRID_ROWS = 9

WARMUP_FRAMES         = 15
DETECT_FRAMES         = 60
MIN_CONF_YOLO         = 0.20
VERIFY_WARMUP_FRAMES  = 20
VERIFY_CHECK_FRAMES   = 40
VERIFY_DIST_THRESHOLD = 0.10

CODO_BUSQUEDA_PASOS = 200
CODO_BUSQUEDA_MAX   = 20

APPROACH_SEQ = [
    ("hombro", +400),
    ("codo",   +600),
]
MAX_APPROACH_MOVES = 30

THRESHOLD_CM = 20.0

ALARM_RAISE_PASOS = 500

EATING_CM = 15.0

EATING_WAIT_SECS = 5.0

FAR_CM = 20.0

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
        print(f"[MLP] Arq: 2 -> {hidden} -> 3 | Params: {n_params} | Entrend: {trained}")
        print(f"[MLP] Salidas: {self.output_names}")
        print(f"[MLP] y_mean={self.y_mean.tolist()}  y_std={self.y_std.tolist()}\n")

    def predecir(self, cx_norm, cy_norm):
        if self.model is None:
            return {"base": 50, "codo": 200, "hombro": 150}
        x = torch.tensor([[cx_norm, cy_norm]], dtype=torch.float32)
        with torch.no_grad():
            y_norm = self.model(x).numpy()[0]
        y_raw = y_norm * self.y_std + self.y_mean
        return {name: int(round(float(v)))
                for name, v in zip(self.output_names, y_raw)}

class Cuadricula:

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
            print(f"[Cuadricula] {self.cols}x{self.rows} | calibración cargada.")
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
        if poly is None: return True
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
        if self.simulate:
            print(f"    [SIM] ^ {cmd}")
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
        print(f"  Gripper -> {angulo}°{etq}")
        resp = self._send(f"PINZA {angulo}")
        if resp == "OK":
            self._pinza = angulo
        return resp

    def go_home(self):
        print("  -> HOME...")
        orden = ["hombro", "codo", "muneca", "rotacion", "base"]
        for ax in orden:
            tgt  = HOME_POSITION[ax]
            cur  = self._pos[ax]
            diff = tgt - cur
            if diff == 0:
                continue
            print(f"    {ax:10s}: {cur:+5d} -> {tgt:+5d}  (d{diff:+d})")
            resp = self._send(f"{AXIS_CMD[ax]} {diff}")
            if resp == "OK":
                self._pos[ax] = tgt
        print("  -> HOME OK.")

    def mover_eje(self, eje, pasos):
        cur  = self._pos[eje]
        new  = int(np.clip(cur + pasos, *JOINT_LIMITS[eje]))
        diff = new - cur
        if diff == 0:
            print(f"    {eje}: ya en límite ({cur}).")
            return "OK"
        print(f"    {eje:10s}: {cur:+5d} -> {new:+5d}  (d{diff:+d})")
        resp = self._send(f"{AXIS_CMD[eje]} {diff}")
        if resp == "OK":
            self._pos[eje] = new
        return resp

    def ejecutar_prediccion(self, pred):
        orden_ejes = [("base","base"), ("codo","codo"), ("hombro","hombro")]
        for nombre, eje in orden_ejes:
            val = pred.get(nombre, 0)
            if val == 0:
                continue
            self.mover_eje(eje, val)
            time.sleep(0.05)

mp_face_mesh = mp.solutions.face_mesh

def _estimar_distancia(landmarks, shape):
    h, w = shape[:2]
    le = landmarks.landmark[33]
    re = landmarks.landmark[263]
    dx = (re.x - le.x) * w
    dy = (re.y - le.y) * h
    eye_dist = np.sqrt(dx*dx + dy*dy)
    nose = landmarks.landmark[1]
    chin = landmarks.landmark[152]
    nc_dx = (chin.x - nose.x) * w
    nc_dy = (chin.y - nose.y) * h
    nose_chin = np.sqrt(nc_dx*nc_dx + nc_dy*nc_dy)
    if eye_dist > 30:
        dist_cm = 5000.0 / eye_dist
    elif nose_chin > 0:
        dist_cm = 2500.0 / nose_chin
    else:
        dist_cm = 50.0
    return float(np.clip(dist_cm, 3.0, 65.0))

def _boca_centro(landmarks, shape):
    h, w = shape[:2]
    ul = landmarks.landmark[13]
    ll = landmarks.landmark[14]
    cx = int((ul.x + ll.x) / 2 * w)
    cy = int((ul.y + ll.y) / 2 * h)
    return cx, cy

def _hablar(texto):
    import threading
    def _speak():
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(texto)
        except Exception:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(texto)
                engine.runAndWait()
            except Exception:
                print(f"[VOZ] {texto}", flush=True)
    threading.Thread(target=_speak, daemon=True).start()

def _dibujar_hud_cam2(frame, dist, fase, threshold, eating_cm, far_cm):
    h, w = frame.shape[:2]
    colores_fase = {
        "buscando-rostro": (180, 180, 0),
        "aproximando": (0, 200, 255),
        "alarma":      (0, 0, 255),
        "comiendo":    (0, 255, 100),
        "alejando":    (100, 255, 200),
    }
    color_fase = colores_fase.get(fase, (200, 200, 200))
    if fase == "alarma":
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 5)
    elif fase == "comiendo":
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,255,100), 3)
    if dist is not None:
        color_dist = (0,0,255) if dist < threshold else (0,255,0)
        cv2.putText(frame, f"Distancia: {dist:.1f} cm", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_dist, 2)
    else:
        cv2.putText(frame, "Buscando rostro...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
    cv2.putText(frame, f"Umbral alarma: {threshold}cm  Comiendo: <{eating_cm}cm  Lejos: >{far_cm}cm",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    mensajes = {
        "buscando-rostro": "BUSCANDO ROSTRO — CODO subiendo...",
        "aproximando": "APROXIMANDO AL ROSTRO...",
        "alarma":      "ALARMA — BRAZO DETENIDO (8-12 cm)",
        "comiendo":    "PERSONA COMIENDO — Esperando...",
        "alejando":    "Esperando que persona se aleje...",
    }
    msg = mensajes.get(fase, fase.upper())
    cv2.putText(frame, msg, (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_fase, 2)
    return frame

def _publicar_frame_cam2(frame):
    if frame is None:
        return
    try:
        np.save(CAM2_FRAME_TMP_WRITE, frame)
        os.replace(CAM2_FRAME_TMP_WRITE, CAM2_FRAME_TMP)
    except Exception as e:
        print(f"[CAM2] No se pudo publicar frame para la interfaz: {e}", flush=True)

def captura_automatica(cuadricula, yolo_model, simulate=False):
    if simulate:
        print("  [SIM] Detección ficticia: celda=15, cx=0.50, cy=0.45, conf=0.85")
        time.sleep(1.5)
        return (
            {
                "cx_norm": 0.50,
                "cy_norm": 0.45,
                "celda_info": {"celda": 15, "fila": 2, "columna": 3},
                "conf": 0.85,
            },
            None, None,
        )

    print(f"  Abriendo cámara 1 (índice {CAMERA_1_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_1_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"  X ERROR: No se pudo abrir cámara {CAMERA_1_INDEX}")
        return None, None, None

    ventana = f"AUTO-DETECCIÓN — Cámara {CAMERA_1_INDEX}"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, 960, 580)
    cv2.moveWindow(ventana, 50, 50)

    print(f"  Estabilizando cámara ({WARMUP_FRAMES} frames warmup)...")
    for _ in range(WARMUP_FRAMES):
        cap.read()
        cv2.waitKey(1)

    mejor     = None
    frame_idx = 0
    print(f"  Esperando trozo de comida... (cámara abierta indefinidamente)")
    print(f"  Presiona ESC o Q en la ventana para salir del programa.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(30)
            continue

        frame_idx += 1
        h, w = frame.shape[:2]

        results    = yolo_model.predict(frame, conf=YOLO_CONF, iou=0.25, verbose=False)[0]
        candidatos = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx_px = (x1 + x2) / 2
            cy_px = (y1 + y2) / 2

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

        preview = frame.copy()

        for c in candidatos:
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]

            cv2.rectangle(preview, (x1, y1), (x2, y2), c["color"], 2)

            cv2.circle(preview, (int(c["cx_px"]), int(c["cy_px"])), 8, (0, 0, 255), -1)

            ci_c = c["celda_info"]
            cv2.putText(preview,
                        f"C{ci_c['celda']} {c['conf']:.2f}",
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if candidatos:
            estado       = (f"Frame {frame_idx} | "
                            f"Trozos detectados: {len(candidatos)} | OBJETIVO DETECTADO")
            color_estado = (0, 255, 0)
        else:
            estado       = (f"Frame {frame_idx} | "
                            f"Sin trozos en zona activa — esperando comida...")
            color_estado = (0, 165, 255)

        cv2.putText(preview, estado, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, color_estado, 2)
        cv2.putText(preview, "Deteccion AUTOMATICA | ESC/Q = salir programa",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        cv2.imshow(ventana, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            print("  Salida manual por el usuario (ESC/Q).")
            cap.release()
            cv2.destroyWindow(ventana)
            cv2.waitKey(1)
            return "SALIR", None, None

        if candidatos:
            mejor = candidatos[0]

            conf_frame = preview.copy()
            x1, y1, x2, y2 = [int(v) for v in mejor["bbox"]]

            cv2.rectangle(conf_frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 255), 4)

            cv2.circle(conf_frame,
                       (int(mejor["cx_px"]), int(mejor["cy_px"])),
                       12, (0, 255, 255), -1)
            cv2.circle(conf_frame,
                       (int(mejor["cx_px"]), int(mejor["cy_px"])),
                       12, (255, 255, 255), 2)
            ci_m = mejor["celda_info"]
            cv2.putText(conf_frame,
                        (f"OBJETIVO: Celda {ci_m['celda']}  "
                         f"conf={mejor['conf']:.2f}  "
                         f"cx={mejor['cx_norm']:.3f}"),
                        (10, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(conf_frame,
                        "Cam1 permanece abierta para verificacion de agarre",
                        (10, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)
            cv2.imshow(ventana, conf_frame)
            cv2.waitKey(900)
            break

    if mejor:
        ci = mejor["celda_info"]
        print(f"  OK Objetivo: Celda {ci['celda']}  "
              f"cx={mejor['cx_norm']:.4f}  cy={mejor['cy_norm']:.4f}  "
              f"conf={mejor['conf']:.3f}")
        print(f"  Cam1 permanece ABIERTA para verificación de agarre.")
    else:
        print("  X Sin trozo en zona activa — plato vacío.")
        cap.release()
        cv2.destroyWindow(ventana)
        cv2.waitKey(1)
        return None, None, None

    return mejor, cap, ventana

def verificar_agarre_cam1(cuadricula, yolo_model,
                          objetivo_cx_norm, objetivo_cy_norm,
                          cap_cam1=None, ventana_cam1=None,
                          simulate=False):
    if simulate:
        print("  [SIM] Verificación de agarre simulada -> EXITOSO")
        time.sleep(1.0)
        return True

    if cap_cam1 is not None and cap_cam1.isOpened():
        cap = cap_cam1
        ventana_v = (ventana_cam1 if ventana_cam1
                     else f"VERIFICACION AGARRE — Cámara {CAMERA_1_INDEX}")
        print(f"\n  Reutilizando Cam1 abierta para verificar agarre — sin retardo USB.")
        cv2.setWindowTitle(ventana_v, f"VERIFICACION AGARRE — Cámara {CAMERA_1_INDEX}")
        ventana_v_real = ventana_v
        abrio_nueva = False
    else:
        print(f"\n  Abriendo Cam1 para verificar agarre (índice {CAMERA_1_INDEX})...")
        time.sleep(1.0)
        cap = cv2.VideoCapture(CAMERA_1_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print(f"  X ERROR: No se pudo abrir Cam1. Se asume EXITOSO.")
            return True
        ventana_v_real = f"VERIFICACION AGARRE — Cámara {CAMERA_1_INDEX}"
        cv2.namedWindow(ventana_v_real, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ventana_v_real, 960, 580)
        cv2.moveWindow(ventana_v_real, 50, 50)
        abrio_nueva = True
        print(f"  Estabilizando Cam1 ({VERIFY_WARMUP_FRAMES} frames warmup)...")
        for _ in range(VERIFY_WARMUP_FRAMES):
            cap.read()
            cv2.waitKey(1)

    print(f"  Vaciando buffer de Cam1 (frames acumulados durante el movimiento)...")
    for _ in range(10):
        cap.grab()
    ret_flush, _ = cap.read()
    if not ret_flush:
        print(f"  AVISO: Cam1 no devolvió frame tras flush. Se asume EXITOSO.")
        cap.release()
        cv2.destroyAllWindows()
        return True

    print(f"  Buffer limpio. Analizando {VERIFY_CHECK_FRAMES} frames para verificar agarre...")

    trozo_encontrado = False
    frame_ultimo     = None

    for frame_idx in range(VERIFY_CHECK_FRAMES):
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(30)
            continue

        frame_ultimo = frame
        h, w = frame.shape[:2]

        results    = yolo_model.predict(frame, conf=YOLO_CONF, iou=0.25, verbose=False)[0]
        candidatos = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx_px = (x1 + x2) / 2
            cy_px = (y1 + y2) / 2

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
                "bbox":    (x1, y1, x2, y2),
                "cx_px":   cx_px,
                "cy_px":   cy_px,
                "cx_norm": cx_px / w,
                "cy_norm": cy_px / h,
                "conf":    conf,
                "color":   COLORES[cls % len(COLORES)],
            })

        cerca = False
        for c in candidatos:
            dist_norm = np.sqrt(
                (c["cx_norm"] - objetivo_cx_norm) ** 2 +
                (c["cy_norm"] - objetivo_cy_norm) ** 2
            )
            if dist_norm <= VERIFY_DIST_THRESHOLD:
                cerca = True
                break

        preview = frame.copy()

        obj_px = (int(objetivo_cx_norm * w), int(objetivo_cy_norm * h))
        cv2.drawMarker(preview, obj_px, (0, 0, 255),
                       cv2.MARKER_CROSS, 30, 2)
        cv2.circle(preview, obj_px,
                   int(VERIFY_DIST_THRESHOLD * w), (0, 0, 200), 1)
        cv2.putText(preview, "POSICION OBJETIVO",
                    (obj_px[0] + 15, obj_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for c in candidatos:
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            color = (0, 80, 255) if cerca else c["color"]
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

            cv2.circle(preview,
                       (int(c["cx_px"]), int(c["cy_px"])),
                       7, (0, 0, 255), -1)

        if cerca:
            estado = (f"Frame {frame_idx+1}/{VERIFY_CHECK_FRAMES} "
                      f"| TROZO EN POSICION — Agarre FALLIDO")
            cv2.putText(preview, estado, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        else:
            estado = (f"Frame {frame_idx+1}/{VERIFY_CHECK_FRAMES} "
                      f"| Trozo no detectado — verificando...")
            cv2.putText(preview, estado, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

        cv2.imshow(ventana_v_real, preview)
        cv2.waitKey(1)

        if cerca:
            trozo_encontrado = True
            fail_frame = preview.copy()
            cv2.putText(fail_frame, "AGARRE FALLIDO — REINTENTANDO",
                        (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 255), 2)
            cv2.imshow(ventana_v_real, fail_frame)
            cv2.waitKey(1500)
            break

    if not trozo_encontrado and frame_ultimo is not None:
        ok_frame = frame_ultimo.copy()

        h_ok, w_ok = ok_frame.shape[:2]
        obj_px = (int(objetivo_cx_norm * w_ok), int(objetivo_cy_norm * h_ok))

        cv2.drawMarker(ok_frame, obj_px, (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(ok_frame, "AGARRE EXITOSO — Trozo desaparecio",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow(ventana_v_real, ok_frame)
        cv2.waitKey(1200)

    cap.release()
    cv2.destroyWindow(ventana_v_real)
    cv2.waitKey(1)
    print("  Cam1 cerrada tras verificación.")

    exito = not trozo_encontrado
    if exito:
        print("  OK Verificación: Trozo NO detectado -> AGARRE EXITOSO.")
    else:
        print("  X Verificación: Trozo AÚN detectado -> AGARRE FALLIDO.")
    return exito

def fase_entrega(robot, simulate=False,
                 threshold_cm=THRESHOLD_CM,
                 eating_cm=EATING_CM,
                 far_cm=FAR_CM,
                 voz_callback=None,
                 leer_comando=None):
    if simulate:
        print("  [SIM] Fase entrega simulada...")
        for s, msg in [
            (1, "Buscando rostro..."),
            (1, "Aproximando..."),
            (1, "Alarma! dist < 12cm"),
            (1, "Persona comiendo (< 5cm) — esperando 12s..."),
            (1, "Persona alejada > 12cm — liberando."),
        ]:
            print(f"    {msg}")
            time.sleep(s)
        return

    import threading as _threading

    _dist        = [None]
    _fase_label  = ["buscando-rostro"]
    _running     = [True]
    _lock        = _threading.Lock()

    def _hilo_captura():
        cap2 = cv2.VideoCapture(CAMERA_2_INDEX, cv2.CAP_DSHOW)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap2.set(cv2.CAP_PROP_FPS, 30)
        cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap2.isOpened():
            print(f"  [CAM2] ERROR: No se pudo abrir cámara {CAMERA_2_INDEX}.", flush=True)
            return

        frames_ok = 0
        for _ in range(40):
            ret, f = cap2.read()
            if ret and f is not None:
                frames_ok += 1
            if frames_ok >= 20:
                break
        print(f"  [CAM2] Hilo iniciado — warmup: {frames_ok} frames OK.", flush=True)

        face_mesh_local = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        while _running[0]:
            ret, frame = cap2.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh_local.process(rgb)

            dist = None
            if res.multi_face_landmarks:
                lm   = res.multi_face_landmarks[0]
                dist = _estimar_distancia(lm, frame.shape)
                bx, by = _boca_centro(lm, frame.shape)
                cv2.circle(frame, (bx, by), 12, (0, 255, 255), -1)
                cv2.circle(frame, (bx, by), 12, (255, 255, 255), 2)
                h_f, w_f = frame.shape[:2]
                le = lm.landmark[33]
                re = lm.landmark[263]
                cv2.circle(frame, (int(le.x*w_f), int(le.y*h_f)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(re.x*w_f), int(re.y*h_f)), 5, (255, 0, 0), -1)

            with _lock:
                _dist[0] = dist
                fase_actual = _fase_label[0]

            frame_hud = _dibujar_hud_cam2(frame, dist, fase_actual,
                                           threshold_cm, eating_cm, far_cm)
            _publicar_frame_cam2(frame_hud)

        cap2.release()
        face_mesh_local.close()

        for ruta in (CAM2_FRAME_TMP, CAM2_FRAME_TMP_WRITE):
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
            except Exception:
                pass
        print("  [CAM2] Hilo de captura detenido.", flush=True)

    hilo = _threading.Thread(target=_hilo_captura, daemon=True, name="CAM2-Capture")
    hilo.start()

    time.sleep(1.0)
    print(f"  OK Cámara 2 lista (hilo de captura activo).")

    def get_dist(fase):
        with _lock:
            _fase_label[0] = fase
            return _dist[0], None

    print(f"\n  --- FASE 0: Búsqueda de rostro con Cam2 ---")
    rostro_encontrado = False
    for intento in range(CODO_BUSQUEDA_MAX + 1):
        dist_prueba = None
        for _ in range(6):
            dist_prueba, _ = get_dist("buscando-rostro")
            if dist_prueba is not None:
                break
            time.sleep(0.08)
        if dist_prueba is not None:
            print(f"  OK Rostro detectado a {dist_prueba:.1f} cm (intento {intento}).")
            rostro_encontrado = True
            break
        if intento < CODO_BUSQUEDA_MAX:
            print(f"  Sin rostro (intento {intento+1}/{CODO_BUSQUEDA_MAX}) — "
                  f"CODO sube {CODO_BUSQUEDA_PASOS} pasos...")
            robot.mover_eje("codo", CODO_BUSQUEDA_PASOS)
            time.sleep(0.4)
    if not rostro_encontrado:
        print(f"  AVISO: Máximo de búsqueda sin detectar rostro. Continuando...")

    print(f"\n  --- FASE A: Aproximación al rostro ---")
    approach_idx    = 0
    last_beep_time  = 0
    alarm_triggered = False

    while approach_idx < MAX_APPROACH_MOVES:
        dist, _ = get_dist("aproximando")
        if dist is None:
            print(f"  Sin rostro — brazo quieto, esperando cara...")
            time.sleep(0.15)
            continue
        if dist <= threshold_cm:
            alarm_triggered = True
            if time.time() - last_beep_time > 1.5:
                if voz_callback:
                    voz_callback("Listo para comer")
                else:
                    _hablar("Listo para comer")
                last_beep_time = time.time()
                print(f"  [!] ALARMA — dist={dist:.1f}cm <= {threshold_cm}cm — BRAZO DETENIDO")
            print(f"  ^ Subiendo CODO {ALARM_RAISE_PASOS} pasos para alcanzar la boca...")
            robot.mover_eje("codo", ALARM_RAISE_PASOS)
            print(f"  OK Elevación completada. Brazo en posición de entrega.")
            break
        eje, pasos = APPROACH_SEQ[approach_idx % len(APPROACH_SEQ)]
        print(f"  dist={dist:.1f}cm — moviendo {eje} +{pasos} (paso {approach_idx+1})")
        robot.mover_eje(eje, pasos)
        approach_idx += 1
        time.sleep(0.05)

    if approach_idx >= MAX_APPROACH_MOVES and not alarm_triggered:
        print(f"  AVISO: Límite de {MAX_APPROACH_MOVES} movimientos sin alarma.")

    print(f"\n  --- FASE B: Brazo detenido ({threshold_cm} cm) ---")
    eating_started = False
    while not eating_started:
        dist, _ = get_dist("alarma")
        if dist is not None and dist < eating_cm:
            print(f"  OK ¡Persona comiendo! dist={dist:.1f}cm < {eating_cm}cm")
            eating_started = True
        time.sleep(0.05)

    print(f"\n  --- FASE C: Persona comiendo ---")
    eating_start_time = time.time()
    eating_done = False

    while not eating_done:

        if leer_comando and leer_comando() in ("DEVOLVER_Y_HOME", "DETENER"):
            print(f"  STOP Comando de parada recibido — deteniendo alimentación.")
            if voz_callback:
                voz_callback("Parando alimentación")
            else:
                _hablar("Parando alimentación")
            eating_done = True
            break

        dist, _ = get_dist("comiendo")
        elapsed          = time.time() - eating_start_time
        tiempo_restante  = max(0, EATING_WAIT_SECS - elapsed)

        if elapsed < EATING_WAIT_SECS:
            if dist is not None:
                print(f"  Comiendo... {elapsed:.1f}s/{EATING_WAIT_SECS}s "
                      f"(quedan {tiempo_restante:.1f}s) — dist={dist:.1f}cm")
            else:
                print(f"  Comiendo... {elapsed:.1f}s/{EATING_WAIT_SECS}s "
                      f"(quedan {tiempo_restante:.1f}s) — sin rostro")
            time.sleep(0.10)
            continue

        if dist is None or dist > far_cm:
            if dist is None:
                print(f"  OK Rostro perdido — se asume que terminó de comer.")
            else:
                print(f"  OK Persona alejada (dist={dist:.1f}cm > {far_cm}cm) — liberando.")
            if voz_callback:
                voz_callback("Parando alimentación")
            else:
                _hablar("Parando alimentación")
            eating_done = True
        else:
            print(f"  Aún cerca (dist={dist:.1f}cm <= {far_cm}cm) — esperando...")
            time.sleep(0.10)

    _running[0] = False
    hilo.join(timeout=3)
    print("  Cámara 2 cerrada.")

def main(simulate=False, threshold_cm=THRESHOLD_CM,
         eating_cm=EATING_CM, far_cm=FAR_CM):

    print("=" * 68)
    print("  BRAZO ROBÓTICO ASISTENTE — MODO COMPLETAMENTE AUTOMÁTICO")
    print("  Sin teclas. Bucle hasta que no haya más trozos en el plato.")
    print("=" * 68)
    print(f"  Cámara 1 (YOLO + verificación): índice {CAMERA_1_INDEX}")
    print(f"  Cámara 2 (MediaPipe)           : índice {CAMERA_2_INDEX}")
    print(f"  Dist. alarma / parada          : {threshold_cm} cm")
    print(f"  Elevación al alcanzar alarma   : CODO +{ALARM_RAISE_PASOS} pasos")
    print(f"  Dist. persona comiendo         : < {eating_cm} cm")
    print(f"  Espera mínima comiendo         : {EATING_WAIT_SECS} s")
    print(f"  Dist. persona alejada          : > {far_cm} cm")
    print(f"  Serial                         : {SERIAL_PORT} @ {SERIAL_BAUD}")
    print(f"  Simulación                     : {'SÍ' if simulate else 'NO'}")
    print(f"  Malla visible en Cam1          : NO (opera solo internamente)")
    print("=" * 68)

    cuadricula = Cuadricula(CALIB_JSON)
    predictor  = MLPPredictor(MODELO_PT, simulate=simulate)

    yolo_model = None
    if not simulate:
        from ultralytics import YOLOWorld
        print("\n[YOLO] Cargando YOLOWorld...")
        yolo_model = YOLOWorld("yolov8m-world.pt")
        yolo_model.set_classes(YOLO_CLASSES)
        print("[YOLO] Listo.\n")

    ciclo = 0

    with RobotInterface(simulate=simulate) as robot:
        continuar = True
        while continuar:
            ciclo += 1
            print(f"\n{'='*68}")
            print(f"  CICLO {ciclo}")
            print(f"{'='*68}")

            print(f"\n[1] HOME...")
            robot.go_home()

            print(f"\n[2] Gripper CERRADO (posición de captura)...")
            robot.set_gripper(PINZA_MIN, "captura")
            time.sleep(0.4)

            print(f"\n[3] Detección automática de trozo objetivo...")
            resultado_cap = captura_automatica(cuadricula, yolo_model, simulate=simulate)
            objetivo, cap_cam1_abierta, ventana_cam1_abierta = resultado_cap

            if objetivo == "SALIR":
                print("\n  Salida manual del usuario (ESC/Q). FIN del programa.")
                continuar = False
                break
            if objetivo is None:
                print("\n  WARN: captura devolvió None inesperado. Reintentando...")
                continue

            obj_cx = objetivo["cx_norm"]
            obj_cy = objetivo["cy_norm"]

            agarre_exitoso = False
            intento_agarre = 0

            while not agarre_exitoso:
                intento_agarre += 1
                print(f"\n{'-'*50}")
                print(f"  INTENTO DE AGARRE #{intento_agarre}")
                print(f"{'-'*50}")

                ci   = objetivo["celda_info"]
                pred = predictor.predecir(obj_cx, obj_cy)

                print(f"\n[4] Predicción MLP (intento {intento_agarre}):")
                print(f"    Entrada  : cx={obj_cx:.4f}  cy={obj_cy:.4f}")
                print(f"    Celda    : {ci['celda']} (Fila {ci['fila']}, Col {ci['columna']})")
                print(f"    Base     : {pred.get('base',0):+d} pasos")
                print(f"    Codo     : {pred.get('codo',0):+d} pasos")
                print(f"    Hombro   : {pred.get('hombro',0):+d} pasos")

                print(f"\n[5] Gripper ABIERTO (pre-agarre)...")
                _hablar("Iniciando agarre")
                robot.set_gripper(PINZA_MAX, "pre-agarre")
                time.sleep(0.5)

                print(f"\n[6] Ejecutando movimientos MLP (Base -> Codo -> Hombro)...")
                robot.ejecutar_prediccion(pred)

                print(f"\n[7] Gripper CERRADO — AGARRE con gripper...")
                robot.set_gripper(PINZA_MIN, "agarrando")
                time.sleep(0.8)

                print(f"\n[8] HOME (Hombro->Codo->Base) para verificar agarre...")
                robot.go_home()
                time.sleep(0.3)

                print(f"\n[9] Verificando agarre con Cam1 "
                      f"(posición objetivo: {obj_cx:.3f}, {obj_cy:.3f})...")
                agarre_exitoso = verificar_agarre_cam1(
                    cuadricula, yolo_model,
                    obj_cx, obj_cy,
                    cap_cam1=cap_cam1_abierta,
                    ventana_cam1=ventana_cam1_abierta,
                    simulate=simulate,
                )
                cap_cam1_abierta    = None
                ventana_cam1_abierta = None

                if not agarre_exitoso:
                    print(f"\n  AVISO Agarre fallido — volviendo a HOME y reintentando...")
                    robot.set_gripper(PINZA_MAX, "soltando-reintento")
                    time.sleep(0.5)
                    robot.set_gripper(PINZA_MIN, "listo-reintento")
                    time.sleep(0.3)
                    robot.go_home()
                    time.sleep(0.3)

            print(f"\n  OK Agarre EXITOSO confirmado (intento #{intento_agarre}). "
                  f"Procediendo a entrega...")

            print(f"\n[10] HOME final con agarre confirmado (H=400, C=400)...")
            robot.go_home()
            time.sleep(0.3)

            print(f"\n[11] Entrega — Cam2 + MediaPipe + Búsqueda rostro + Aproximación...")
            _hablar("Llevando alimento")
            fase_entrega(
                robot,
                simulate=simulate,
                threshold_cm=threshold_cm,
                eating_cm=eating_cm,
                far_cm=far_cm,
            )

            print(f"\n[12] HOME final (después de entrega)...")
            robot.go_home()

            print(f"\n[13] Gripper ABIERTO (soltando)...")
            robot.set_gripper(PINZA_MAX, "soltando")
            time.sleep(0.8)

            print(f"\n[14] Gripper CERRADO (listo para siguiente ciclo)...")
            robot.set_gripper(PINZA_MIN, "listo-siguiente")
            time.sleep(0.4)

            print(f"\n  OK Ciclo {ciclo} completado. Iniciando siguiente ciclo...\n")
            time.sleep(0.5)

    cv2.destroyAllWindows()
    ciclos_exitosos = ciclo - (1 if not continuar and ciclo > 0 else 0)
    print(f"\n{'='*68}")
    print(f"  Programa terminado. Trozos entregados: ~{ciclos_exitosos}")
    print(f"{'='*68}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brazo robótico asistente — bucle automático completo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Simulación sin hardware real.",
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
    MODELO_PT      = args.modelo
    CALIB_JSON     = args.calib
    CAMERA_1_INDEX = args.cam1
    CAMERA_2_INDEX = args.cam2

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