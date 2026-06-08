import argparse
import time
import sys
import threading
import json
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp

SERIAL_PORT  = "COM3"
SERIAL_BAUD  = 115200
TIMEOUT_SEG  = 60

HOME_POSITION = {"base": 0, "hombro": 400, "codo": 400, "muneca": 0, "rotacion": 0}

HOME2_ABSOLUTO = {"base": 950, "hombro": 1650, "codo": 2100, "muneca": 0}

AXIS_CMD = {
    "base":     "BASE",
    "hombro":   "HOMBRO",
    "codo":     "CODO",
    "muneca":   "GRIPPER",
    "rotacion": "GIRO",
}

LIMITE_CODO_BOCA = 3000
LIMITE_CODO_MAX  = 4000

ARCHIVO_CALIB_SOPA = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_FINAL\calibracion.pkl"
MODELO_YOLO_SEG    = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_FINAL\yolov8n-seg.pt"
UMBRAL_VACIO_PORC  = 85.0
MAX_CUCHARADAS     = 25

CAMARA_COMPARTIDA_INDEX  = 2
CAMARA_MEDIAPIPE_INDEX   = 1

ARUCO_ID_BUSCADO = 1
ARUCO_DICT       = cv2.aruco.DICT_4X4_50
ARUCO_ESPERA_SEG = 3.0
ARUCO_REINTENTOS = 2

# --- Búsqueda de rostro (Fase 0): igual que modo sólido, 20 intentos ---
CODO_BUSQUEDA_PASOS    = 500      # codo +500 en cada intento
GRIPPER_BUSQUEDA_PASOS = -100     # gripper (muneca) -100 en cada intento
CODO_BUSQUEDA_MAX      = 20       # hasta 20 veces, como modo sólido

# --- Aproximación al rostro (Fase A): la secuencia se CICLA hasta 30 movimientos ---
APPROACH_SEQ = [
    ("hombro", +400),
    ("codo",   +800),
    ("muneca", -100),             # muneca = GRIPPER
]
MAX_APPROACH_MOVES = 30           # hasta 30 movimientos, como modo sólido

THRESHOLD_CM = 20

# --- Al llegar a 20 cm: codo +700 y gripper -100 ---
ALARM_CODO_PASOS    = 700
ALARM_GRIPPER_PASOS = -100

EATING_CM = 13

BOCA_CONFIRMACION_SEG = 1.5

EATING_WAIT_SECS = 6.0

FAR_CM = 25

TIMEOUT_ENTREGA_SEG = 90

ESTADO_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estado.json")

def leer_estado():
    for _ in range(3):
        try:
            with open(ESTADO_JSON, "r", encoding="utf-8") as f:
                contenido = f.read()
            if contenido.strip():
                return json.loads(contenido)
        except Exception:
            time.sleep(0.005)
    return {}

def debe_detener():
    return leer_estado().get("comando", "") in ("DETENER", "DEVOLVER_Y_HOME")

def escribir_voz(texto):
    try:
        estado = leer_estado()
        if not estado:
            estado = {}
        estado["voz"] = texto
        with open(ESTADO_JSON, "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=2)
        print(f"[VOZ] >>> {texto}", flush=True)
    except Exception as e:
        print(f"[ERROR VOZ] {e}")

def escribir_estado_campo(campo, valor):
    try:
        estado = leer_estado()
        if not estado:
            estado = {}
        estado[campo] = valor
        with open(ESTADO_JSON, "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Estado: {e}")

try:
    import serial
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False
    print("[ADVERTENCIA] pyserial no instalado. Modo simulación.")

class RobotInterface:
    def __init__(self, simulate=False):
        self.simulate = simulate

        self._pos  = dict(HOME_POSITION)
        self._ser  = None
        self.lock  = threading.Lock()

    def conectar(self):
        if self.simulate or not SERIAL_OK:
            print("[Robot] Modo SIMULADO")
            return True
        try:
            print(f"[Robot] Conectando {SERIAL_PORT} @ {SERIAL_BAUD}...")
            self._ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=5)
            time.sleep(2)
            self._ser.flushInput()
            deadline = time.time() + 15
            while time.time() < deadline:
                line = self._ser.readline().decode(errors="ignore").strip()
                if line == "READY":
                    print("[Robot] Arduino listo.")
                    return True
            print("[Robot] No se recibió READY, continuando...")
            return True
        except Exception as e:
            print(f"[Robot] Error: {e} -> simulación")
            self.simulate = True
            return True

    def _send_and_wait(self, cmd, timeout=TIMEOUT_SEG):
        with self.lock:
            if self.simulate:
                print(f"    [SIM] {cmd}")
                time.sleep(0.2)
                return "OK"
            self._ser.flushInput()
            self._ser.write((cmd + "\n").encode())
            start = time.time()
            while time.time() - start < timeout:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode(errors="ignore").strip()
                    if line == "OK":
                        return "OK"
                    if line == "ERR":
                        return "ERR"
            print(f"    [TIMEOUT] {cmd}")
            return "TIMEOUT"

    def mover_relativo(self, eje, pasos, limite_codo=LIMITE_CODO_MAX):
        if pasos == 0:
            return True
        cur = self._pos[eje]
        new = cur + pasos
        if eje == "codo":
            tope = min(limite_codo, LIMITE_CODO_MAX)
            if new > tope:
                pasos = tope - cur
                if pasos <= 0:
                    print(f"    [LÍMITE] codo ya en {cur}, no se mueve más.")
                    return True
                new = cur + pasos
                print(f"    [LÍMITE] codo truncado -> {new} (tope={tope})")
        print(f"    {eje:10s}: {cur:+5d} -> {new:+5d}  (d{pasos:+d})")
        resp = self._send_and_wait(f"{AXIS_CMD[eje]} {pasos}")
        if resp == "OK":
            self._pos[eje] = new
            return True
        return False

    def mover_a_absoluto(self, eje, objetivo, limite_codo=LIMITE_CODO_MAX):
        delta = objetivo - self._pos[eje]
        if delta == 0:
            print(f"    {eje:10s}: ya en {objetivo} (sin movimiento)")
            return True
        return self.mover_relativo(eje, delta, limite_codo=limite_codo)

    def ir_a_home2(self):
        print("\n[Robot] -- Volviendo a HOME2 ----------------------------")
        print(f"  Posición real actual: {self._pos}")
        print(f"  Objetivo HOME2:       {HOME2_ABSOLUTO}")

        self.mover_a_absoluto("muneca", HOME2_ABSOLUTO["muneca"])

        self.mover_a_absoluto("codo", HOME2_ABSOLUTO["codo"])

        self.mover_a_absoluto("hombro", HOME2_ABSOLUTO["hombro"])

        self.mover_a_absoluto("base", HOME2_ABSOLUTO["base"])

        print(f"[Robot] HOME2 alcanzado. Posición: {self._pos}\n")

    def ir_home(self):
        print("\n[Robot] Yendo a HOME...")

        self.mover_a_absoluto("muneca",   HOME_POSITION["muneca"])
        self.mover_a_absoluto("codo",     HOME_POSITION["codo"])
        self.mover_a_absoluto("hombro",   HOME_POSITION["hombro"])
        self.mover_a_absoluto("rotacion", HOME_POSITION["rotacion"])
        self.mover_a_absoluto("base",     HOME_POSITION["base"])
        print("[Robot] HOME alcanzado.\n")

    def set_gripper(self, angulo, label=""):
        angulo = int(np.clip(angulo, 0, 90))
        print(f"  Gripper -> {angulo}° {label}")
        resp = self._send_and_wait(f"PINZA {angulo}")
        return resp

    def desconectar(self):

        if getattr(self, '_ser_prestado', False):
            print("[Robot] Puerto serial prestado — no se cierra aqui.")
            return
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[Robot] Desconectado.")

class CamaraCompartida:
    def __init__(self, index=CAMARA_COMPARTIDA_INDEX):
        self.index = index
        self.cap   = None

    def abrir(self, ancho=1280, alto=720, warmup=5):
        self.cerrar()
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        if not self.cap.isOpened():
            print(f"[CamaraCompartida] ERROR al abrir índice {self.index}")
            return False
        for _ in range(warmup):
            self.cap.read()
        print(f"[CamaraCompartida] Cámara {self.index} abierta.")
        return True

    def leer(self):
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None

    def cerrar(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()
            print(f"[CamaraCompartida] Cámara {self.index} cerrada.")

class ArucoDetector:
    def __init__(self, camara: CamaraCompartida):
        self.camara = camara
        aruco_dict  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        params      = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate        = 0.01
        params.maxMarkerPerimeterRate        = 4.0
        params.polygonalApproxAccuracyRate   = 0.06
        params.adaptiveThreshWinSizeMin      = 3
        params.adaptiveThreshWinSizeMax      = 23
        params.adaptiveThreshWinSizeStep     = 10
        params.adaptiveThreshConstant        = 7
        params.perspectiveRemovePixelPerCell = 8
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def _detectar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None and ARUCO_ID_BUSCADO in ids:
            idx = list(ids.flatten()).index(ARUCO_ID_BUSCADO)
            return True, corners[idx]
        return False, None

    def verificar_presencia(self, espera_seg=2.0):
        vent = "ArUco — Presencia"
        cv2.namedWindow(vent, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(vent, 960, 540)
        inicio    = time.time()
        detectado = False
        print(f"[ArUco] Buscando ID {ARUCO_ID_BUSCADO} durante {espera_seg}s...")
        while time.time() - inicio < espera_seg:
            ret, frame = self.camara.leer()
            if not ret:
                continue
            ok, corner = self._detectar(frame)
            if ok:
                detectado = True
                cv2.aruco.drawDetectedMarkers(frame, [corner],
                                              np.array([[ARUCO_ID_BUSCADO]]))
                cv2.putText(frame, f"ID {ARUCO_ID_BUSCADO} DETECTADO", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(vent, frame); cv2.waitKey(500)
                break
            else:
                cv2.putText(frame, "Buscando marcador...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(vent, frame); cv2.waitKey(30)
            time.sleep(0.05)
        cv2.destroyWindow(vent)
        print(f"[ArUco] {'OK Detectado' if detectado else 'X No detectado'}")
        return detectado

    def verificar_ausencia(self, espera_seg=3.0):
        vent = "ArUco — Ausencia"
        cv2.namedWindow(vent, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(vent, 960, 540)
        inicio         = time.time()
        desaparecio    = False
        t_sin_detectar = None
        print(f"[ArUco] Verificando ausencia de ID {ARUCO_ID_BUSCADO}...")
        while time.time() - inicio < espera_seg:
            ret, frame = self.camara.leer()
            if not ret:
                continue
            ok, corner = self._detectar(frame)
            if ok:
                t_sin_detectar = None
                cv2.aruco.drawDetectedMarkers(frame, [corner],
                                              np.array([[ARUCO_ID_BUSCADO]]))
                cv2.putText(frame, "AUN VISIBLE — agarre fallido", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            else:
                if t_sin_detectar is None:
                    t_sin_detectar = time.time()
                elif time.time() - t_sin_detectar >= 1.0:
                    desaparecio = True
                    cv2.putText(frame, "AUSENTE — agarre OK", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.imshow(vent, frame); cv2.waitKey(400)
                    break
                cv2.putText(frame, "Marcador no visible...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
            cv2.imshow(vent, frame); cv2.waitKey(30)
            time.sleep(0.05)
        cv2.destroyWindow(vent)
        print(f"[ArUco] {'OK Ausencia confirmada' if desaparecio else 'X Sigue visible'}")
        return desaparecio

class SopaNivelDetector:
    def __init__(self, calib_path, camara: CamaraCompartida):
        self.calib_path = Path(calib_path)
        if not self.calib_path.exists():
            raise FileNotFoundError(f"No existe calibración: {calib_path}")
        with open(self.calib_path, "rb") as f:
            datos = pickle.load(f)
        self.ref_plato  = datos["referencia_plato"]
        self.ref_sombra = datos.get("referencia_sombra", None)
        self.tol        = datos["tolerancias"]
        self.anti       = datos["anti_sombra"]
        self.camara     = camara
        self.model      = YOLO(MODELO_YOLO_SEG)
        print("[SopaDetector] Inicializado.")

    def _segmentar_plato(self, frame):
        fh, fw  = frame.shape[:2]
        mascara = np.zeros((fh, fw), dtype=np.uint8)
        results = self.model(frame, verbose=False, device="cpu")[0]
        if results.masks is not None:
            for i in range(len(results.masks.data)):
                cls_id = int(results.boxes.cls[i])
                nombre = self.model.names.get(cls_id, "").lower()
                if nombre in {"bowl", "cup", "plate", "dish"}:
                    mask_raw   = results.masks.data[i].cpu().numpy()
                    mask_frame = cv2.resize(mask_raw, (fw, fh))
                    mascara    = cv2.bitwise_or(mascara,
                                               (mask_frame * 255).astype(np.uint8))
        if np.any(mascara):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            return cv2.erode(mascara, k, iterations=1)

        gris     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris     = cv2.GaussianBlur(gris, (11, 11), 0)
        circulos = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT, 1.2, fh // 3,
                                    param1=80, param2=40,
                                    minRadius=80, maxRadius=400)
        if circulos is not None:
            cx, cy, r = circulos[0][0]
            cv2.circle(mascara, (int(cx), int(cy)), max(0, int(r) - 12), 255, -1)
        return mascara

    def medir_nivel(self, mostrar_ventana=True):
        ret, frame = self.camara.leer()
        if not ret:
            print("[SopaDetector] Sin frame.")
            return 0.0
        mascara = self._segmentar_plato(frame)
        if not np.any(mascara):
            return 0.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        dh = np.minimum(np.abs(h - self.ref_plato["H_mean"]),
                        180 - np.abs(h - self.ref_plato["H_mean"]))
        ds = np.abs(s - self.ref_plato["S_mean"])
        dv = np.abs(v - self.ref_plato["V_mean"])
        es_diferente = (dh > self.tol["H"]) | (ds > self.tol["S"]) | (dv > self.tol["V"])
        es_sombra    = ((s < self.anti["S_max"]) & (v < self.anti["V_max"])
                        if self.ref_sombra is not None
                        else np.zeros_like(es_diferente))
        sopa_bool = (mascara > 0) & es_diferente & (~es_sombra)
        pix_tot   = int(np.sum(mascara > 0))
        if pix_tot == 0:
            return 0.0
        porcentaje = 100.0 * np.sum(sopa_bool) / pix_tot
        if mostrar_ventana:
            display = frame.copy()
            overlay = np.zeros_like(display)
            overlay[sopa_bool]                    = (0, 0, 255)
            overlay[(mascara > 0) & (~sopa_bool)] = (0, 200, 0)
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
            cv2.putText(display, f"Sopa: {porcentaje:.1f}%", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("Nivel Sopa", display)
            cv2.waitKey(1500)
            cv2.destroyWindow("Nivel Sopa")
        return porcentaje

def _estimar_distancia(landmarks, shape):
    h, w      = shape[:2]
    le        = landmarks.landmark[33]
    re        = landmarks.landmark[263]
    eye_dist  = np.hypot((re.x - le.x) * w, (re.y - le.y) * h)
    nose      = landmarks.landmark[1]
    chin      = landmarks.landmark[152]
    nose_chin = np.hypot((chin.x - nose.x) * w, (chin.y - nose.y) * h)
    if eye_dist > 30:
        dist = 5000.0 / eye_dist
    elif nose_chin > 0:
        dist = 2500.0 / nose_chin
    else:
        dist = 50.0
    return float(np.clip(dist, 3.0, 65.0))

def _boca_centro(landmarks, shape):
    h, w = shape[:2]
    ul   = landmarks.landmark[13]
    ll   = landmarks.landmark[14]
    return int((ul.x + ll.x) / 2 * w), int((ul.y + ll.y) / 2 * h)

def _dibujar_hud(frame, dist, fase):
    color_fase = {
        "busqueda":    (0, 255, 255),
        "aproximando": (255, 200, 0),
        "alarma":      (0, 0, 255),
        "comiendo":    (0, 255, 0),
        "alejando":    (200, 200, 200),
    }.get(fase, (255, 255, 255))
    texto_dist = f"Dist: {dist:.1f} cm" if dist is not None else "Sin rostro"
    cv2.putText(frame, f"[{fase.upper()}] {texto_dist}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_fase, 2)
    return frame

def fase_entrega_sopa(robot: RobotInterface):
    print("\n[EntregaSopa] == Iniciando fase de entrega adaptativa ==")

    cap = cv2.VideoCapture(CAMARA_MEDIAPIPE_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[EntregaSopa] ERROR: no se pudo abrir cámara MediaPipe (índice 0).")

        return True

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    vent = "Entrega Sopa — Cara"
    cv2.namedWindow(vent, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(vent, 640, 480)

    def leer_distancia_y_frame():
        ret, frame = cap.read()
        if not ret:
            return None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm   = res.multi_face_landmarks[0]
            dist = _estimar_distancia(lm, frame.shape)
            bx, by = _boca_centro(lm, frame.shape)
            cv2.circle(frame, (bx, by), 12, (0, 255, 255), -1)
            return dist, frame
        return None, frame

    print(f"\n  +- Fase 0: Búsqueda de cara (subir codo hasta {CODO_BUSQUEDA_MAX} veces)")
    rostro_encontrado = False
    for intento in range(CODO_BUSQUEDA_MAX + 1):
        if debe_detener():
            cap.release(); face_mesh.close(); cv2.destroyWindow(vent)
            return False

        dist, frame = leer_distancia_y_frame()
        if frame is not None:
            _dibujar_hud(frame, dist, "busqueda")
            cv2.imshow(vent, frame); cv2.waitKey(1)

        if dist is not None:
            print(f"  +- OK Cara encontrada a {dist:.1f} cm (intento {intento})")
            rostro_encontrado = True
            break

        if intento < CODO_BUSQUEDA_MAX:
            print(f"  |  Sin cara (intento {intento+1}/{CODO_BUSQUEDA_MAX}) "
                  f"-> CODO +{CODO_BUSQUEDA_PASOS}, GRIPPER {GRIPPER_BUSQUEDA_PASOS}")
            robot.mover_relativo("codo", CODO_BUSQUEDA_PASOS,
                                 limite_codo=LIMITE_CODO_BOCA)
            robot.mover_relativo("muneca", GRIPPER_BUSQUEDA_PASOS)
            time.sleep(0.3)

    if not rostro_encontrado:
        print("  +- AVISO  No se detectó cara. Continuando de todas formas...")

    print(f"\n  +- Fase A: Aproximación (parar si dist <= {THRESHOLD_CM} cm, "
          f"máx {MAX_APPROACH_MOVES} movimientos)")
    alarm_triggered = False
    approach_idx    = 0

    while approach_idx < MAX_APPROACH_MOVES:
        if debe_detener():
            cap.release(); face_mesh.close(); cv2.destroyWindow(vent)
            return False

        dist, frame = leer_distancia_y_frame()
        if frame is not None:
            _dibujar_hud(frame, dist, "aproximando")
            cv2.imshow(vent, frame); cv2.waitKey(1)

        if dist is None:
            print("  |  Sin cara — brazo quieto, esperando rostro...")
            time.sleep(0.15)
            continue

        if dist <= THRESHOLD_CM:
            print(f"  |  [!] Cara a {dist:.1f} cm <= {THRESHOLD_CM} cm "
                  f"-> CODO +{ALARM_CODO_PASOS}, GRIPPER {ALARM_GRIPPER_PASOS}")
            robot.mover_relativo("codo", ALARM_CODO_PASOS,
                                 limite_codo=LIMITE_CODO_BOCA)
            robot.mover_relativo("muneca", ALARM_GRIPPER_PASOS)
            alarm_triggered = True
            escribir_voz("Listo para comer")
            break

        eje, pasos = APPROACH_SEQ[approach_idx % len(APPROACH_SEQ)]
        print(f"  |  dist={dist:.1f} cm — moviendo {eje} {pasos:+d} "
              f"(mov {approach_idx+1}/{MAX_APPROACH_MOVES})")
        if eje == "codo":
            robot.mover_relativo(eje, pasos, limite_codo=LIMITE_CODO_BOCA)
        else:
            robot.mover_relativo(eje, pasos)
        approach_idx += 1
        time.sleep(0.15)

    if approach_idx >= MAX_APPROACH_MOVES and not alarm_triggered:
        print(f"  +- AVISO: Límite de {MAX_APPROACH_MOVES} movimientos "
              f"sin llegar a {THRESHOLD_CM} cm.")

    if not alarm_triggered:
        escribir_voz("Listo para comer")

    print(f"  +- Fase A completada. Posición: {robot._pos}")

    print(f"\n  +- Fase B: Brazo quieto — esperando boca < {EATING_CM} cm o cara ausente")
    t_inicio_b  = time.time()
    comiendo    = False
    t_sin_cara  = None

    while time.time() - t_inicio_b < TIMEOUT_ENTREGA_SEG:
        if debe_detener():
            cap.release(); face_mesh.close(); cv2.destroyWindow(vent)
            return False

        dist, frame = leer_distancia_y_frame()
        if frame is not None:
            _dibujar_hud(frame, dist, "alarma")
            cv2.imshow(vent, frame); cv2.waitKey(1)

        if dist is None:

            if t_sin_cara is None:
                t_sin_cara = time.time()
            elif time.time() - t_sin_cara >= 1.5:

                print("  +- OK Cara ausente 1.5 s -> persona acercó la boca")
                comiendo = True
                break
        else:
            t_sin_cara = None
            if dist < EATING_CM:
                print(f"  +- OK Boca muy cerca ({dist:.1f} cm < {EATING_CM} cm) -> comiendo")
                comiendo = True
                break

        time.sleep(0.05)

    if not comiendo:
        print("  +- [T] Timeout esperando boca — continuando de todas formas.")

    print(f"\n  +- Fase C: Comiendo — delay {EATING_WAIT_SECS}s + verificar alejamiento")
    t_comiendo     = time.time()
    t_cara_ausente = None

    while True:
        if debe_detener():
            cap.release(); face_mesh.close(); cv2.destroyWindow(vent)
            return False

        elapsed = time.time() - t_comiendo
        dist, frame = leer_distancia_y_frame()
        fase_hud    = "comiendo" if elapsed < EATING_WAIT_SECS else "alejando"
        if frame is not None:
            _dibujar_hud(frame, dist, fase_hud)
            cv2.putText(frame,
                        f"t={elapsed:.1f}/{EATING_WAIT_SECS}s",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(vent, frame); cv2.waitKey(1)

        if elapsed < EATING_WAIT_SECS:
            time.sleep(0.1)
            continue

        if dist is None:
            if t_cara_ausente is None:
                t_cara_ausente = time.time()
                print("  |  Cara perdida — esperando 3 s para confirmar alejamiento...")
            elif time.time() - t_cara_ausente >= 3.0:
                print("  +- OK Cara ausente > 3 s -> se asume terminó de comer.")
                break
        else:
            t_cara_ausente = None
            if dist > FAR_CM:
                print(f"  +- OK Persona alejada ({dist:.1f} cm > {FAR_CM} cm) -> cucharada completa.")
                break
            else:
                print(f"  |  Aún cerca ({dist:.1f} cm <= {FAR_CM} cm) — esperando...")

        if time.time() - t_comiendo > TIMEOUT_ENTREGA_SEG:
            print("  +- [T] Timeout global de entrega — continuando.")
            break

        time.sleep(0.1)

    cap.release()
    face_mesh.close()
    cv2.destroyWindow(vent)
    print(f"[EntregaSopa] Fase completada. Posición real acumulada: {robot._pos}\n")
    return True

def agarrar_cuchara(robot: RobotInterface, aruco: ArucoDetector,
                    camara: CamaraCompartida):
    for intento in range(ARUCO_REINTENTOS + 1):
        if intento > 0:
            print(f"\n[ArUco] Reintento {intento}/{ARUCO_REINTENTOS}")
            escribir_voz(f"Reintentando agarre, intento {intento}")

        robot.set_gripper(90, "abrir pinza")
        time.sleep(0.4)

        robot.mover_relativo("base",   -195)
        robot.mover_relativo("codo",   1430)
        robot.mover_relativo("hombro", 1200)

        print("\n[3.5] Verificando presencia de la cuchara (ArUco)...")
        if not aruco.verificar_presencia(espera_seg=2.0):
            print("[ArUco] X Cuchara no encontrada. Retrocediendo...")
            escribir_voz("No se encontró la cuchara")
            robot.mover_relativo("hombro", -1200)
            robot.mover_relativo("codo",   -1430)
            robot.mover_relativo("base",    195)
            time.sleep(3.0)
            continue

        robot.mover_relativo("hombro", 1340)

        robot.set_gripper(0, "cerrar pinza")
        time.sleep(0.6)

        robot.mover_relativo("hombro", -1200)
        robot.mover_relativo("base",    400)

        print("\n[5] Verificando agarre (ArUco debe desaparecer)...")
        time.sleep(0.5)
        if aruco.verificar_ausencia(espera_seg=ARUCO_ESPERA_SEG):
            print("[ArUco] OK Agarre exitoso.")
            escribir_voz("Cuchara agarrada correctamente")
            return True
        else:
            print("[ArUco] X Agarre fallido — marcador sigue visible.")
            robot.set_gripper(90, "soltar cuchara")
            time.sleep(0.5)
            robot.ir_home()
            time.sleep(0.5)

    print("[ArUco] X Todos los intentos de agarre fallaron.")
    escribir_voz("No se pudo agarrar la cuchara")
    return False

def ejecutar_una_cucharada(robot: RobotInterface, sopa_detector: SopaNivelDetector,
                            camara: CamaraCompartida, num: int):
    print(f"\n{'='*55}")
    print(f"  Cucharada {num}")
    print(f"{'='*55}")

    print("\n[6] Volviendo a HOME2 (posición absoluta)...")
    robot.ir_a_home2()
    time.sleep(1.0)

    if not camara.abrir():
        print("[ERROR] No se pudo abrir cámara para medir sopa.")
        return False

    nivel = sopa_detector.medir_nivel(mostrar_ventana=True)
    camara.cerrar()
    print(f"[SOPA] Nivel: {nivel:.1f}%")

    if nivel < UMBRAL_VACIO_PORC:
        escribir_voz("La sopa está casi vacía, terminando servicio")
        return False

    if debe_detener():
        return False

    print("\n[7] Posicionando dentro del plato")
    robot.mover_relativo("base",   -550)
    robot.mover_relativo("hombro",  110)
    robot.mover_relativo("codo",   -580)
    robot.mover_relativo("muneca", -1000)

    if debe_detener():
        return False

    print("\n[8] Sumergiendo cuchara")
    robot.mover_relativo("hombro", 350)

    if debe_detener():
        return False

    print("\n[9] Levantar y centrar")
    robot.mover_relativo("codo",   400)
    robot.mover_relativo("hombro", -500)
    robot.mover_relativo("base",   -425)

    if debe_detener():
        return False

    print("\n[10] Fase de entrega adaptativa")
    if not fase_entrega_sopa(robot):
        return False

    return True

def bucle_sopa(robot: RobotInterface = None, ya_en_home: bool = False):
    robot_propio = (robot is None)
    if robot_propio:
        robot = RobotInterface(simulate=False)
        robot.conectar()
        if ya_en_home:
            robot._pos = dict(HOME_POSITION)
            print(f"[SOPA] Brazo ya en HOME. _pos={robot._pos}", flush=True)
        else:
            robot.ir_home()

    print("\n" + "="*60)
    print("  SERVICIO DE SOPA — Múltiples cucharadas")
    print("="*60)

    if not robot_propio:
        for eje, val in HOME_POSITION.items():
            robot._pos[eje] = val
        print(f"[bucle_sopa] _pos sincronizado con HOME: {robot._pos}", flush=True)

    camara = CamaraCompartida(index=CAMARA_COMPARTIDA_INDEX)

    print("[SOPA] Esperando que la interfaz ceda la cámara (cam1_owner=cuchara)...",
          flush=True)
    _t_espera_cam = time.time()
    while time.time() - _t_espera_cam < 10.0:
        _est = leer_estado()
        if _est.get("cam1_owner", "") == "cuchara":
            break
        time.sleep(0.2)

    time.sleep(1.0)
    print("[SOPA] Cámara disponible. Abriendo...", flush=True)

    if not camara.abrir():
        print("[ERROR] No se pudo abrir camara para agarre.")
        robot.ir_home()
        return

    aruco = ArucoDetector(camara)
    exito = agarrar_cuchara(robot, aruco, camara)
    camara.cerrar()

    if not exito:
        robot.ir_home()
        return

    escribir_voz("Iniciando servicio de sopa")

    try:
        sopa_detector = SopaNivelDetector(ARCHIVO_CALIB_SOPA, camara)
    except Exception as e:
        print(f"[ERROR] Calibración: {e}")
        escribir_voz("Error de calibración de sopa")
        robot.ir_home()
        return

    cucharadas = 0
    while cucharadas < MAX_CUCHARADAS:
        if debe_detener():
            escribir_voz("Parando servicio de sopa")
            break

        cucharadas += 1
        print(f"\n[R] Cucharada {cucharadas}/{MAX_CUCHARADAS}")

        if not ejecutar_una_cucharada(robot, sopa_detector, camara, cucharadas):
            break

        time.sleep(1.0)

        if cucharadas % 3 == 0:
            if camara.abrir():
                nivel = sopa_detector.medir_nivel(mostrar_ventana=True)
                camara.cerrar()
                print(f"[SOPA] Control periódico: {nivel:.1f}%")
                if nivel < UMBRAL_VACIO_PORC:
                    escribir_voz("La sopa está casi vacía")
                    break

        if cucharadas >= 20:
            escribir_voz("¿Quieres seguir comiendo sopa?")
            for _ in range(30):
                if debe_detener():
                    break
                time.sleep(0.5)
            if debe_detener():
                escribir_voz("Terminando servicio")
                break

    escribir_voz("Terminando servicio de sopa")

    robot.ir_home()
    robot.set_gripper(90, "soltar cuchara")
    time.sleep(0.8)
    robot.set_gripper(0, "pinza cerrada")

    camara.cerrar()

    if robot_propio:
        robot.desconectar()

    print(f"\nOK Servicio completado. Cucharadas servidas: {cucharadas}")

def main():
    parser = argparse.ArgumentParser(
        description="Brazo de sopa automático con cuchara v4"
    )
    parser.add_argument("--sim",    action="store_true",
                        help="Modo simulación (sin Arduino)")
    parser.add_argument("--puerto", default="COM3",
                        help="Puerto serial del Arduino")
    args = parser.parse_args()

    global SERIAL_PORT
    SERIAL_PORT = args.puerto

    if args.sim:
        robot = RobotInterface(simulate=True)
        robot._pos = dict(HOME_POSITION)
        bucle_sopa(robot)
    else:

        bucle_sopa(None, ya_en_home=True)
    print("\nOK Programa finalizado.")

if __name__ == "__main__":
    main()