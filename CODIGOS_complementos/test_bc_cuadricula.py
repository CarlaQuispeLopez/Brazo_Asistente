"""
test_bc.py  —  Prueba de Behavior Cloning
==========================================
Carga demonstrations_clean.pkl y ejecuta el brazo robótico imitando
los movimientos de la demo más cercana a la celda detectada.

Flujo por ciclo:
  1. HOME (C=400, H=400)
  2. Gripper cierra (PINZA 0)
  3. Cámara abre con malla → usuario presiona ESPACIO para capturar celda
  4. Cámara cierra
  5. Buscar demo más cercana (behavior cloning)
  6. Gripper abre (PINZA 90) — posición previa al agarre
  7. Reproducir pasos de la demo: Base → Codo → Hombro
  8. Gripper cierra (PINZA 0) — agarre
  9. Volver a HOME: Hombro → Codo → Base
 10. Gripper abre (PINZA 90) — soltar trozo
 11. Gripper cierra (PINZA 0) — listo para siguiente captura
 12. Volver al paso 3 (o salir con 'q')
"""

import os
import json
import pickle
import time
import argparse
import numpy as np
import cv2
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────
CLEAN_PKL  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\demos\demonstrations_clean.pkl"
CALIB_JSON = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\calibracion_cuadricula.json"

SERIAL_PORT  = "COM5"
SERIAL_BAUD  = 115200
CAMERA_INDEX = 2

HOME_POSITION = {
    "base":     0,
    "hombro":   400,
    "codo":     400,
    "muneca":   0,
    "rotacion": 0,
}

JOINT_LIMITS = {
    "base":     (-3200, 3200),
    "hombro":   (-3200, 3200),
    "codo":     (-3200, 3200),
    "muneca":   (-3200, 3200),
    "rotacion": (-3200, 3200),
}

AXIS_CMD = {
    "base":     "BASE",
    "hombro":   "HOMBRO",
    "codo":     "CODO",
    "muneca":   "GRIPPER",
    "rotacion": "GIRO",
}

PINZA_MIN = 0    # cerrada
PINZA_MAX = 90   # abierta

GRID_COLS = 12
GRID_ROWS = 9

YOLO_CONF = 0.20
YOLO_CLASSES = [
    "apple", "pear", "peach", "plum", "apricot", "cherry",
    "strawberry", "raspberry", "blueberry", "blackberry",
    "grape", "watermelon", "melon", "cantaloupe", "honeydew",
    "banana", "mango", "papaya", "pineapple chunk", "kiwi",
    "orange slice", "mandarin", "lemon slice", "lime slice",
    "fig", "date", "lychee", "guava", "passion fruit",
    "dragon fruit", "star fruit", "persimmon", "pomegranate seed",
    "tomato", "cherry tomato", "carrot piece", "broccoli floret",
    "cauliflower", "lettuce piece", "cucumber slice", "zucchini",
    "eggplant", "bell pepper", "corn kernel", "pea",
    "green bean", "asparagus", "artichoke", "celery piece",
    "beet", "radish", "turnip", "potato chunk", "sweet potato",
    "mushroom", "onion piece", "leek", "spinach", "kale",
    "cabbage piece", "brussels sprout", "bok choy",
    "chicken piece", "beef piece", "pork piece", "lamb piece",
    "turkey piece", "sausage slice", "meatball", "nugget",
    "shrimp", "fish piece", "salmon chunk", "tuna piece",
    "squid piece", "octopus piece", "crab meat",
    "boiled egg", "fried egg piece", "omelette piece",
    "tofu cube", "tempeh piece",
    "pasta piece", "noodle", "gnocchi", "dumpling",
    "rice ball", "bread piece", "crouton",
    "cheese cube", "mozzarella", "ham piece",
    "olive", "pickle slice", "sun-dried tomato",
    "chickpea", "lentil", "bean",
    "food piece", "fruit piece", "vegetable piece", "meat piece",
]

COLORES = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,255,0),(100,255,100),(255,150,50),(50,200,255),(200,50,255),
]


# ─────────────────────────────────────────────
#  BEHAVIOR CLONING — Dataset
# ─────────────────────────────────────────────
class BCDataset:
    """
    Carga demonstrations_clean.pkl y permite buscar la demo
    más cercana a una celda dada usando distancia Manhattan en la malla.
    """

    def __init__(self, pkl_path):
        print(f"Cargando dataset: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.episodes = data.get("episodes", [])
        print(f"  {len(self.episodes)} demos cargadas.\n")

        # Índice: celda → lista de demos
        self._index = {}
        for ep in self.episodes:
            c = ep.get("celda_objetivo")
            if c is not None:
                self._index.setdefault(c, []).append(ep)

        celdas = sorted(self._index.keys())
        print(f"  Celdas cubiertas ({len(celdas)}): {celdas}\n")

    @staticmethod
    def celda_a_rc(celda, cols=GRID_COLS):
        """Convierte número de celda a (fila, columna) base-0."""
        row = (celda - 1) // cols
        col = (celda - 1) %  cols
        return row, col

    def demo_mas_cercana(self, celda_target):
        """
        Devuelve la demo cuya celda_objetivo es más cercana a celda_target.
        Si hay varias demos para la misma celda, devuelve la primera (exitosa si existe).
        Devuelve (demo_ep, celda_usada, distancia_grid).
        """
        if celda_target in self._index:
            demos = self._index[celda_target]
            # Preferir demos exitosas
            exitosas = [d for d in demos if d.get("success")]
            elegida  = exitosas[0] if exitosas else demos[0]
            return elegida, celda_target, 0

        # Buscar la celda más cercana (distancia Euclidiana en malla)
        tr, tc = self.celda_a_rc(celda_target)
        mejor_dist  = float("inf")
        mejor_celda = None

        for celda in self._index:
            r, c   = self.celda_a_rc(celda)
            dist   = ((r - tr)**2 + (c - tc)**2) ** 0.5
            if dist < mejor_dist:
                mejor_dist  = dist
                mejor_celda = celda

        demos    = self._index[mejor_celda]
        exitosas = [d for d in demos if d.get("success")]
        elegida  = exitosas[0] if exitosas else demos[0]
        return elegida, mejor_celda, mejor_dist


# ─────────────────────────────────────────────
#  CUADRICULA
# ─────────────────────────────────────────────
class Cuadricula:
    COLOR_LINEA   = (0, 255, 255)
    COLOR_NUMERO  = (255, 255, 0)
    COLOR_RELLENO = (0, 60, 60)

    def __init__(self, json_path):
        self.cols  = GRID_COLS
        self.rows  = GRID_ROWS
        self.M     = None
        self.M_inv = None
        self._cargar(json_path)

    def _cargar(self, path):
        try:
            with open(path, "r") as f:
                datos = json.load(f)
            self.cols = datos["grid_cols"]
            self.rows = datos["grid_rows"]
            puntos    = [tuple(p) for p in datos["puntos"]]
            src = np.float32([[0,0],[self.cols,0],[self.cols,self.rows],[0,self.rows]])
            dst = np.float32(puntos)
            self.M     = cv2.getPerspectiveTransform(src, dst)
            self.M_inv = np.linalg.inv(self.M)
            print(f"[Cuadricula] Cargada: {self.cols}x{self.rows} "
                  f"({self.cols*self.rows} celdas)\n")
        except Exception as e:
            print(f"[Cuadricula] ERROR al cargar '{path}': {e}\n")

    @property
    def disponible(self):
        return self.M is not None

    def _t(self, pts):
        return cv2.perspectiveTransform(
            np.float32(pts).reshape(-1, 1, 2), self.M).reshape(-1, 2)

    def info_celda(self, px, py):
        if not self.disponible:
            return {"celda": None, "fila": None, "columna": None}
        gx, gy = cv2.perspectiveTransform(
            np.float32([[[px, py]]]), self.M_inv)[0][0]
        if 0 <= gx <= self.cols and 0 <= gy <= self.rows:
            col = min(int(gx), self.cols - 1)
            row = min(int(gy), self.rows - 1)
            return {"celda": row * self.cols + col + 1,
                    "fila": row + 1, "columna": col + 1}
        return {"celda": None, "fila": None, "columna": None}

    def dibujar(self, frame, highlight=None, celdas_con_demo=None):
        """
        highlight        : celda detectada actualmente (naranja)
        celdas_con_demo  : set de celdas que tienen demos (verde oscuro)
        """
        if not self.disponible:
            return frame
        if celdas_con_demo is None:
            celdas_con_demo = set()

        overlay = frame.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                num  = r * self.cols + c + 1
                corn = [[c,r],[c+1,r],[c+1,r+1],[c,r+1]]
                pts  = self._t(corn).astype(np.int32)
                if num == highlight:
                    color = (0, 80, 180)        # naranja oscuro: objetivo actual
                elif num in celdas_con_demo:
                    color = (0, 80, 0)          # verde oscuro: tiene demo
                else:
                    color = self.COLOR_RELLENO
                cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Resalte sólido celda objetivo
        if highlight and 1 <= highlight <= self.cols * self.rows:
            r = (highlight - 1) // self.cols
            c = (highlight - 1) % self.cols
            pts = self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
            cv2.fillPoly(frame, [pts], (0, 140, 255))   # naranja brillante

        # Celdas con demo: borde verde
        for celda_num in celdas_con_demo:
            if celda_num == highlight:
                continue
            r = (celda_num - 1) // self.cols
            c = (celda_num - 1) % self.cols
            pts = self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 0), 2)

        # Líneas de malla
        for c in range(self.cols + 1):
            p1 = self._t([[c, 0        ]]).astype(int)[0]
            p2 = self._t([[c, self.rows]]).astype(int)[0]
            cv2.line(frame, tuple(p1), tuple(p2), self.COLOR_LINEA,
                     2 if c in (0, self.cols) else 1)
        for r in range(self.rows + 1):
            p1 = self._t([[0,         r]]).astype(int)[0]
            p2 = self._t([[self.cols, r]]).astype(int)[0]
            cv2.line(frame, tuple(p1), tuple(p2), self.COLOR_LINEA,
                     2 if r in (0, self.rows) else 1)

        # Números de celda
        for r in range(self.rows):
            for c in range(self.cols):
                num = r * self.cols + c + 1
                cx  = self._t([[c + 0.5, r + 0.5]])[0].astype(int)
                txt = str(num)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                color = (255,255,255) if num == highlight else self.COLOR_NUMERO
                cv2.putText(frame, txt,
                            (int(cx[0]) - tw//2, int(cx[1]) + th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
        return frame


# ─────────────────────────────────────────────
#  ROBOT INTERFACE
# ─────────────────────────────────────────────
class RobotInterface:
    def __init__(self, simulate=False):
        self.simulate      = simulate
        self._positions    = dict(HOME_POSITION)
        self._angulo_pinza = 0
        self._ser          = None

    def __enter__(self):
        if not self.simulate:
            import serial
            self._ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=10)
            time.sleep(2)
            self._ser.flushInput()
            deadline = time.time() + 20
            while time.time() < deadline:
                line = self._ser.readline().decode(errors="ignore").strip()
                if line == "READY":
                    break
            print(f"Arduino conectado en {SERIAL_PORT} a {SERIAL_BAUD} baud.")
        else:
            print("Robot [SIMULADO] inicializado.")
        return self

    def __exit__(self, *_):
        if self._ser and self._ser.is_open:
            self._ser.close()
        print("Robot desconectado.")

    def _send(self, cmd):
        if self.simulate:
            print(f"    [SIM] -> {cmd}")
            time.sleep(0.1)
            return "OK"
        self._ser.flushInput()
        self._ser.write((cmd + "\n").encode())
        deadline = time.time() + 20
        while time.time() < deadline:
            line = self._ser.readline().decode(errors="ignore").strip()
            if line == "OK":
                return "OK"
            if line == "ERR":
                return "ERR"
        return "TIMEOUT"

    def set_gripper(self, angulo, label=""):
        angulo = int(np.clip(angulo, PINZA_MIN, PINZA_MAX))
        estado = "cerrada" if angulo == 0 else f"abierta {angulo}°"
        print(f"  Gripper → {angulo}° ({estado}){' — '+label if label else ''}")
        resp = self._send(f"PINZA {angulo}")
        if resp == "OK":
            self._angulo_pinza = angulo
        return resp

    def go_home(self):
        """Vuelve a HOME en orden: Hombro → Codo → Base."""
        orden = ["hombro", "codo", "muneca", "rotacion", "base"]
        print("  Regresando a HOME (Hombro → Codo → Base)...")
        for ax in orden:
            target  = HOME_POSITION[ax]
            current = self._positions[ax]
            diff    = target - current
            if diff == 0:
                continue
            print(f"    {ax}: {current:+d} → {target:+d}  ({diff:+d} pasos)")
            self._send(f"{AXIS_CMD[ax]} {diff}")
            self._positions[ax] = target
        print(f"  HOME alcanzado: {HOME_POSITION}")

    def ejecutar_paso(self, step):
        """
        Ejecuta un paso del dataset limpio.
        step = {"cmd": "B-", "steps_executed": 500, ...}
        """
        cmd   = step.get("cmd", "")
        pasos = step.get("steps_executed", 0)

        mapa_eje = {
            "B+": ("base",    +1),
            "B-": ("base",    -1),
            "C+": ("codo",    +1),
            "C-": ("codo",    -1),
            "H+": ("hombro",  +1),
            "H-": ("hombro",  -1),
        }

        if cmd == "GRIP 0":
            return self.set_gripper(0, "agarrar")

        if cmd == "GRIP 90":
            return self.set_gripper(90, "abrir")

        if cmd not in mapa_eje:
            print(f"    [SKIP] cmd desconocido: {cmd}")
            return "SKIP"

        ax, direccion = mapa_eje[cmd]
        pasos_reales  = direccion * pasos

        lo, hi  = JOINT_LIMITS[ax]
        new_pos = self._positions[ax] + pasos_reales
        new_pos = int(np.clip(new_pos, lo, hi))
        diff    = new_pos - self._positions[ax]

        if diff == 0:
            return "OK"

        print(f"    {cmd} {abs(diff):>6} pasos  ({ax}: {self._positions[ax]:+d} → {new_pos:+d})")
        resp = self._send(f"{AXIS_CMD[ax]} {diff}")
        if resp == "OK":
            self._positions[ax] = new_pos
        return resp

    def get_raw_positions(self):
        return dict(self._positions)


# ─────────────────────────────────────────────
#  VISION — cámara + YOLO
# ─────────────────────────────────────────────
class VisionPipeline:
    def __init__(self, cuadricula, simulate=False):
        self.simulate   = simulate
        self.cuadricula = cuadricula
        self._cap       = None
        self._yolo      = None

    def __enter__(self):
        if not self.simulate:
            from ultralytics import YOLOWorld
            self._cap = cv2.VideoCapture(CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self._cap.isOpened():
                raise RuntimeError(f"No se pudo abrir la camara {CAMERA_INDEX}.")
            print(f"Camara {CAMERA_INDEX} abierta.")
            print("Cargando YOLOWorld...")
            self._yolo = YOLOWorld("yolov8m-world.pt")
            self._yolo.set_classes(YOLO_CLASSES)
            print("YOLOWorld listo.\n")
        else:
            print("Vision [SIMULADA] inicializada.")
        return self

    def __exit__(self, *_):
        if self._cap:
            self._cap.release()

    def read_frame(self):
        if self.simulate or self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def detectar(self, frame, celdas_con_demo=None):
        """
        Devuelve (detections, frame_anotado).
        detections: lista de dicts con bbox, celda_info, etc.
        """
        if celdas_con_demo is None:
            celdas_con_demo = set()

        if self.simulate or frame is None:
            return [], np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        results    = self._yolo.predict(frame, conf=YOLO_CONF,
                                        iou=0.25, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls   = int(box.cls[0])
            cx_px = (x1 + x2) / 2
            cy_px = (y1 + y2) / 2
            ci    = self.cuadricula.info_celda(cx_px, cy_px)
            detections.append({
                "bbox":      (x1, y1, x2, y2),
                "cx_px":     cx_px, "cy_px":  cy_px,
                "cx_norm":   cx_px / w, "cy_norm": cy_px / h,
                "conf":      float(box.conf[0]),
                "color":     COLORES[cls % len(COLORES)],
                "celda_info": ci,
            })
        detections.sort(key=lambda x: x["conf"], reverse=True)

        # Anotar frame
        annotated = frame.copy()
        hl = detections[0]["celda_info"]["celda"] if detections else None
        annotated = self.cuadricula.dibujar(annotated,
                                             highlight=hl,
                                             celdas_con_demo=celdas_con_demo)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cx_i = int(det["cx_px"]); cy_i = int(det["cy_px"])
            color = det["color"]
            ci    = det["celda_info"]
            celda_str = f"C{ci['celda']}" if ci["celda"] else "fuera"
            tag = f"Trozo ({celda_str})  {det['conf']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx_i, cy_i), 6, (0, 0, 255), -1)
            cv2.circle(annotated, (cx_i, cy_i), 6, (255, 255, 255), 1)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(annotated, tag, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        if detections:
            x1, y1, x2, y2 = [int(v) for v in detections[0]["bbox"]]
            cv2.rectangle(annotated, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 3)
            ci  = detections[0]["celda_info"]
            txt = (f"OBJETIVO  Celda {ci['celda']}  Fila {ci['fila']}  Col {ci['columna']}"
                   if ci["celda"] else "OBJETIVO (fuera de malla)")
            cv2.putText(annotated, txt, (x1, y2 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.rectangle(annotated, (0, h-32), (w, h), (0, 0, 0), -1)
        cv2.putText(annotated,
                    f"Trozos: {len(detections)}  |  "
                    f"ESPACIO = confirmar celda   ESC = cancelar   q = salir",
                    (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        return detections, annotated


# ─────────────────────────────────────────────
#  CAPTURA INTERACTIVA
# ─────────────────────────────────────────────
def capturar_celda(vision, celdas_con_demo):
    """
    Abre ventana con YOLO + malla.
    ESPACIO → confirmar celda detectada.
    ESC / q → cancelar (devuelve None).
    Devuelve dict con celda_info o None.
    """
    VENTANA = "BC TEST  |  ESPACIO = confirmar celda   ESC/q = salir"
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, 1100, 650)
    cv2.moveWindow(VENTANA, 50, 50)
    try:
        cv2.setWindowProperty(VENTANA, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    # Pantalla de espera
    espera = np.zeros((650, 1100, 3), dtype=np.uint8)
    cv2.putText(espera, "Iniciando camara...",
                (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(espera, "Verde = celdas con demo    Naranja = objetivo detectado",
                (160, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(espera, "ESPACIO = confirmar   ESC/q = salir",
                (280, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.imshow(VENTANA, espera)
    cv2.waitKey(1)

    # Warm-up
    for _ in range(15):
        f = vision.read_frame()
        if f is not None:
            cv2.imshow(VENTANA, f); cv2.waitKey(1)
            break
        cv2.waitKey(50)

    resultado = None

    while True:
        frame = vision.read_frame()
        if frame is None:
            cv2.imshow(VENTANA, espera)
            key = cv2.waitKey(80) & 0xFF
            if key in (27, ord('q')):
                break
            continue

        detections, annotated = vision.detectar(frame, celdas_con_demo)
        cv2.imshow(VENTANA, annotated)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord('q')):
            print("  Cancelado por usuario.")
            resultado = "SALIR"
            break

        if key == 32:   # ESPACIO
            if not detections:
                print("  Sin deteccion YOLO. Apunta mejor al plato.")
                continue
            best = detections[0]
            ci   = best["celda_info"]
            if ci["celda"] is None:
                print("  El trozo esta fuera de la malla. Reposiciona el plato.")
                continue
            resultado = {
                "celda_info": ci,
                "cx_norm":    best["cx_norm"],
                "cy_norm":    best["cy_norm"],
                "cx_px":      best["cx_px"],
                "cy_px":      best["cy_px"],
                "conf":       best["conf"],
            }
            print(f"  Celda confirmada: {ci['celda']}  "
                  f"(Fila {ci['fila']}, Col {ci['columna']})")
            break

    cv2.destroyWindow(VENTANA)
    cv2.waitKey(1)
    return resultado


# ─────────────────────────────────────────────
#  BEHAVIOR CLONING — ejecutar demo
# ─────────────────────────────────────────────
def ejecutar_demo(robot, demo, celda_detectada, celda_usada, distancia):
    """
    Reproduce los pasos de la demo en el robot.
    El dataset limpio tiene pasos en orden: B → C → H → GRIP 0.
    """
    pasos = demo.get("steps", [])
    print(f"\n{'─'*55}")
    print(f"  BEHAVIOR CLONING")
    print(f"  Celda detectada : {celda_detectada}")
    print(f"  Celda de la demo: {celda_usada}  "
          f"{'(exacta)' if distancia == 0 else f'(dist. malla = {distancia:.2f})'}")
    print(f"  Pasos a ejecutar: {len(pasos)}")
    for s in pasos:
        print(f"    {s['cmd']:<8} {s.get('steps_executed', 0):>6} pasos")
    print(f"{'─'*55}\n")

    for i, step in enumerate(pasos, 1):
        cmd = step.get("cmd", "")
        print(f"  Paso {i}/{len(pasos)}: {cmd}  {step.get('steps_executed',0)} pasos")
        resp = robot.ejecutar_paso(step)
        if resp not in ("OK", "SKIP"):
            print(f"  ⚠ Respuesta inesperada: {resp}. Continuando...")
        time.sleep(0.1)

    print("\n  Demo ejecutada.")


# ─────────────────────────────────────────────
#  BUCLE PRINCIPAL
# ─────────────────────────────────────────────
def main(simulate=False):
    # Cargar recursos
    cuadricula = Cuadricula(CALIB_JSON)
    dataset    = BCDataset(CLEAN_PKL)
    celdas_con_demo = set(dataset._index.keys())

    ciclo = 0

    with RobotInterface(simulate=simulate) as robot:
        vision = VisionPipeline(cuadricula=cuadricula, simulate=simulate)
        with vision:

            print("=" * 58)
            print("  BEHAVIOR CLONING TEST")
            print(f"  {len(dataset.episodes)} demos cargadas")
            print(f"  {len(celdas_con_demo)} celdas cubiertas")
            print("=" * 58)
            print()
            print("  Teclas durante captura:")
            print("    ESPACIO  →  confirmar celda del trozo")
            print("    ESC / q  →  terminar programa")
            print()

            continuar = True

            while continuar:
                ciclo += 1
                print(f"\n{'='*58}")
                print(f"  CICLO {ciclo}")
                print(f"{'='*58}")

                # ── 1. HOME ───────────────────────────────────────────────
                print("\n[1] Ir a HOME...")
                robot.go_home()

                # ── 2. Cerrar gripper ─────────────────────────────────────
                print("\n[2] Cerrando gripper para captura...")
                robot.set_gripper(PINZA_MIN, "listo para captura")
                time.sleep(0.5)

                # ── 3. Captura con cámara ─────────────────────────────────
                print("\n[3] Abriendo camara — apunta al plato y presiona ESPACIO...")
                captura = capturar_celda(vision, celdas_con_demo)

                if captura is None or captura == "SALIR":
                    print("\n  Saliendo...")
                    continuar = False
                    break

                ci             = captura["celda_info"]
                celda_target   = ci["celda"]

                # ── 4. Buscar demo más cercana (BC) ───────────────────────
                print(f"\n[4] Buscando demo para celda {celda_target}...")
                demo, celda_usada, dist = dataset.demo_mas_cercana(celda_target)

                if dist == 0:
                    print(f"  Demo exacta encontrada para celda {celda_usada}.")
                else:
                    print(f"  No hay demo exacta. Usando celda {celda_usada} "
                          f"(dist. malla = {dist:.2f}).")

                # ── 5. Abrir gripper antes del agarre ─────────────────────
                print("\n[5] Abriendo gripper para agarre...")
                robot.set_gripper(PINZA_MAX, "pre-agarre")
                time.sleep(0.5)

                # ── 6. Ejecutar demo (BC) ─────────────────────────────────
                print("\n[6] Ejecutando movimientos de la demo (BC)...")
                # Los pasos del dataset ya incluyen GRIP 0 al final,
                # así que ejecutamos todo excepto el GRIP 0 final
                # (el cierre del gripper lo manejamos aparte para claridad)
                pasos_sin_grip = [s for s in demo.get("steps", [])
                                  if s.get("cmd") != "GRIP 0"]
                demo_sin_grip  = {**demo, "steps": pasos_sin_grip}
                ejecutar_demo(robot, demo_sin_grip, celda_target, celda_usada, dist)

                # ── 7. Cerrar gripper (agarre) ────────────────────────────
                print("\n[7] Cerrando gripper — AGARRE...")
                robot.set_gripper(PINZA_MIN, "agarrando alimento")
                time.sleep(0.8)

                # ── 8. Volver a HOME: Hombro → Codo → Base ───────────────
                print("\n[8] Volviendo a HOME...")
                robot.go_home()
                time.sleep(0.3)

                # ── 9. Abrir gripper (soltar) ─────────────────────────────
                print("\n[9] Abriendo gripper — SOLTANDO alimento...")
                robot.set_gripper(PINZA_MAX, "soltando")
                time.sleep(0.8)

                # ── 10. Cerrar gripper para siguiente ciclo ───────────────
                print("\n[10] Cerrando gripper para siguiente captura...")
                robot.set_gripper(PINZA_MIN, "listo para siguiente")
                time.sleep(0.4)

                # ── Continuar? ────────────────────────────────────────────
                print(f"\n  Ciclo {ciclo} completado.")
                print("  Presiona ENTER para continuar con otro trozo, "
                      "o escribe 'q' para salir: ", end="", flush=True)
                try:
                    resp = input().strip().lower()
                    if resp == "q":
                        print("  Saliendo.")
                        continuar = False
                except (EOFError, KeyboardInterrupt):
                    continuar = False

    cv2.destroyAllWindows()
    print("\nPrograma terminado.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behavior Cloning Test — Brazo Robótico Asistente")
    parser.add_argument("--sim", action="store_true",
                        help="Modo simulacion (sin Arduino ni camara real)")
    parser.add_argument("--calib", default=CALIB_JSON,
                        help="Ruta al JSON de calibracion de la malla")
    parser.add_argument("--dataset", default=CLEAN_PKL,
                        help="Ruta al PKL del dataset limpio")
    args = parser.parse_args()

    # Permitir sobreescribir rutas por argumento
    if args.calib  != CALIB_JSON:  globals()["CALIB_JSON"] = args.calib
    if args.dataset != CLEAN_PKL:  globals()["CLEAN_PKL"]  = args.dataset

    main(simulate=args.sim)
