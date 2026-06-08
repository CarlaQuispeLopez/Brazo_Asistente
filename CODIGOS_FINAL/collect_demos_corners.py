import os
import json
import pickle
import threading
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

DEMO_DIR        = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\demos"
DEMO_FILE       = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\demos\demonstrations copy.pkl"
CHECKPOINT_FILE = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\demos\checkpoint_copy_activo.pkl"
CALIB_JSON      = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\calibracion_cuadricula.json"

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

PINZA_MIN = 0
PINZA_MAX = 90

COMMAND_MAP = {
    "B+": ("base",      +1,  0),
    "B-": ("base",      -1,  1),
    "H+": ("hombro",    +1,  2),
    "H-": ("hombro",    -1,  3),
    "C+": ("codo",      +1,  4),
    "C-": ("codo",      -1,  5),
    "M+": ("muneca",    +1,  6),
    "M-": ("muneca",    -1,  7),
    "G+": ("rotacion",  +1,  8),
    "G-": ("rotacion",  -1,  9),
}

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


class Cuadricula:
    COLOR_LINEA   = (0, 255, 255)
    COLOR_NUMERO  = (255, 255, 0)
    COLOR_RELLENO = (0, 60, 60)

    def __init__(self, json_path):
        self.cols  = 12
        self.rows  = 9
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
            print(f"[Cuadricula] No se pudo cargar '{path}': {e}\n")

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

    def num_esquinas(self):
        return (self.cols + 1) * (self.rows + 1)

    def esquina_num(self, col_e, row_e):
        return row_e * (self.cols + 1) + col_e + 1

    def esquina_colrow(self, num):
        idx   = num - 1
        row_e = idx // (self.cols + 1)
        col_e = idx %  (self.cols + 1)
        return col_e, row_e

    def esquina_px(self, num):
        col_e, row_e = self.esquina_colrow(num)
        return self._t([[col_e, row_e]])[0]

    def todas_esquinas_px(self):
        result = []
        for row_e in range(self.rows + 1):
            for col_e in range(self.cols + 1):
                num = self.esquina_num(col_e, row_e)
                px  = self._t([[col_e, row_e]])[0]
                result.append((num, px))
        return result

    def esquina_mas_cercana(self, px, py):
        if not self.disponible:
            return None
        gx, gy = cv2.perspectiveTransform(
            np.float32([[[px, py]]]), self.M_inv)[0][0]
        col_e = int(round(np.clip(gx, 0, self.cols)))
        row_e = int(round(np.clip(gy, 0, self.rows)))
        num   = self.esquina_num(col_e, row_e)
        pt_px = self._t([[col_e, row_e]])[0]
        return {
            "esquina":  num,
            "col_e":    col_e,
            "row_e":    row_e,
            "px_img":   float(pt_px[0]),
            "py_img":   float(pt_px[1]),
        }

    def dibujar(self, frame, highlight=None, celdas_usadas=None,
                esquina_highlight=None, esquinas_usadas=None):
        if not self.disponible:
            return frame

        if celdas_usadas  is None: celdas_usadas  = {}
        if esquinas_usadas is None: esquinas_usadas = {}

        overlay = frame.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                num  = r * self.cols + c + 1
                corn = [[c,r],[c+1,r],[c+1,r+1],[c,r+1]]
                pts  = self._t(corn).astype(np.int32)
                if num == highlight:
                    color = (0, 0, 100)
                elif num in celdas_usadas:
                    color = (0, 0, 180)
                else:
                    color = self.COLOR_RELLENO
                cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        if highlight and 1 <= highlight <= self.cols * self.rows:
            r = (highlight - 1) // self.cols
            c = (highlight - 1) % self.cols
            pts = self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
            cv2.fillPoly(frame, [pts], (180, 60, 0))

        for celda_num in celdas_usadas:
            if celda_num == highlight:
                continue
            r = (celda_num - 1) // self.cols
            c = (celda_num - 1) % self.cols
            pts = self._t([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]).astype(np.int32)
            cv2.fillPoly(frame, [pts], (0, 0, 200))

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

        for r in range(self.rows):
            for c in range(self.cols):
                num = r * self.cols + c + 1
                cx  = self._t([[c + 0.5, r + 0.5]])[0].astype(int)
                cnt = celdas_usadas.get(num, 0)
                txt = f"{num}" if cnt == 0 else f"{num}({cnt})"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                if num == highlight:
                    color = (255, 255, 255)
                elif num in celdas_usadas:
                    color = (255, 255, 255)
                else:
                    color = self.COLOR_NUMERO
                cv2.putText(frame, txt,
                            (int(cx[0]) - tw//2, int(cx[1]) + th//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

        RADIO        = 12
        COLOR_NORMAL = (0, 0, 220)
        COLOR_USADO  = (0, 0, 255)
        COLOR_TARGET = (0, 255, 255)

        for row_e in range(self.rows + 1):
            for col_e in range(self.cols + 1):
                num_e = self.esquina_num(col_e, row_e)
                pt    = self._t([[col_e, row_e]])[0].astype(int)
                centro = tuple(pt)
                cnt_e  = esquinas_usadas.get(num_e, 0)

                if num_e == esquina_highlight:
                    cv2.circle(frame, centro, RADIO + 4, (255, 255, 255), 2)
                    cv2.circle(frame, centro, RADIO,     COLOR_TARGET,    -1)
                    cv2.circle(frame, centro, RADIO,     (255, 255, 255),  1)
                elif cnt_e > 0:
                    cv2.circle(frame, centro, RADIO, COLOR_USADO, -1)
                    cv2.circle(frame, centro, RADIO, (255, 255, 255), 1)
                    txt_e = str(cnt_e)
                    (tw2, th2), _ = cv2.getTextSize(txt_e, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                    cv2.putText(frame, txt_e,
                                (centro[0] - tw2//2, centro[1] + th2//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                else:
                    cv2.circle(frame, centro, RADIO, COLOR_NORMAL, 2)
                    cv2.line(frame,
                             (centro[0]-5, centro[1]), (centro[0]+5, centro[1]),
                             (0, 220, 220), 1)
                    cv2.line(frame,
                             (centro[0], centro[1]-5), (centro[0], centro[1]+5),
                             (0, 220, 220), 1)

        return frame


class RobotInterface:
    def __init__(self, simulate=False):
        self.simulate      = simulate
        self._positions    = {ax: 0 for ax in HOME_POSITION}
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
            return "OK"
        self._ser.flushInput()
        self._ser.write((cmd + "\n").encode())
        deadline = time.time() + 15
        while time.time() < deadline:
            line = self._ser.readline().decode(errors="ignore").strip()
            if line == "OK":
                return "OK"
            if line == "ERR":
                return "ERR"
        return "TIMEOUT"

    def go_home(self):
        orden = ["hombro", "codo", "muneca", "rotacion", "base"]
        print("Regresando a HOME...")
        for ax in orden:
            target  = HOME_POSITION[ax]
            current = self._positions[ax]
            diff    = target - current
            if diff == 0:
                continue
            print(f"  {ax}: {current:+d} -> {target:+d}  ({diff:+d} pasos)")
            self._send(f"{AXIS_CMD[ax]} {diff}")
            self._positions[ax] = target
        print(f"HOME alcanzado: {HOME_POSITION}")

    def move_joint(self, axis, steps):
        lo, hi  = JOINT_LIMITS[axis]
        new_pos = self._positions[axis] + steps
        if new_pos < lo or new_pos > hi:
            return False
        resp = self._send(f"{AXIS_CMD[axis]} {steps}")
        if resp != "OK":
            return False
        self._positions[axis] = new_pos
        return True

    def set_gripper_angle(self, angulo):
        angulo = int(np.clip(angulo, PINZA_MIN, PINZA_MAX))
        resp   = self._send(f"PINZA {angulo}")
        if resp == "OK":
            self._angulo_pinza = angulo
        return resp

    def get_gripper_angle(self):
        return self._angulo_pinza

    def get_raw_positions(self):
        return dict(self._positions)

    def get_joint_positions(self):
        norm = []
        for ax in ["base", "hombro", "codo", "muneca", "rotacion"]:
            lo, hi = JOINT_LIMITS[ax]
            rango  = hi - lo
            norm.append((self._positions[ax] - lo) / rango * 2 - 1)
        return np.array(norm, dtype=np.float32)


class FoodDetection:
    def __init__(self, label, confidence, center_norm, center_px,
                 celda_info, esquina_info=None):
        self.label        = label
        self.confidence   = confidence
        self.center_norm  = center_norm
        self.center_px    = center_px
        self.celda_info   = celda_info
        self.esquina_info = esquina_info


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

    def get_detections(self, frame, celdas_usadas=None, esquinas_usadas=None):
        if celdas_usadas   is None: celdas_usadas   = {}
        if esquinas_usadas is None: esquinas_usadas  = {}

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
            celda_info  = self.cuadricula.info_celda(cx_px, cy_px)
            esquina_info = self.cuadricula.esquina_mas_cercana(cx_px, cy_px)
            detections.append({
                "bbox":        (x1, y1, x2, y2),
                "cx_px":       cx_px,
                "cy_px":       cy_px,
                "cx_norm":     cx_px / w,
                "cy_norm":     cy_px / h,
                "conf":        float(box.conf[0]),
                "color":       COLORES[cls % len(COLORES)],
                "celda_info":  celda_info,
                "esquina_info": esquina_info,
            })
        detections.sort(key=lambda x: x["conf"], reverse=True)

        annotated = frame.copy()
        hl      = detections[0]["celda_info"]["celda"]   if detections else None
        hl_esq  = detections[0]["esquina_info"]["esquina"] if detections else None
        annotated = self.cuadricula.dibujar(annotated,
                                             highlight=hl,
                                             celdas_usadas=celdas_usadas,
                                             esquina_highlight=hl_esq,
                                             esquinas_usadas=esquinas_usadas)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cx_i = int(det["cx_px"]); cy_i = int(det["cy_px"])
            color = det["color"]
            ci    = det["celda_info"]
            ei    = det["esquina_info"]
            celda_str = f"C{ci['celda']}" if ci["celda"] else "fuera"
            esq_str   = f"E{ei['esquina']}" if ei else "?"
            tag = f"Trozo {celda_str}/{esq_str}  {det['conf']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx_i, cy_i), 6, (0, 0, 255), -1)
            cv2.circle(annotated, (cx_i, cy_i), 6, (255, 255, 255), 1)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(annotated, tag, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        if detections:
            x1, y1, x2, y2 = [int(v) for v in detections[0]["bbox"]]
            cv2.rectangle(annotated, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 3)
            ei  = detections[0]["esquina_info"]
            ci  = detections[0]["celda_info"]
            txt = (f"OBJETIVO  Celda {ci['celda']}  |  Esquina {ei['esquina']} "
                   f"(col={ei['col_e']}, row={ei['row_e']})"
                   if ei else "OBJETIVO  (fuera de malla)")
            cv2.putText(annotated, txt, (x1, y2 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        n  = len(detections)
        nu = len(esquinas_usadas)
        cv2.rectangle(annotated, (0, h-32), (w, h), (0,0,0), -1)
        cv2.putText(annotated,
                    f"Trozos: {n}  |  ESPACIO=confirmar esquina mas cercana  ESC=cancelar  "
                    f"Rojo=esquinas con demos ({nu} usadas)",
                    (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        return detections, annotated

    def food_to_state_vector(self, food, joint_positions):
        obs = (np.array(food.center_norm, dtype=np.float32)
               if food else np.zeros(2, dtype=np.float32))
        return np.concatenate([obs, joint_positions]).astype(np.float32)


def capturar_alimento(vision, celdas_usadas=None, esquinas_usadas=None):
    if celdas_usadas   is None: celdas_usadas   = {}
    if esquinas_usadas is None: esquinas_usadas  = {}

    VENTANA = "CAPTURA ESQUINAS  |  ESPACIO = confirmar   ESC = cancelar"

    print("\n  Abriendo ventana de captura (modo ESQUINAS)...")
    print("  El sistema seleccionará la ESQUINA mas cercana al trozo de comida.")
    print(f"  Esquinas con demos (rojo relleno): {sorted(esquinas_usadas.keys())}")
    print("  ESPACIO = confirmar esquina   ESC = cancelar\n")

    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, 1100, 650)
    cv2.moveWindow(VENTANA, 50, 50)
    try:
        cv2.setWindowProperty(VENTANA, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    pantalla_inicio = np.zeros((650, 1100, 3), dtype=np.uint8)
    cv2.putText(pantalla_inicio, "Iniciando camara...",
                (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(pantalla_inicio, "ESPACIO = confirmar esquina mas cercana",
                (220, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(pantalla_inicio, "Circulos ROJOS = esquinas con demos",
                (280, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
    cv2.imshow(VENTANA, pantalla_inicio)
    cv2.waitKey(1)

    for _ in range(15):
        f = vision.read_frame()
        if f is not None:
            cv2.imshow(VENTANA, f)
            cv2.waitKey(1)
            break
        cv2.waitKey(50)

    resultado = None

    while True:
        frame = vision.read_frame()
        if frame is None:
            cv2.imshow(VENTANA, pantalla_inicio)
            if cv2.waitKey(80) & 0xFF == 27:
                break
            continue

        h, w = frame.shape[:2]
        detections, annotated = vision.get_detections(
            frame,
            celdas_usadas=celdas_usadas,
            esquinas_usadas=esquinas_usadas,
        )
        cv2.imshow(VENTANA, annotated)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            print("  Captura cancelada.\n")
            break

        if key == 32:
            if not detections:
                print("  Sin deteccion YOLO. Sigue apuntando al plato.")
                continue
            best = detections[0]
            ei   = best["esquina_info"]
            ci   = best["celda_info"]

            cx_norm_esq = ei["px_img"] / w
            cy_norm_esq = ei["py_img"] / h

            resultado = FoodDetection(
                "Trozo_Comida",
                best["conf"],
                (cx_norm_esq, cy_norm_esq),
                (int(ei["px_img"]), int(ei["py_img"])),
                ci,
                esquina_info=ei,
            )
            ya_tiene = esquinas_usadas.get(ei["esquina"], 0)
            aviso    = f"  *** ya tiene {ya_tiene} demo(s) ***" if ya_tiene else ""
            print(f"  Esquina confirmada:")
            print(f"    Esquina num  : {ei['esquina']}  "
                  f"(col={ei['col_e']}, row={ei['row_e']}){aviso}")
            print(f"    Pos px       : ({ei['px_img']:.1f}, {ei['py_img']:.1f})")
            print(f"    Pos norm     : cx={cx_norm_esq:.4f}  cy={cy_norm_esq:.4f}")
            print(f"    Celda adj    : {ci['celda']}  "
                  f"(Fila {ci['fila']}, Col {ci['columna']})" if ci["celda"] else "")
            print(f"    Confianza YOLO: {best['conf']:.2f}")
            print()
            break

    cv2.destroyWindow(VENTANA)
    cv2.waitKey(1)
    return resultado


class DemoCollector:

    def __init__(self, robot, vision):
        self.robot  = robot
        self.vision = vision

        self._recording    = False
        self._food_target  = None
        self._cam_captured = False
        self._current_ep   = []

        Path(DEMO_DIR).mkdir(parents=True, exist_ok=True)
        self.data = self._load()
        self._load_checkpoint()

    def _load(self):
        if os.path.exists(DEMO_FILE):
            with open(DEMO_FILE, "rb") as f:
                d = pickle.load(f)
            eps = d.get("episodes", [])
            print(f"Dataset cargado: {len(eps)} episodios previos.")
            celdas = [e.get("celda_objetivo") for e in eps if e.get("celda_objetivo")]
            if celdas:
                from collections import Counter
                cnt = Counter(celdas)
                print(f"  Celdas con demos: {dict(sorted(cnt.items()))}")
            return d
        return {"episodes": [], "created": datetime.now().isoformat()}

    def _save(self):
        with open(DEMO_FILE, "wb") as f:
            pickle.dump(self.data, f)

    def _save_checkpoint(self):
        ft   = self._food_target
        ckpt = {
            "episode_num":  self._ep_count() + 1,
            "food_target":  ft,
            "current_ep":   list(self._current_ep),
            "positions":    self.robot.get_raw_positions(),
            "angulo_pinza": self.robot.get_gripper_angle(),
            "timestamp":    datetime.now().isoformat(),
        }
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(ckpt, f)

    def _load_checkpoint(self):
        if not os.path.exists(CHECKPOINT_FILE):
            return
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                ckpt = pickle.load(f)
        except Exception:
            os.remove(CHECKPOINT_FILE)
            return

        n  = len(ckpt.get("current_ep", []))
        ep = ckpt.get("episode_num", "?")
        ts = ckpt.get("timestamp", "?")
        ft = ckpt.get("food_target")
        celda_str = ""
        if ft and hasattr(ft, "celda_info"):
            celda_str = f"  Celda: {ft.celda_info.get('celda','?')}"
        print(f"\n{'='*52}")
        print(f"  CHECKPOINT encontrado — episodio {ep} incompleto")
        print(f"  Pasos grabados: {n}   |   Guardado: {ts}{celda_str}")
        print(f"{'='*52}")
        print("  Recuperar episodio? (s/n): ", end="", flush=True)
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans == "s":
            self._food_target  = ft
            self._cam_captured = ft is not None
            self._current_ep   = ckpt.get("current_ep", [])
            self._recording    = True
            saved_pos = ckpt.get("positions", {})
            for ax, val in saved_pos.items():
                if ax in self.robot._positions:
                    self.robot._positions[ax] = val
            self.robot._angulo_pinza = ckpt.get("angulo_pinza", 0)
            print(f"  Episodio {ep} recuperado con {n} pasos.")
            print(f"  El brazo debe estar en la posicion: {saved_pos}")
            print(f"  Pinza en: {self.robot._angulo_pinza} grados")
            print(f"  Continua grabando normalmente. F para finalizar.\n")
        else:
            os.remove(CHECKPOINT_FILE)
            print("  Checkpoint descartado.\n")

    def _clear_checkpoint(self):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    def _ep_count(self):
        return len(self.data.get("episodes", []))

    def _get_state(self):
        return self.vision.food_to_state_vector(
            self._food_target, self.robot.get_joint_positions())

    def _pedir_pasos(self):
        try:
            return int(input("  Pasos: ").strip())
        except (ValueError, EOFError):
            print("  Valor invalido, usando 0.")
            return 0

    def _pedir_angulo(self):
        try:
            v = int(input(f"  Angulo pinza ({PINZA_MIN}-{PINZA_MAX} grados): ").strip())
            return int(np.clip(v, PINZA_MIN, PINZA_MAX))
        except (ValueError, EOFError):
            print("  Valor invalido, usando 0.")
            return 0

    def _reset(self):
        self._recording    = False
        self._food_target  = None
        self._cam_captured = False
        self._current_ep   = []

    def _celdas_usadas(self):
        from collections import Counter
        eps    = self.data.get("episodes", [])
        celdas = [e.get("celda_objetivo") for e in eps if e.get("celda_objetivo")]
        return dict(Counter(celdas))

    def _esquinas_usadas(self):
        from collections import Counter
        eps     = self.data.get("episodes", [])
        esquinas = [e.get("esquina_objetivo") for e in eps if e.get("esquina_objetivo")]
        return dict(Counter(esquinas))

    def cmd_cam(self):
        print("  Cerrando garra antes de capturar...")
        self.robot.set_gripper_angle(PINZA_MIN)
        food = capturar_alimento(
            self.vision,
            celdas_usadas=self._celdas_usadas(),
            esquinas_usadas=self._esquinas_usadas(),
        )
        if food is None:
            print("  Sin objetivo confirmado.")
            return
        self._food_target  = food
        self._cam_captured = True
        ei = food.esquina_info
        if ei:
            print(f"  Objetivo: Esquina {ei['esquina']} "
                  f"(col={ei['col_e']}, row={ei['row_e']})")
        print("  Escribe I cuando el brazo este fisicamente en HOME.")

    def cmd_iniciar(self):
        if self._recording:
            print("Ya hay una grabacion activa.")
            return
        self.robot._positions = dict(HOME_POSITION)
        self._recording  = True
        self._current_ep = []
        print(f"Grabacion iniciada. Episodio {self._ep_count() + 1}.")
        print(f"Posicion HOME asumida: {HOME_POSITION}")
        if self._food_target:
            ci = self._food_target.celda_info
            ei = getattr(self._food_target, "esquina_info", None)
            esq_str = f"Esquina {ei['esquina']} (col={ei['col_e']}, row={ei['row_e']})" if ei else "sin esquina"
            print(f"  Objetivo: Celda {ci.get('celda','?')} | {esq_str}")
        if not self._cam_captured:
            print("Aviso: aun no capturaste el alimento. Usa CAM antes de mover.")

    def cmd_finalizar(self):
        if not self._recording:
            print("No hay grabacion activa.")
            return
        self._recording = False
        n = len(self._current_ep)
        if n == 0:
            print("Episodio vacio, descartado.")
            self._reset()
            return

        print(f"Grabacion detenida. {n} pasos.")
        print("Fue exitoso el agarre? (s/n): ", end="", flush=True)
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        success = ans == "s"

        ft = self._food_target
        ci = ft.celda_info   if ft else {"celda": None, "fila": None, "columna": None}
        ei = ft.esquina_info if (ft and hasattr(ft, "esquina_info")) else None

        episode = {
            "food_target": {
                "label":      ft.label           if ft else None,
                "confidence": ft.confidence      if ft else None,
                "cx_norm":    ft.center_norm[0]  if ft else None,
                "cy_norm":    ft.center_norm[1]  if ft else None,
                "cx_px":      ft.center_px[0]    if ft else None,
                "cy_px":      ft.center_px[1]    if ft else None,
            },
            "celda_objetivo":   ci["celda"],
            "fila_objetivo":    ci["fila"],
            "columna_objetivo": ci["columna"],
            "esquina_objetivo": ei["esquina"] if ei else None,
            "esquina_col":      ei["col_e"]   if ei else None,
            "esquina_row":      ei["row_e"]   if ei else None,
            "esquina_px":       ei["px_img"]  if ei else None,
            "esquina_py":       ei["py_img"]  if ei else None,
            "steps":     list(self._current_ep),
            "success":   success,
            "timestamp": datetime.now().isoformat(),
            "n_steps":   n,
            "home_used": dict(HOME_POSITION),
        }
        self.data.setdefault("episodes", []).append(episode)
        self._save()
        self._clear_checkpoint()
        esq_str = f"Esquina={ei['esquina']}" if ei else "Esquina=?"
        print(f"Episodio {self._ep_count()} guardado. "
              f"Celda={ci['celda']}  {esq_str}  Pasos={n}  Exito={success}")

        self.robot.go_home()
        print("  Abriendo garra...")
        self.robot.set_gripper_angle(PINZA_MAX)
        self._reset()

    def cmd_cancelar(self):
        n = len(self._current_ep)
        if self._recording:
            print("Regresando a HOME antes de cancelar...")
            self.robot.go_home()
        self._clear_checkpoint()
        self._reset()
        print(f"Episodio cancelado. {n} pasos descartados.")

    def cmd_grip(self):
        angulo       = self._pedir_angulo()
        estado       = "REC" if self._recording else "libre"
        state_before = self._get_state()
        resp         = self.robot.set_gripper_angle(angulo)
        print(f"[{estado}] GRIP  pinza -> {angulo} grados  ({resp})")
        if self._recording:
            self._current_ep.append({
                "state":          state_before.copy(),
                "action":         10,
                "cmd":            f"GRIP {angulo}",
                "angulo_pinza":   angulo,
                "steps_executed": 0,
            })
            self._save_checkpoint()

    def cmd_mover(self, cmd):
        axis, direction, action_idx = COMMAND_MAP[cmd]
        steps = self._pedir_pasos()
        if steps == 0:
            return
        state_before = self._get_state()
        ok = self.robot.move_joint(axis, direction * steps)
        if not ok:
            print(f"Limite articular alcanzado en {axis}.")
            return
        pos    = self.robot.get_raw_positions().get(axis, 0)
        estado = "REC" if self._recording else "libre"
        print(f"[{estado}] {cmd} {steps}  eje={axis}  pos={pos}")
        if self._recording and action_idx >= 0:
            self._current_ep.append({
                "state":          state_before.copy(),
                "action":         action_idx,
                "cmd":            cmd,
                "steps_executed": steps,
            })
            self._save_checkpoint()

    def cmd_status(self):
        pos  = self.robot.get_raw_positions()
        norm = self.robot.get_joint_positions()
        axes = ["base", "hombro", "codo", "muneca", "rotacion"]
        print("─" * 52)
        for ax, nv in zip(axes, norm):
            hv   = HOME_POSITION[ax]
            diff = pos.get(ax, 0) - hv
            print(f"  {ax:<10} actual={pos.get(ax,0):+6d}  "
                  f"home={hv:+6d}  diff={diff:+6d}  norm={nv:+.3f}")
        print(f"  Pinza:     {self.robot.get_gripper_angle()} grados")
        print(f"  Grabando:  {self._recording}")
        print(f"  Episodios: {self._ep_count()}")
        if self._food_target:
            ft = self._food_target
            ci = ft.celda_info
            ei = getattr(ft, "esquina_info", None)
            esq_str = f"Esq={ei['esquina']} (c={ei['col_e']},r={ei['row_e']})" if ei else ""
            print(f"  Objetivo:  Celda={ci['celda']}  {esq_str}  "
                  f"cx={ft.center_norm[0]:.3f} cy={ft.center_norm[1]:.3f}")
        print("─" * 52)

    def run(self):
        print("\nComandos disponibles:")
        print("  CAM          — abrir camara (YOLO+malla+circulos), ESPACIO para confirmar ESQUINA")
        print("  I            — brazo esta en HOME, iniciar grabacion")
        print("  F            — finalizar, guardar y REGRESAR A HOME")
        print("  CANCEL       — cancelar episodio y regresar a HOME")
        print("  STATUS       — posicion de articulaciones vs HOME")
        print("  B+/B-        — base        (pedira pasos)")
        print("  H+/H-        — hombro      (pedira pasos)")
        print("  C+/C-        — codo        (pedira pasos)")
        print("  M+/M-        — muneca      (pedira pasos)")
        print("  G+/G-        — rotacion    (pedira pasos)")
        print("  GRIP         — pinza       (pedira angulo 0-90 grados)")
        print("  EXIT         — salir\n")
        print(f"HOME = {HOME_POSITION}")
        print(f"Dataset: demonstrations copy.pkl\n")

        while True:
            ep  = self._ep_count() + 1
            rec = "REC" if self._recording else "---"
            ft  = self._food_target
            ei  = getattr(ft, "esquina_info", None) if ft else None
            cel = f"E{ei['esquina']}" if ei else "sin_esq"
            try:
                raw = input(f"EP{ep} [{rec}] [{cel}] > ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                raw = "EXIT"

            if not raw:
                continue

            if raw == "EXIT":
                if self._recording:
                    print("Grabacion activa. Regresando a HOME y cancelando.")
                    self.cmd_cancelar()
                print(f"Total episodios guardados: {self._ep_count()}")
                break
            elif raw == "CAM":    self.cmd_cam()
            elif raw == "I":      self.cmd_iniciar()
            elif raw == "F":      self.cmd_finalizar()
            elif raw == "CANCEL": self.cmd_cancelar()
            elif raw == "STATUS": self.cmd_status()
            elif raw == "GRIP":   self.cmd_grip()
            elif raw in COMMAND_MAP:
                self.cmd_mover(raw)
            else:
                print(f"Comando no reconocido: '{raw}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true",
                        help="Modo simulacion (sin Arduino ni camara)")
    parser.add_argument("--calib", default=CALIB_JSON,
                        help="Ruta al JSON de calibracion de la malla")
    args = parser.parse_args()

    cuadricula = Cuadricula(args.calib)

    with RobotInterface(simulate=args.sim) as robot:
        vision = VisionPipeline(cuadricula=cuadricula, simulate=args.sim)
        with vision:
            collector = DemoCollector(robot, vision)
            try:
                collector.run()
            finally:
                cv2.destroyAllWindows()