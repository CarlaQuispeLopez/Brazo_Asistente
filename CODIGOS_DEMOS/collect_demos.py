import os
import pickle
import threading
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

# ─────────────────────────────────────────────
#  CONFIGURACION
# ─────────────────────────────────────────────
DEMO_DIR     = "demos"
DEMO_FILE    = "demos/demonstrations.pkl"
DEPTH_MIN_CM = 10.0
DEPTH_MAX_CM = 60.0

SERIAL_PORT  = "COM5"
SERIAL_BAUD  = 115200
CAMERA_INDEX = 0

# Posicion HOME en pasos (lo que AccelStepper llama posicion 0
# despues de que el usuario presiona I con el brazo ya en HOME fisico)
HOME_POSITION = {
    "base":     0,
    "hombro":   100,
    "codo":     1100,
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

# Nombre del comando que el Arduino entiende para cada eje
# Segun el sketch: BASE, HOMBRO, CODO, GRIPPER, GIRO
AXIS_CMD = {
    "base":     "BASE",
    "hombro":   "HOMBRO",
    "codo":     "CODO",
    "muneca":   "GRIPPER",
    "rotacion": "GIRO",
}

# ─────────────────────────────────────────────
#  MAPA DE COMANDOS
# ─────────────────────────────────────────────
COMMAND_MAP = {
    "B+":    ("base",      +1,  0),
    "B-":    ("base",      -1,  1),
    "H+":    ("hombro",    +1,  2),
    "H-":    ("hombro",    -1,  3),
    "C+":    ("codo",      +1,  4),
    "C-":    ("codo",      -1,  5),
    "M+":    ("muneca",    +1,  6),
    "M-":    ("muneca",    -1,  7),
    "G+":    ("rotacion",  +1,  8),
    "G-":    ("rotacion",  -1,  9),
    "GRIP+": ("gripper_close", 0, 10),
    "GRIP-": ("gripper_open",  0, -1),
}


# ─────────────────────────────────────────────
#  ROBOT INTERFACE
# ─────────────────────────────────────────────
class RobotInterface:
    """
    Protocolo serial — mismo que usa el Arduino existente:
      Mover eje  ->  "CODO 500\n"   (nombre del eje + pasos con signo)
      Gripper    ->  "PINZA CERRAR\n"  /  "PINZA ABRIR\n"
      Home       ->  "HOME\n"
      Arduino responde "OK\n" al terminar cada comando.
    """

    def __init__(self, simulate=False):
        self.simulate        = simulate
        self._positions      = {ax: 0 for ax in HOME_POSITION}  # todo en 0 al arrancar
        self._gripper_closed = False
        self._ser            = None

    def __enter__(self):
        if not self.simulate:
            import serial
            self._ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=10)
            time.sleep(2)           # espera reset del Arduino
            self._ser.flushInput()
            # esperar el "READY" que manda el Arduino en setup()
            deadline = time.time() + 5
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
        """
        Envía un comando y espera 'OK'.
        En simulacion solo imprime el comando.
        """
        if self.simulate:
            print(f"    [SIM] -> {cmd}")
            return "OK"
        self._ser.flushInput()
        self._ser.write((cmd + "\n").encode())
        # leer hasta recibir OK (el Arduino puede mandar lineas intermedias)
        deadline = time.time() + 15
        while time.time() < deadline:
            line = self._ser.readline().decode(errors="ignore").strip()
            if line == "OK":
                return "OK"
            if line == "ERR":
                return "ERR"
        return "TIMEOUT"

    def set_home(self):
        """
        Sincroniza el contador interno con HOME_POSITION.
        Llama esto cuando el brazo YA esta fisicamente en HOME.
        En el Arduino, AccelStepper ya arranca en 0 —
        nosotros le enviamos los movimientos relativos desde aqui.
        """
        self._positions      = dict(HOME_POSITION)
        self._gripper_closed = False
        # Decirle al Arduino que su posicion actual es el origen
        # (AccelStepper ya hace esto al encender, pero si hubo movimientos
        #  previos en la misma sesion, reseteamos con HOME)
        self._send("HOME")

    def go_home(self):
        """
        Calcula los pasos de diferencia para cada eje y los envia.
        El Arduino mueve cada motor la cantidad exacta para volver a HOME.
        """
        orden = ["muneca", "rotacion", "codo", "hombro", "base"]
        print("Regresando a HOME...")
        for ax in orden:
            target  = HOME_POSITION[ax]
            current = self._positions[ax]
            diff    = target - current
            if diff == 0:
                continue
            nombre = AXIS_CMD[ax]
            print(f"  {ax}: {current:+d} -> {target:+d}  ({diff:+d} pasos)")
            self._send(f"{nombre} {diff}")
            self._positions[ax] = target
        print("HOME alcanzado.")

    def move_joint(self, axis, steps):
        lo, hi  = JOINT_LIMITS[axis]
        new_pos = self._positions[axis] + steps
        if new_pos < lo or new_pos > hi:
            return False
        nombre = AXIS_CMD[axis]
        resp   = self._send(f"{nombre} {steps}")
        if resp != "OK":
            return False
        self._positions[axis] = new_pos
        return True

    def close_gripper(self):
        self._gripper_closed = True
        self._send("PINZA CERRAR")

    def open_gripper(self):
        self._gripper_closed = False
        self._send("PINZA ABRIR")

    def get_raw_positions(self):
        return dict(self._positions)

    def get_joint_positions(self):
        """Posiciones normalizadas en [-1, 1] respecto a los limites."""
        norm = []
        for ax in ["base", "hombro", "codo", "muneca", "rotacion"]:
            lo, hi = JOINT_LIMITS[ax]
            rango  = hi - lo
            norm.append((self._positions[ax] - lo) / rango * 2 - 1)
        return np.array(norm, dtype=np.float32)


# ─────────────────────────────────────────────
#  VISION PIPELINE — YOLO + Depth Anything
# ─────────────────────────────────────────────
YOLO_CONF      = 0.35
DEPTH_EVERY_N  = 6          # estimar profundidad cada N frames
DEPTH_COLORMAP = 2          # cv2.COLORMAP_TURBO = 2
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ── Calibracion de profundidad ────────────────────────────────
# Formula obtenida con calibrar_profundidad.py midiendo 5 puntos reales.
# cm_real = DEPTH_CALIB_A * raw_value + DEPTH_CALIB_B
# Coeficiente negativo es normal: en Depth Anything raw alto = mas cercano.
DEPTH_CALIB_A =  0.000643
DEPTH_CALIB_B = 23.467003

YOLO_CLASSES = [
    "apple","pear","peach","plum","apricot","cherry",
    "strawberry","raspberry","blueberry","blackberry",
    "grape","watermelon","melon","banana","mango","papaya",
    "pineapple chunk","kiwi","orange slice","lemon slice",
    "tomato","cherry tomato","carrot piece","broccoli floret",
    "cauliflower","lettuce piece","cucumber slice","zucchini",
    "eggplant","bell pepper","mushroom","onion piece","potato chunk",
    "chicken piece","beef piece","pork piece","meatball","nugget",
    "shrimp","fish piece","boiled egg","tofu cube",
    "pasta piece","dumpling","rice ball","bread piece","olive",
    "food piece","fruit piece","vegetable piece","meat piece",
]

COLORES = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,255,0),(100,255,100),(255,150,50),(50,200,255),(200,50,255),
]


class FoodDetection:
    def __init__(self, label, confidence, center_norm, depth_cm):
        self.label       = label
        self.confidence  = confidence
        self.center_norm = center_norm   # (cx, cy) normalizados en [0,1]
        self.depth_cm    = depth_cm
        self.depth_norm  = float(np.clip(
            (depth_cm - DEPTH_MIN_CM) / (DEPTH_MAX_CM - DEPTH_MIN_CM), 0.0, 1.0
        ))


class VisionPipeline:

    def __init__(self, simulate=False):
        self.simulate   = simulate
        self._cap       = None
        self._yolo      = None
        self._depth_pipe = None
        self._frame_n   = 0

        # Estado del mapa de profundidad (actualizado en hilo aparte)
        self._depth_state = {"map": None, "visual": None, "processing": False}
        self._depth_lock  = threading.Lock()

    def __enter__(self):
        if not self.simulate:
            import cv2
            from ultralytics import YOLOWorld
            from transformers import pipeline as hf_pipeline

            # Camara
            self._cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self._cap.isOpened():
                raise RuntimeError(f"No se pudo abrir la camara {CAMERA_INDEX}.")
            print(f"Camara {CAMERA_INDEX} abierta.")

            # YOLO-World
            print("Cargando YOLOWorld... (puede tardar unos segundos)")
            self._yolo = YOLOWorld("yolov8m-world.pt")
            self._yolo.set_classes(YOLO_CLASSES)
            print("YOLOWorld listo.")

            # Depth Anything V2
            print("Cargando Depth Anything V2... (puede tardar unos segundos)")
            self._depth_pipe = hf_pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Base-hf",
                device=0 if DEVICE == "cuda" else -1,
            )
            print(f"Depth Anything listo. Dispositivo: {DEVICE.upper()}")
        else:
            print("Vision [SIMULADA] inicializada.")
        return self

    def __exit__(self, *_):
        if self._cap:
            self._cap.release()

    # ── Hilo de profundidad asíncrono ─────────────────────────
    def _run_depth_async(self, frame_rgb):
        import cv2
        from PIL import Image
        pil    = Image.fromarray(frame_rgb)
        out    = self._depth_pipe(pil)
        d      = np.array(out["depth"], dtype=np.float32)
        p_low  = np.percentile(d, 2)
        p_high = np.percentile(d, 98)
        d_clip = np.clip(d, p_low, p_high)
        norm   = ((d_clip - p_low) / (p_high - p_low + 1e-6) * 255).astype(np.uint8)
        vis    = cv2.applyColorMap(norm, DEPTH_COLORMAP)
        with self._depth_lock:
            self._depth_state["map"]        = d
            self._depth_state["visual"]     = vis
            self._depth_state["processing"] = False

    def _depth_at_bbox(self, d_map, x1, y1, x2, y2):
        """
        Devuelve la profundidad en cm usando la formula lineal calibrada:
            cm_real = DEPTH_CALIB_A * raw_value + DEPTH_CALIB_B
        """
        if d_map is None:
            return (DEPTH_MIN_CM + DEPTH_MAX_CM) / 2.0
        h, w = d_map.shape
        roi  = d_map[max(0, int(y1)):min(h, int(y2)),
                     max(0, int(x1)):min(w, int(x2))]
        if roi.size == 0:
            return (DEPTH_MIN_CM + DEPTH_MAX_CM) / 2.0
        raw_val    = float(np.mean(roi))
        depth_real = DEPTH_CALIB_A * raw_val + DEPTH_CALIB_B
        return float(np.clip(depth_real, DEPTH_MIN_CM, DEPTH_MAX_CM))

    # ── Interfaz pública ──────────────────────────────────────

    def read_frame(self):
        if self.simulate or self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def get_target_food(self, frame):
        import cv2

        # Modo simulacion
        if self.simulate or frame is None:
            cx   = float(np.random.uniform(0.2, 0.8))
            cy   = float(np.random.uniform(0.2, 0.8))
            dept = float(np.random.uniform(DEPTH_MIN_CM + 5, DEPTH_MAX_CM - 5))
            return (FoodDetection("tomate", 0.91, (cx, cy), dept),
                    np.zeros((480, 640, 3), dtype=np.uint8))

        h, w = frame.shape[:2]
        self._frame_n += 1

        # Lanzar depth cada N frames si no hay uno corriendo ya
        if (self._frame_n % DEPTH_EVERY_N == 0
                and not self._depth_state["processing"]
                and self._depth_pipe is not None):
            self._depth_state["processing"] = True
            rgb_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            threading.Thread(
                target=self._run_depth_async,
                args=(rgb_copy,),
                daemon=True,
            ).start()

        # Detección YOLO
        results    = self._yolo.predict(frame, conf=YOLO_CONF,
                                        iou=0.3, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            detections.append({
                "label":  "Trozo_Comida",
                "bbox":   (x1, y1, x2, y2),
                "cx":     (x1 + x2) / 2,
                "cy":     (y1 + y2) / 2,
                "conf":   float(box.conf[0]),
                "color":  COLORES[cls % len(COLORES)],
            })
        detections.sort(key=lambda x: x["conf"], reverse=True)

        # Mapa de profundidad actual
        with self._depth_lock:
            d_map = (self._depth_state["map"].copy()
                     if self._depth_state["map"] is not None else None)
            d_vis = (self._depth_state["visual"].copy()
                     if self._depth_state["visual"] is not None else None)

        # Superponer mapa de profundidad en el frame anotado
        annotated = frame.copy()
        if d_vis is not None:
            annotated = cv2.addWeighted(
                annotated, 0.55,
                cv2.resize(d_vis, (w, h)), 0.45, 0
            )

        # Dibujar todas las detecciones
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cx, cy  = int(det["cx"]), int(det["cy"])
            color   = det["color"]
            depth_v = self._depth_at_bbox(d_map, x1, y1, x2, y2)
            tag     = f"{det['label']} {det['conf']:.2f}  d:{depth_v:.1f}cm"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 1)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(annotated, tag, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Marcar el objetivo principal (mayor confianza)
        if detections:
            best   = detections[0]
            x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
            cv2.rectangle(annotated, (x1-3, y1-3), (x2+3, y2+3), (0,255,255), 3)
            cv2.putText(annotated, "OBJETIVO", (x1, y2+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cx_norm  = best["cx"] / w
            cy_norm  = best["cy"] / h
            depth_cm = self._depth_at_bbox(d_map, *best["bbox"])
            food     = FoodDetection(best["label"], best["conf"],
                                     (cx_norm, cy_norm), depth_cm)
        else:
            food = None

        cv2.putText(annotated, f"{DEVICE.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return food, annotated

    def food_to_state_vector(self, food, joint_positions):
        if food is None:
            obs = np.zeros(3, dtype=np.float32)
        else:
            obs = np.array([food.center_norm[0],
                            food.center_norm[1],
                            food.depth_norm], dtype=np.float32)
        return np.concatenate([obs, joint_positions]).astype(np.float32)


# ─────────────────────────────────────────────
#  HILO DE CAMARA
# ─────────────────────────────────────────────
class CameraThread(threading.Thread):

    def __init__(self, vision):
        super().__init__(daemon=True)
        self.vision    = vision
        self.active    = False
        self._stop_ev  = threading.Event()
        self._lock     = threading.Lock()
        self.last_food = None

    def run(self):
        while not self._stop_ev.is_set():
            if not self.active:
                time.sleep(0.08)
                continue
            frame = self.vision.read_frame()
            if frame is None:
                food, _ = self.vision.get_target_food(None)
                with self._lock:
                    self.last_food = food
                time.sleep(0.1)
                continue
            food, annotated = self.vision.get_target_food(frame)
            with self._lock:
                self.last_food = food
            try:
                import cv2
                cv2.imshow("Camara", annotated)
                if cv2.waitKey(30) & 0xFF == 27:
                    self.active = False
                    cv2.destroyAllWindows()
            except Exception:
                pass

    def enable(self):
        self.active = True

    def disable(self):
        self.active = False
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    def get_snapshot(self):
        with self._lock:
            return self.last_food

    def stop(self):
        self._stop_ev.set()
        self.disable()


# ─────────────────────────────────────────────
#  COLECTOR DE DEMOS
# ─────────────────────────────────────────────
class DemoCollector:

    def __init__(self, robot, vision, cam):
        self.robot  = robot
        self.vision = vision
        self.cam    = cam

        self._recording    = False
        self._food_target  = None
        self._cam_captured = False
        self._current_ep   = []

        Path(DEMO_DIR).mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        if os.path.exists(DEMO_FILE):
            with open(DEMO_FILE, "rb") as f:
                d = pickle.load(f)
            print(f"Dataset cargado: {len(d.get('episodes', []))} episodios previos.")
            return d
        return {"episodes": [], "created": datetime.now().isoformat()}

    def _save(self):
        with open(DEMO_FILE, "wb") as f:
            pickle.dump(self.data, f)

    def _ep_count(self):
        return len(self.data.get("episodes", []))

    def _get_state(self):
        return self.vision.food_to_state_vector(
            self._food_target, self.robot.get_joint_positions()
        )

    def _pedir_pasos(self):
        """Pide la cantidad de pasos al usuario."""
        try:
            raw = input("  Pasos: ").strip()
            v = int(raw)
            return v   # puede ser negativo si el usuario escribe -200
        except (ValueError, EOFError):
            print("  Valor invalido, usando 0.")
            return 0

    def _reset(self):
        self._recording    = False
        self._food_target  = None
        self._cam_captured = False
        self._current_ep   = []

    # ── Comandos ──────────────────────────────

    def cmd_cam(self):
        if not self.cam.active:
            self.cam.enable()
            print("Camara activa. Apunta al plato y escribe CAM de nuevo.")
            return
        time.sleep(0.6)
        food = self.cam.get_snapshot()
        if food is None:
            print("Sin deteccion. Verifica la posicion y escribe CAM otra vez.")
            return
        self._food_target  = food
        self._cam_captured = True
        print(f"Alimento:    {food.label}  confianza: {food.confidence:.2f}")
        print(f"Posicion 2D: cx={food.center_norm[0]:.4f}  cy={food.center_norm[1]:.4f}")
        print(f"Profundidad: {food.depth_cm:.2f} cm  (norm={food.depth_norm:.4f})")
        print("Escribe I cuando el brazo este fisicamente en HOME.")

    def cmd_iniciar(self):
        if self._recording:
            print("Ya hay una grabacion activa.")
            return
        # El brazo YA esta fisicamente en HOME.
        # Solo sincronizamos el contador — no enviamos nada al Arduino.
        self.robot._positions = dict(HOME_POSITION)
        self._recording  = True
        self._current_ep = []
        print(f"Grabacion iniciada. Episodio {self._ep_count() + 1}.")
        print(f"Posicion HOME asumida: {HOME_POSITION}")
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

        episode = {
            "food_target": {
                "label":       self._food_target.label,
                "confidence":  self._food_target.confidence,
                "cx_norm":     self._food_target.center_norm[0],
                "cy_norm":     self._food_target.center_norm[1],
                "depth_cm":    self._food_target.depth_cm,
                "depth_norm":  self._food_target.depth_norm,
                "location_3d": [
                    float(self._food_target.center_norm[0]),
                    float(self._food_target.center_norm[1]),
                    float(self._food_target.depth_norm),
                ],
            },
            "steps":     list(self._current_ep),
            "success":   success,
            "timestamp": datetime.now().isoformat(),
            "n_steps":   n,
        }
        self.data.setdefault("episodes", []).append(episode)
        self._save()
        print(f"Episodio {self._ep_count()} guardado. ({n} pasos, exito={success})")

        # Regresar a HOME automaticamente
        self.robot.go_home()
        self._reset()

    def cmd_cancelar(self):
        n = len(self._current_ep)
        if self._recording:
            print("Regresando a HOME antes de cancelar...")
            self.robot.go_home()
        self._reset()
        print(f"Episodio cancelado. {n} pasos descartados.")

    def cmd_mover(self, cmd):
        axis, direction, action_idx = COMMAND_MAP[cmd]

        # El gripper es un servo — no usa pasos, va directo
        if axis in ("gripper_close", "gripper_open"):
            state_before = self._get_state()
            if axis == "gripper_close":
                self.robot.close_gripper()
                print(f"[{'REC' if self._recording else 'libre'}] GRIP+  servo -> CERRAR (0 grados)")
            else:
                self.robot.open_gripper()
                print(f"[{'REC' if self._recording else 'libre'}] GRIP-  servo -> ABRIR (90 grados)")
            if self._recording and action_idx >= 0:
                self._current_ep.append({
                    "state":          state_before.copy(),
                    "action":         action_idx,
                    "cmd":            cmd,
                    "steps_executed": 0,
                })
            return

        # Motores paso a paso — pedir cantidad de pasos
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

    def cmd_status(self):
        pos  = self.robot.get_raw_positions()
        norm = self.robot.get_joint_positions()
        axes = ["base", "hombro", "codo", "muneca", "rotacion"]
        print("─" * 52)
        for ax, nv in zip(axes, norm):
            home_val = HOME_POSITION[ax]
            diff     = pos.get(ax, 0) - home_val
            print(f"  {ax:<10} actual={pos.get(ax,0):+6d}  "
                  f"home={home_val:+6d}  diff={diff:+6d}  norm={nv:+.3f}")
        print(f"  Grabando:  {self._recording}")
        print(f"  Episodios: {self._ep_count()}")
        if self._food_target:
            ft = self._food_target
            print(f"  Objetivo:  {ft.label} @ {ft.depth_cm:.1f} cm  "
                  f"cx={ft.center_norm[0]:.3f} cy={ft.center_norm[1]:.3f}")
        print("─" * 52)

    # ── Bucle principal ───────────────────────

    def run(self):
        print("\nComandos disponibles:")
        print("  CAM          — detectar alimento con la camara")
        print("  I            — brazo esta en HOME, iniciar grabacion")
        print("  F            — finalizar, guardar y REGRESAR A HOME")
        print("  CANCEL       — cancelar episodio y regresar a HOME")
        print("  STATUS       — posicion de articulaciones vs HOME")
        print("  B+/B-        — base        (pedira pasos)")
        print("  H+/H-        — hombro      (pedira pasos)")
        print("  C+/C-        — codo        (pedira pasos)")
        print("  M+/M-        — muneca/gripper  (pedira pasos)")
        print("  G+/G-        — rotacion/giro   (pedira pasos)")
        print("  GRIP+/GRIP-  — abrir/cerrar pinza")
        print("  EXIT         — salir\n")
        print(f"HOME = {HOME_POSITION}\n")

        while True:
            ep  = self._ep_count() + 1
            rec = "REC" if self._recording else "---"
            try:
                raw = input(f"EP{ep} [{rec}] > ").strip().upper()
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
            elif raw in COMMAND_MAP:
                self.cmd_mover(raw)
            else:
                print(f"Comando no reconocido: '{raw}'")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true",
                        help="Modo simulacion (sin Arduino ni camara)")
    args = parser.parse_args()

    with RobotInterface(simulate=args.sim) as robot:
        vision = VisionPipeline(simulate=args.sim)
        with vision:
            cam = CameraThread(vision)
            cam.start()
            collector = DemoCollector(robot, vision, cam)
            try:
                collector.run()
            finally:
                cam.stop()