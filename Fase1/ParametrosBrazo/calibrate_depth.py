"""
SCRIPT 1 — Calibración Depth Anything (raw → centímetros)

USO:
    python calibrate_depth.py

QUÉ HACE:
    Abre la cámara y muestra en tiempo real el valor raw de Depth Anything
    en el centro del frame. Tú colocas un objeto a distancias conocidas
    (10, 15, 20, 30, 40, 50 cm) y presionas SPACE para registrar cada punto.
    Al final imprime las listas listas para pegar en config.py.

INSTRUCCIONES:
    1. Coloca una caja o libro plano a exactamente 10 cm del lente.
    2. Espera que el valor en pantalla se estabilice.
    3. Presiona SPACE para registrar.
    4. Repite para 15, 20, 30, 40, 50 cm.
    5. Presiona ESC para terminar y ver los resultados.
"""

import cv2
import numpy as np
import torch
import threading
import time
from PIL import Image
from transformers import pipeline as hf_pipeline

CAMERA_INDEX   = 0
FRAME_W        = 640
FRAME_H        = 480
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DISTANCES_CM   = [10, 15, 20, 30, 40, 50]
SAMPLE_RADIUS  = 40

print(f"Cargando Depth Anything ({DEVICE})...")
depth_pipe = hf_pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Base-hf",
    device=0 if DEVICE == "cuda" else -1,
)
print("Listo.")

depth_state = {"raw_map": None, "processing": False}
depth_lock  = threading.Lock()

def depth_worker(rgb):
    pil = Image.fromarray(rgb)
    out = depth_pipe(pil)
    d   = np.array(out["depth"], dtype=np.float32)
    with depth_lock:
        depth_state["raw_map"]    = d
        depth_state["processing"] = False

def get_center_raw(frame_w, frame_h):
    with depth_lock:
        d = depth_state["raw_map"]
    if d is None:
        return None
    h, w = d.shape
    cx, cy = w // 2, h // 2
    r = SAMPLE_RADIUS
    roi = d[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
    return float(np.mean(roi)) if roi.size > 0 else None

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

recorded_raw = []
recorded_cm  = []
frame_n      = 0
next_dist_idx = 0

print("\nDistancias a medir:", DISTANCES_CM)
print("Presiona SPACE para registrar, ESC para terminar.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_n += 1

    if frame_n % 6 == 0 and not depth_state["processing"]:
        depth_state["processing"] = True
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        threading.Thread(target=depth_worker, args=(rgb,), daemon=True).start()

    raw_val = get_center_raw(FRAME_W, FRAME_H)
    display = frame.copy()
    h, w    = display.shape[:2]
    cx, cy  = w // 2, h // 2

    cv2.rectangle(display,
                  (cx - SAMPLE_RADIUS, cy - SAMPLE_RADIUS),
                  (cx + SAMPLE_RADIUS, cy + SAMPLE_RADIUS),
                  (0, 255, 255), 2)
    cv2.putText(display, "Zona de medicion", (cx - SAMPLE_RADIUS, cy - SAMPLE_RADIUS - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if next_dist_idx < len(DISTANCES_CM):
        target_cm = DISTANCES_CM[next_dist_idx]
        msg = f"Coloca objeto a {target_cm} cm  |  SPACE=registrar"
    else:
        msg = "Todas las distancias registradas  |  ESC=terminar"

    cv2.putText(display, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    raw_str = f"raw actual: {raw_val:.4f}" if raw_val is not None else "raw actual: calculando..."
    cv2.putText(display, raw_str, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i, (r, c) in enumerate(zip(recorded_raw, recorded_cm)):
        cv2.putText(display, f"  {c} cm → raw {r:.4f}", (10, 100 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Calibracion Depth Anything", display)
    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

    if key == ord(' ') and raw_val is not None and next_dist_idx < len(DISTANCES_CM):
        cm = DISTANCES_CM[next_dist_idx]
        recorded_raw.append(round(raw_val, 4))
        recorded_cm.append(float(cm))
        print(f"  Registrado: {cm} cm → raw {raw_val:.4f}")
        next_dist_idx += 1
        time.sleep(0.3)

cap.release()
cv2.destroyAllWindows()

if len(recorded_raw) >= 2:
    print("\n" + "="*55)
    print("RESULTADO — pegar en config.py:")
    print("="*55)
    print(f"DEPTH_CALIB_RAW_POINTS = {recorded_raw}")
    print(f"DEPTH_CALIB_CM_POINTS  = {recorded_cm}")
    print("="*55)
else:
    print("No se registraron suficientes puntos.")
