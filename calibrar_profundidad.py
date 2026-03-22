"""
calibrar_profundidad_click.py

Selecciona manualmente los 5 puntos con el mouse:
1. Centro
2. Esquina sup izq
3. Esquina sup der
4. Esquina inf izq
5. Esquina inf der
"""

import cv2
import numpy as np
import threading
import torch
from transformers import pipeline as hf_pipeline
from PIL import Image

CAMERA_INDEX = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PUNTOS_NOMBRES = [
    "CENTRO del plato",
    "ESQUINA superior izquierda",
    "ESQUINA superior derecha",
    "ESQUINA inferior izquierda",
    "ESQUINA inferior derecha",
]

# ─────────────────────────────────────────────
print("Cargando modelo...")
depth_pipe = hf_pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Base-hf",
    device=0 if DEVICE == "cuda" else -1,
)

depth_state = {"map": None, "processing": False}
depth_lock = threading.Lock()

def run_depth(frame_rgb):
    pil = Image.fromarray(frame_rgb)
    out = depth_pipe(pil)
    d = np.array(out["depth"], dtype=np.float32)
    with depth_lock:
        depth_state["map"] = d
        depth_state["processing"] = False

def depth_en_punto(d_map, x, y, radio=30):
    if d_map is None:
        return None
    h, w = d_map.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), radio, 255, -1)
    vals = d_map[mask == 255]
    return float(np.mean(vals)) if vals.size > 0 else None

def abrir_camara():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def pedir_distancia_real(nombre):
    while True:
        try:
            v = float(input(f"Distancia real a '{nombre}' (cm): "))
            if v > 0:
                return v
        except:
            pass
        print("Valor invalido")

# ─────────────────────────────────────────────
# 📍 SELECCIÓN DE PUNTOS CON MOUSE
# ─────────────────────────────────────────────

puntos_px = []

def click_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos_px) < 5:
        puntos_px.append((x, y))
        print(f"Punto {len(puntos_px)} guardado: {x}, {y}")

cap = abrir_camara()
cv2.namedWindow("Selecciona puntos")
cv2.setMouseCallback("Selecciona puntos", click_mouse)

print("\nHaz clic en los 5 puntos en este orden:")
for i, n in enumerate(PUNTOS_NOMBRES):
    print(f"{i+1}. {n}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # dibujar puntos seleccionados
    for i, (x, y) in enumerate(puntos_px):
        cv2.circle(frame, (x, y), 10, (0,255,0), -1)
        cv2.putText(frame, str(i+1), (x+10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if len(puntos_px) < 5:
        cv2.putText(frame, f"Click punto {len(puntos_px)+1}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    else:
        cv2.putText(frame, "Presiona ESPACIO para continuar",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Selecciona puntos", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and len(puntos_px) == 5:
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 🔁 CALIBRACIÓN
# ─────────────────────────────────────────────

mediciones = []
idx = 0
frame_n = 0
cap = abrir_camara()

while idx < 5:
    ret, frame = cap.read()
    if not ret:
        break

    frame_n += 1

    if frame_n % 4 == 0 and not depth_state["processing"]:
        depth_state["processing"] = True
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        threading.Thread(target=run_depth, args=(rgb,), daemon=True).start()

    with depth_lock:
        d_map = depth_state["map"]

    x, y = puntos_px[idx]
    val = depth_en_punto(d_map, x, y)
    txt = f"{val:.4f}" if val else "..."

    # dibujar punto actual
    cv2.circle(frame, (x, y), 20, (0,255,255), 3)

    cv2.putText(frame, f"Punto {idx+1}: {PUNTOS_NOMBRES[idx]}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Depth: {txt}",
                (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Calibracion", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if val is None:
            print("Esperando depth...")
            continue

        cap.release()
        cv2.destroyAllWindows()

        dist = pedir_distancia_real(PUNTOS_NOMBRES[idx])
        mediciones.append((val, dist))
        idx += 1

        if idx < 5:
            cap = abrir_camara()

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# 📊 REGRESIÓN
# ─────────────────────────────────────────────

raw = np.array([m[0] for m in mediciones])
real = np.array([m[1] for m in mediciones])

A = np.vstack([raw, np.ones(len(raw))]).T
a, b = np.linalg.lstsq(A, real, rcond=None)[0]

print("\nRESULTADO:")
print(f"cm = {a:.6f} * raw + {b:.6f}")