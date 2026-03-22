"""
=============================================================
  PIPELINE v5 - Brazo Robótico Asistente
  YOLO-World + Depth Anything V2 (vitb)
  SIN MediaPipe

  RTX 4070 Laptop | i7-13620H | Python 3.11 Windows

  Instalación:
    pip install ultralytics
    pip install transformers pillow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python numpy
=============================================================
"""

import cv2
import numpy as np
import threading
import time
import torch
from ultralytics import YOLOWorld
from transformers import pipeline as hf_pipeline

# ──────────────────────────────────────────────
#  CONFIGURACIÓN
# ──────────────────────────────────────────────
CAMARA_INDEX  = 0
FRAME_W       = 1280
FRAME_H       = 720
YOLO_CONF     = 0.35
DEPTH_EVERY_N = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🔧 Dispositivo: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")

COLORES = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,255,0),(100,255,100),(255,150,50),(50,200,255),(200,50,255),
]

# ──────────────────────────────────────────────
#  CARGAR MODELOS
# ──────────────────────────────────────────────
print("📦 Cargando modelos...\n")

# 1. YOLO-World
print("  [1/2] YOLO-World m...")
yolo = YOLOWorld('yolov8m-world.pt')
yolo.set_classes([
    # Frutas
    "apple","pear","banana","grape","strawberry","watermelon",
    "orange slice","mango","kiwi","peach","cherry",
    # Verduras
    "tomato","cherry tomato","carrot piece","broccoli floret",
    "cucumber slice","bell pepper","lettuce piece","mushroom",
    "potato chunk","corn kernel","spinach",
    # Proteínas
    "chicken piece","beef piece","pork piece","meatball","nugget",
    "shrimp","fish piece","boiled egg","tofu cube",
    # Carbohidratos
    "pasta piece","rice ball","bread piece","dumpling",
    # Genéricos
    "food piece","fruit piece","vegetable piece","meat piece"
])
print("        ✅ YOLO-World listo")

# 2. Depth Anything V2 - Base
print("  [2/2] Depth Anything V2 - Base (vitb)...")
print("        (primera vez: descarga ~400MB...)")
depth_pipe = hf_pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Base-hf",
    device=0 if DEVICE == "cuda" else -1
)
print("        ✅ Depth Anything V2 listo\n")
print("🚀 Modelos listos. Abriendo cámara...\n")

# ──────────────────────────────────────────────
#  ESTADO DEPTH (hilo asíncrono)
# ──────────────────────────────────────────────
depth_state = {"map": None, "visual": None, "processing": False}
depth_lock  = threading.Lock()

def run_depth_async(frame_rgb):
    from PIL import Image
    pil = Image.fromarray(frame_rgb)
    out = depth_pipe(pil)
    d   = np.array(out["depth"], dtype=np.float32)
    d_min, d_max = d.min(), d.max()
    norm = ((d - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
    vis  = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    with depth_lock:
        depth_state["map"]        = d
        depth_state["visual"]     = vis
        depth_state["processing"] = False

def depth_at_bbox(d_map, x1, y1, x2, y2):
    if d_map is None:
        return None
    h, w = d_map.shape
    roi = d_map[max(0,int(y1)):min(h,int(y2)),
                max(0,int(x1)):min(w,int(x2))]
    return float(np.mean(roi)) if roi.size > 0 else None

# ──────────────────────────────────────────────
#  ABRIR CÁMARA
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"❌ Cámara {CAMARA_INDEX} no disponible. Prueba con 0 o 1.")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

show_depth = True
show_yolo  = True
frame_n    = 0
fps_t      = time.time()
fps_cnt    = 0
fps_val    = 0.0

print("🎥 Pipeline activo.")
print("   Teclas: [Q] Salir | [D] Depth | [Y] YOLO | [S] Screenshot\n")

# ──────────────────────────────────────────────
#  LOOP PRINCIPAL
# ──────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_n += 1
    fps_cnt += 1
    h, w = frame.shape[:2]

    # FPS
    elapsed = time.time() - fps_t
    if elapsed >= 1.0:
        fps_val = fps_cnt / elapsed
        fps_cnt = 0
        fps_t   = time.time()

    # ── Depth asíncrono ───────────────────────
    if show_depth and frame_n % DEPTH_EVERY_N == 0 and not depth_state["processing"]:
        depth_state["processing"] = True
        rgb_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        threading.Thread(target=run_depth_async, args=(rgb_copy,), daemon=True).start()

    # ── YOLO-World ────────────────────────────
    alimentos = []
    if show_yolo:
        results = yolo.predict(frame, conf=YOLO_CONF, iou=0.3, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            alimentos.append({
                'clase':  results.names[cls],
                'bbox':   (int(x1), int(y1), int(x2), int(y2)),
                'centro': (int((x1+x2)/2), int((y1+y2)/2)),
                'conf':   float(box.conf[0]),
                'color':  COLORES[cls % len(COLORES)]
            })
        alimentos.sort(key=lambda x: x['conf'], reverse=True)

    # ── Composición visual ────────────────────
    with depth_lock:
        d_vis = depth_state["visual"].copy() if depth_state["visual"] is not None else None
        d_map = depth_state["map"].copy()    if depth_state["map"]    is not None else None

    # Overlay depth
    if show_depth and d_vis is not None:
        frame = cv2.addWeighted(frame, 0.65, cv2.resize(d_vis, (w, h)), 0.35, 0)

    # BBoxes YOLO
    for a in alimentos:
        x1, y1, x2, y2 = a['bbox']
        cx, cy = a['centro']
        color  = a['color']
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.circle(frame, (cx,cy), 6, (0,0,255), -1)
        cv2.circle(frame, (cx,cy), 6, (255,255,255), 1)
        dv  = depth_at_bbox(d_map, x1, y1, x2, y2)
        tag = f"{a['clase']} {a['conf']:.2f}" + (f"  d:{dv:.2f}" if dv else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, tag, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # Objetivo principal (mayor confianza)
    if alimentos:
        x1, y1, x2, y2 = alimentos[0]['bbox']
        cv2.rectangle(frame, (x1-3,y1-3), (x2+3,y2+3), (0,255,255), 3)
        cv2.putText(frame, "OBJETIVO", (x1, y2+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        dv = depth_at_bbox(d_map, *alimentos[0]['bbox'])
        print(f"\r🎯 {alimentos[0]['clase']:<25} "
              f"conf:{alimentos[0]['conf']:.2f}  "
              f"centro:{alimentos[0]['centro']}  "
              f"d:{f'{dv:.2f}' if dv else 'N/A'}   ",
              end="", flush=True)

    # HUD
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,80), (15,15,15), -1)
    frame = cv2.addWeighted(frame, 0.6, ov, 0.4, 0)
    cv2.putText(frame, f"FPS: {fps_val:.1f}  |  {DEVICE.upper()}",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame,
                f"YOLO: {'ON' if show_yolo else 'OFF'} ({len(alimentos)} obj)",
                (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0,255,100) if show_yolo else (100,100,100), 1)
    cv2.putText(frame,
                f"Depth: {'ON' if show_depth else 'OFF'}"
                f"{'  [procesando...]' if depth_state['processing'] else ''}",
                (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (100,200,255) if show_depth else (100,100,100), 1)
    cv2.putText(frame, "Q:Salir | D:Depth | Y:YOLO | S:Screenshot",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

    cv2.imshow("Brazo Robotico | YOLO-World + Depth Anything V2", frame)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'): break
    elif key == ord('d'):
        show_depth = not show_depth
        print(f"\n  🔵 Depth: {'ON' if show_depth else 'OFF'}")
    elif key == ord('y'):
        show_yolo = not show_yolo
        print(f"\n  🟢 YOLO: {'ON' if show_yolo else 'OFF'}")
    elif key == ord('s'):
        fn = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(fn, frame)
        print(f"\n  📸 Guardado: {fn}")

cap.release()
cv2.destroyAllWindows()
print("\n\n✅ Pipeline cerrado.")