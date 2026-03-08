"""
=============================================================
  PIPELINE v5 - Brazo Robótico Asistente
  YOLO-World + Depth Anything V2 (vitb)
  SIN MediaPipe

  Ventanas separadas: YOLO y Depth alternan cada 5 segundos.
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
CAMARA_INDEX    = 2
FRAME_W         = 1280
FRAME_H         = 720
YOLO_CONF       = 0.35
DEPTH_EVERY_N   = 6
SWITCH_INTERVAL = 5.0   # segundos entre cambio de ventana activa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🔧 Dispositivo: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")

COLORES = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,255,0),(100,255,100),(255,150,50),(50,200,255),(200,50,255),
]

# Nombres de ventanas
WIN_YOLO  = "Brazo Robotico | YOLO-World"
WIN_DEPTH = "Brazo Robotico | Depth Anything V2"

# ──────────────────────────────────────────────
#  CARGAR MODELOS
# ──────────────────────────────────────────────
print("📦 Cargando modelos...\n")

print("  [1/2] YOLO-World m...")
yolo = YOLOWorld('yolov8m-world.pt')
yolo.set_classes([
    "apple","pear","banana","grape","strawberry","watermelon",
    "orange slice","mango","kiwi","peach","cherry",
    "tomato","cherry tomato","carrot piece","broccoli floret",
    "cucumber slice","bell pepper","lettuce piece","mushroom",
    "potato chunk","corn kernel","spinach",
    "chicken piece","beef piece","pork piece","meatball","nugget",
    "shrimp","fish piece","boiled egg","tofu cube",
    "pasta piece","rice ball","bread piece","dumpling",
    "food piece","fruit piece","vegetable piece","meat piece"
])
print("        ✅ YOLO-World listo")

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

# ──────────────────────────────────────────────
#  ESTADO DE ALTERNANCIA
# ──────────────────────────────────────────────
# Modos: "yolo" o "depth"
mode          = "yolo"      # empieza con YOLO
mode_manual   = False       # True = usuario tomó control manual
last_switch   = time.time()

frame_n   = 0
fps_t     = time.time()
fps_cnt   = 0
fps_val   = 0.0

# Crear ambas ventanas desde el inicio
cv2.namedWindow(WIN_YOLO,  cv2.WINDOW_NORMAL)
cv2.namedWindow(WIN_DEPTH, cv2.WINDOW_NORMAL)

print("🎥 Pipeline activo.")
print("   Ventanas alternan automáticamente cada 5 segundos.")
print("   Teclas: [Q] Salir | [Y] Forzar YOLO | [D] Forzar Depth")
print("          [A] Volver a alternancia automática | [S] Screenshot\n")

# ──────────────────────────────────────────────
#  HELPERS PARA DIBUJAR MARCOS
# ──────────────────────────────────────────────
def dibujar_marco_activo(frame_in, color_rgb):
    """Dibuja un borde de color en la ventana activa."""
    out = frame_in.copy()
    grosor = 8
    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w-1, h-1), color_rgb, grosor)
    return out

def dibujar_hud(frame_in, texto_modo, tiempo_restante, fps):
    """HUD superior con modo, tiempo restante y FPS."""
    out  = frame_in.copy()
    h, w = out.shape[:2]
    ov   = out.copy()
    cv2.rectangle(ov, (0,0), (w, 85), (15,15,15), -1)
    out  = cv2.addWeighted(out, 0.55, ov, 0.45, 0)
    cv2.putText(out, f"FPS: {fps:.1f}  |  {DEVICE.upper()}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(out, texto_modo,
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,200), 2)
    cv2.putText(out, f"Siguiente cambio en: {tiempo_restante:.1f}s" if not mode_manual else "Modo MANUAL (A para auto)",
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,100), 1)
    cv2.putText(out, "Q:Salir | Y:YOLO | D:Depth | A:Auto | S:Screenshot",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)
    return out

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

    # ── Alternancia automática cada SWITCH_INTERVAL segundos ──
    tiempo_transcurrido = time.time() - last_switch
    tiempo_restante     = max(0.0, SWITCH_INTERVAL - tiempo_transcurrido)

    if not mode_manual and tiempo_transcurrido >= SWITCH_INTERVAL:
        mode       = "depth" if mode == "yolo" else "yolo"
        last_switch = time.time()
        tiempo_restante = SWITCH_INTERVAL
        print(f"\n  🔄 Cambio automático → {mode.upper()}")

    # ── Depth asíncrono (siempre procesando en background) ────
    if frame_n % DEPTH_EVERY_N == 0 and not depth_state["processing"]:
        depth_state["processing"] = True
        rgb_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        threading.Thread(target=run_depth_async, args=(rgb_copy,), daemon=True).start()

    with depth_lock:
        d_vis = depth_state["visual"].copy() if depth_state["visual"] is not None else None
        d_map = depth_state["map"].copy()    if depth_state["map"]    is not None else None

    # ══════════════════════════════════════════
    #  VENTANA YOLO
    # ══════════════════════════════════════════
    frame_yolo = frame.copy()

    # Detectar y dibujar objetos
    alimentos = []
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

    for a in alimentos:
        x1, y1, x2, y2 = a['bbox']
        cx, cy = a['centro']
        color  = a['color']
        cv2.rectangle(frame_yolo, (x1,y1), (x2,y2), color, 2)
        cv2.circle(frame_yolo, (cx,cy), 6, (0,0,255), -1)
        cv2.circle(frame_yolo, (cx,cy), 6, (255,255,255), 1)
        dv  = depth_at_bbox(d_map, x1, y1, x2, y2)
        tag = f"{a['clase']} {a['conf']:.2f}" + (f"  d:{dv:.2f}" if dv else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame_yolo, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(frame_yolo, tag, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    if alimentos:
        x1, y1, x2, y2 = alimentos[0]['bbox']
        cv2.rectangle(frame_yolo, (x1-3,y1-3), (x2+3,y2+3), (0,255,255), 3)
        cv2.putText(frame_yolo, "OBJETIVO", (x1, y2+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        dv = depth_at_bbox(d_map, *alimentos[0]['bbox'])
        print(f"\r🎯 {alimentos[0]['clase']:<25} "
              f"conf:{alimentos[0]['conf']:.2f}  "
              f"centro:{alimentos[0]['centro']}  "
              f"d:{f'{dv:.2f}' if dv else 'N/A'}   ",
              end="", flush=True)

    # Marco verde si YOLO es el activo, gris si no
    if mode == "yolo":
        frame_yolo = dibujar_marco_activo(frame_yolo, (0, 255, 100))
        frame_yolo = dibujar_hud(frame_yolo, "▶  YOLO-World ACTIVO", tiempo_restante, fps_val)
    else:
        # Ventana inactiva: oscurecer un poco y poner label
        overlay = frame_yolo.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        frame_yolo = cv2.addWeighted(frame_yolo, 0.45, overlay, 0.55, 0)
        cv2.putText(frame_yolo, "YOLO-World  [inactivo]",
                    (w//2 - 160, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (120,120,120), 2)

    # ══════════════════════════════════════════
    #  VENTANA DEPTH
    # ══════════════════════════════════════════
    if d_vis is not None:
        frame_depth = cv2.addWeighted(frame, 0.4, cv2.resize(d_vis, (w,h)), 0.6, 0)
    else:
        # Mientras no hay depth todavía
        frame_depth = frame.copy()
        cv2.putText(frame_depth, "Calculando profundidad...",
                    (w//2 - 220, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (100,200,255), 2)

    if mode == "depth":
        frame_depth = dibujar_marco_activo(frame_depth, (100, 150, 255))
        frame_depth = dibujar_hud(frame_depth, "▶  Depth Anything V2 ACTIVO", tiempo_restante, fps_val)
    else:
        overlay = frame_depth.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        frame_depth = cv2.addWeighted(frame_depth, 0.45, overlay, 0.55, 0)
        cv2.putText(frame_depth, "Depth Anything V2  [inactivo]",
                    (w//2 - 230, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (120,120,120), 2)

    # ── Mostrar ventanas ──────────────────────
    cv2.imshow(WIN_YOLO,  frame_yolo)
    cv2.imshow(WIN_DEPTH, frame_depth)

    # ── Teclas ───────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('y'):
        mode       = "yolo"
        mode_manual = True
        last_switch = time.time()
        print("\n  🟢 Forzado: YOLO")
    elif key == ord('d'):
        mode       = "depth"
        mode_manual = True
        last_switch = time.time()
        print("\n  🔵 Forzado: Depth")
    elif key == ord('a'):
        mode_manual = False
        last_switch = time.time()
        print("\n  🔄 Alternancia automática activada")
    elif key == ord('s'):
        fn_y = f"yolo_{int(time.time())}.png"
        fn_d = f"depth_{int(time.time())}.png"
        cv2.imwrite(fn_y, frame_yolo)
        cv2.imwrite(fn_d, frame_depth)
        print(f"\n  📸 Guardados: {fn_y}  y  {fn_d}")

cap.release()
cv2.destroyAllWindows()
print("\n\n✅ Pipeline cerrado.")