import cv2
import numpy as np
import threading
import time
import torch
from ultralytics import YOLOWorld
from transformers import pipeline as hf_pipeline

CAMARA_INDEX   = 2
FRAME_W        = 1280
FRAME_H        = 720
YOLO_CONF      = 0.35
DEPTH_EVERY_N  = 6
DEPTH_COLORMAP = cv2.COLORMAP_TURBO
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

COLORES = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,255,0),(100,255,100),(255,150,50),(50,200,255),(200,50,255),
]

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

depth_pipe = hf_pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Base-hf",
    device=0 if DEVICE == "cuda" else -1
)

depth_state = {"map": None, "visual": None, "processing": False}
depth_lock  = threading.Lock()

def run_depth_async(frame_rgb):
    from PIL import Image
    pil    = Image.fromarray(frame_rgb)
    out    = depth_pipe(pil)
    d      = np.array(out["depth"], dtype=np.float32)
    p_low  = np.percentile(d, 2)
    p_high = np.percentile(d, 98)
    d_clip = np.clip(d, p_low, p_high)
    norm   = ((d_clip - p_low) / (p_high - p_low + 1e-6) * 255).astype(np.uint8)
    vis    = cv2.applyColorMap(norm, DEPTH_COLORMAP)
    with depth_lock:
        depth_state["map"]        = d
        depth_state["visual"]     = vis
        depth_state["processing"] = False

def depth_at_bbox(d_map, x1, y1, x2, y2):
    if d_map is None:
        return None
    h, w = d_map.shape
    roi  = d_map[max(0,int(y1)):min(h,int(y2)), max(0,int(x1)):min(w,int(x2))]
    return float(np.mean(roi)) if roi.size > 0 else None

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

frame_n = 0
fps_t   = time.time()
fps_cnt = 0
fps_val = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_n += 1
    fps_cnt += 1
    h, w = frame.shape[:2]

    elapsed = time.time() - fps_t
    if elapsed >= 1.0:
        fps_val = fps_cnt / elapsed
        fps_cnt = 0
        fps_t   = time.time()

    if frame_n % DEPTH_EVERY_N == 0 and not depth_state["processing"]:
        depth_state["processing"] = True
        rgb_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        threading.Thread(target=run_depth_async, args=(rgb_copy,), daemon=True).start()

    results   = yolo.predict(frame, conf=YOLO_CONF, iou=0.3, verbose=False)[0]
    alimentos = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        alimentos.append({
            'clase':  "TROZO DE COMIDA",
            'bbox':   (int(x1), int(y1), int(x2), int(y2)),
            'centro': (int((x1+x2)/2), int((y1+y2)/2)),
            'conf':   float(box.conf[0]),
            'color':  COLORES[cls % len(COLORES)]
        })
    alimentos.sort(key=lambda x: x['conf'], reverse=True)

    with depth_lock:
        d_vis = depth_state["visual"].copy() if depth_state["visual"] is not None else None
        d_map = depth_state["map"].copy()    if depth_state["map"]    is not None else None

    if d_vis is not None:
        frame = cv2.addWeighted(frame, 0.55, cv2.resize(d_vis, (w, h)), 0.45, 0)

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
        cv2.putText(frame, tag, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    if alimentos:
        x1, y1, x2, y2 = alimentos[0]['bbox']
        cv2.rectangle(frame, (x1-3,y1-3), (x2+3,y2+3), (0,255,255), 3)
        cv2.putText(frame, "OBJETIVO", (x1, y2+22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

    cv2.putText(frame, f"FPS: {fps_val:.1f}  {DEVICE.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Brazo Robotico", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
