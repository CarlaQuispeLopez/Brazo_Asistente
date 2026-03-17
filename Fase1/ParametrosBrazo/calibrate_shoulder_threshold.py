"""
SCRIPT 3 — Umbral de altura del hombro para activar MediaPipe (fase DELIVERY)

USO:
    python calibrate_shoulder_threshold.py --port COM3
    python calibrate_shoulder_threshold.py --sim

QUÉ HACE:
    Mueve el hombro hacia arriba con W. Muestra en pantalla los pasos
    actuales del hombro. Cuando el gripper esté visualmente a la altura
    de la boca de una persona sentada frente al brazo, presionas SPACE
    para registrar ese valor de pasos.

    También mide la profundidad mínima y máxima alcanzable por el brazo
    extendido y retraído, para REACHABLE_DEPTH_MIN_CM y REACHABLE_DEPTH_MAX_CM.

CONTROLES:
    W / S  → hombro arriba / abajo
    A / D  → base izquierda / derecha
    Q / E  → codo sube / baja
    SPACE  → registrar pasos actuales del hombro como umbral delivery
    M      → registrar profundidad actual (brazo extendido = min, retraído = max)
    H      → HOME
    ESC    → terminar
"""

import cv2
import numpy as np
import argparse
import sys
import os
import threading
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_interface import RobotInterface
from config import ACTION_STEPS

CAMERA_INDEX  = 0
FRAME_W       = 640
FRAME_H       = 480
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RADIUS = 40

depth_state = {"raw_map": None, "processing": False}
depth_lock  = threading.Lock()

def depth_worker(rgb, pipe):
    pil = Image.fromarray(rgb)
    out = pipe(pil)
    d   = np.array(out["depth"], dtype=np.float32)
    with depth_lock:
        depth_state["raw_map"]    = d
        depth_state["processing"] = False

def get_center_raw():
    with depth_lock:
        d = depth_state["raw_map"]
    if d is None:
        return None
    h, w = d.shape
    cx, cy = w // 2, h // 2
    r  = SAMPLE_RADIUS
    roi = d[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
    return float(np.mean(roi)) if roi.size > 0 else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--sim",  action="store_true")
    parser.add_argument("--no_depth", action="store_true",
                        help="Omitir Depth Anything (mas rapido)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    depth_pipe = None
    if not args.no_depth:
        print(f"Cargando Depth Anything ({DEVICE})...")
        depth_pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=0 if DEVICE == "cuda" else -1,
        )
        print("Listo.")

    KEY_MAP = {
        ord('w'): ("hombro",   +ACTION_STEPS),
        ord('s'): ("hombro",   -ACTION_STEPS),
        ord('a'): ("base",     -ACTION_STEPS),
        ord('d'): ("base",     +ACTION_STEPS),
        ord('q'): ("codo",     +ACTION_STEPS),
        ord('e'): ("codo",     -ACTION_STEPS),
    }

    delivery_threshold   = None
    depth_min_raw        = None
    depth_max_raw        = None
    frame_n              = 0

    print("\nInstrucciones:")
    print("  1. Pide a una persona que se siente frente al brazo en posicion normal.")
    print("  2. Mueve el hombro con W hasta que el gripper quede a la altura de su boca.")
    print("  3. Presiona SPACE para registrar ese umbral.")
    print("  4. Extiende el brazo al maximo hacia adelante y presiona M (profundidad minima).")
    print("  5. Retrae el brazo al maximo y presiona M (profundidad maxima).")
    print("  6. ESC para terminar.\n")

    with RobotInterface(simulate=args.sim) as robot:
        robot.home()

        while True:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            frame_n += 1
            display = frame.copy()

            if depth_pipe and frame_n % 6 == 0 and not depth_state["processing"]:
                depth_state["processing"] = True
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                threading.Thread(target=depth_worker, args=(rgb, depth_pipe),
                                 daemon=True).start()

            raw_val      = get_center_raw() if depth_pipe else None
            joints       = robot.get_raw_positions()
            shoulder_now = joints.get("hombro", 0)

            h_lim = 2400
            bar_h  = 200
            bar_x  = FRAME_W - 40
            filled = int(bar_h * max(0, shoulder_now) / h_lim)
            cv2.rectangle(display, (bar_x, FRAME_H//2 - bar_h//2),
                          (bar_x + 20, FRAME_H//2 + bar_h//2), (60, 60, 60), -1)
            cv2.rectangle(display, (bar_x, FRAME_H//2 + bar_h//2 - filled),
                          (bar_x + 20, FRAME_H//2 + bar_h//2), (0, 200, 255), -1)
            cv2.putText(display, f"{shoulder_now}", (bar_x - 10, FRAME_H//2 + bar_h//2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            if delivery_threshold is not None:
                thresh_y = int(FRAME_H//2 + bar_h//2 - bar_h * delivery_threshold / h_lim)
                cv2.line(display, (bar_x - 5, thresh_y), (bar_x + 25, thresh_y), (0, 255, 0), 2)
                cv2.putText(display, "UMBRAL", (bar_x - 50, thresh_y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            info_lines = [
                f"Hombro: {shoulder_now} pasos",
                f"Raw depth centro: {raw_val:.4f}" if raw_val else "Raw depth: calculando...",
                f"Umbral delivery: {delivery_threshold}" if delivery_threshold else "Umbral delivery: no registrado",
                f"Depth min raw: {depth_min_raw}" if depth_min_raw else "Depth min: no registrado",
                f"Depth max raw: {depth_max_raw}" if depth_max_raw else "Depth max: no registrado",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(display, line, (8, 30 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            cv2.putText(display,
                        "W/S=hombro  SPACE=umbral_delivery  M=depth_alcance  H=home  ESC=fin",
                        (8, FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

            cv2.imshow("Calibracion Umbral Hombro", display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key == ord('h'):
                robot.home()
            elif key in KEY_MAP:
                axis, steps = KEY_MAP[key]
                robot.move_joint(axis, steps)
            elif key == ord(' '):
                delivery_threshold = shoulder_now
                print(f"  Umbral delivery registrado: {delivery_threshold} pasos")
            elif key == ord('m') and raw_val is not None:
                if depth_min_raw is None:
                    depth_min_raw = round(raw_val, 4)
                    print(f"  Depth MIN raw registrado: {depth_min_raw}  (brazo extendido)")
                else:
                    depth_max_raw = round(raw_val, 4)
                    print(f"  Depth MAX raw registrado: {depth_max_raw}  (brazo retraido)")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print("RESULTADO — pegar en config.py:")
    print("="*55)
    if delivery_threshold is not None:
        print(f"DELIVERY_SHOULDER_STEPS = {delivery_threshold}")
    else:
        print("DELIVERY_SHOULDER_STEPS = ???  (no registrado)")
    if depth_min_raw is not None:
        print(f"# depth_min_raw = {depth_min_raw}  → convertir con tu mapa de calibracion")
    if depth_max_raw is not None:
        print(f"# depth_max_raw = {depth_max_raw}  → convertir con tu mapa de calibracion")
    print("="*55)

if __name__ == "__main__":
    main()
