"""
SCRIPT 4 — Tiempo de estabilización mecánica tras cierre del gripper

USO:
    python calibrate_stabilization.py --port COM3
    python calibrate_stabilization.py --sim

QUÉ HACE:
    Cierra el gripper y mide cuántos milisegundos tarda en dejar de
    vibrar, usando el flujo óptico de la cámara como detector de movimiento.
    Repite N veces y promedia. Imprime GRIPPER_STABILIZE_SECS para config.py.

    El flujo óptico mide el movimiento píxel a píxel entre frames.
    Cuando el movimiento cae por debajo de un umbral, el brazo se considera
    estabilizado.

CONTROLES:
    SPACE → ejecutar una medición (abre, cierra, mide)
    ESC   → terminar y mostrar promedio
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_interface import RobotInterface

CAMERA_INDEX      = 0
FRAME_W           = 640
FRAME_H           = 480
MOTION_THRESHOLD  = 1.5
MAX_WAIT_SECS     = 5.0
MEASURE_REPS      = 5

def measure_stabilization(robot, cap) -> float:
    robot.open_gripper()
    time.sleep(1.0)

    ret, prev = cap.read()
    if not ret:
        return -1.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    print("  Cerrando gripper...")
    robot.close_gripper()
    t_close = time.time()

    t_stable = None
    t0       = time.time()

    while time.time() - t0 < MAX_WAIT_SECS:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion = float(np.mean(mag))

        elapsed = time.time() - t_close

        cv2.putText(frame, f"Movimiento: {motion:.3f}  (umbral={MOTION_THRESHOLD})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Tiempo desde cierre: {elapsed:.2f}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        bar_w = int((motion / 5.0) * (FRAME_W - 40))
        bar_w = min(bar_w, FRAME_W - 40)
        color = (0, 255, 0) if motion < MOTION_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (20, 80), (20 + bar_w, 105), color, -1)
        cv2.rectangle(frame, (20, 80), (FRAME_W - 20, 105), (180, 180, 180), 1)

        thresh_x = int(20 + (MOTION_THRESHOLD / 5.0) * (FRAME_W - 40))
        cv2.line(frame, (thresh_x, 75), (thresh_x, 110), (0, 255, 255), 2)

        cv2.imshow("Medicion Estabilizacion", frame)
        cv2.waitKey(1)

        if motion < MOTION_THRESHOLD and t_stable is None:
            t_stable = elapsed
            print(f"  Estabilizado en {t_stable:.3f}s  (movimiento={motion:.3f})")
            time.sleep(0.5)
            break

        prev_gray = gray

    if t_stable is None:
        t_stable = MAX_WAIT_SECS
        print(f"  No se estabilizo en {MAX_WAIT_SECS}s, usando {MAX_WAIT_SECS}s como valor.")

    return t_stable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--sim",  action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    measurements = []

    print(f"Mide el tiempo de estabilizacion del gripper.")
    print(f"Se haran {MEASURE_REPS} mediciones y se promediara.")
    print("Presiona SPACE para cada medicion, ESC para terminar.\n")

    with RobotInterface(simulate=args.sim) as robot:
        robot.home()

        while True:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

            display = frame.copy()
            cv2.putText(display, f"Mediciones: {len(measurements)}/{MEASURE_REPS}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if measurements:
                avg = sum(measurements) / len(measurements)
                cv2.putText(display, f"Promedio actual: {avg:.3f}s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                for i, m in enumerate(measurements):
                    cv2.putText(display, f"  Rep {i+1}: {m:.3f}s",
                                (10, 100 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.putText(display, "SPACE=medir  ESC=terminar",
                        (10, FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

            cv2.imshow("Medicion Estabilizacion", display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key == ord(' '):
                print(f"\nMedicion {len(measurements)+1}/{MEASURE_REPS}:")
                t = measure_stabilization(robot, cap)
                measurements.append(t)
                if len(measurements) >= MEASURE_REPS:
                    print(f"\nCompletadas {MEASURE_REPS} mediciones.")
                    break

    cap.release()
    cv2.destroyAllWindows()

    if measurements:
        avg  = sum(measurements) / len(measurements)
        safe = round(avg * 1.3, 2)
        print("\n" + "="*55)
        print("RESULTADO — pegar en config.py:")
        print("="*55)
        print(f"# Mediciones individuales: {[round(m,3) for m in measurements]}")
        print(f"# Promedio medido:          {avg:.3f}s")
        print(f"# Con margen de seguridad (x1.3):")
        print(f"GRIPPER_STABILIZE_SECS = {safe}")
        print("="*55)
    else:
        print("No se realizaron mediciones.")

if __name__ == "__main__":
    main()
