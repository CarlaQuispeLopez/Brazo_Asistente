"""
SCRIPT 2 — Zona alcanzable del brazo en coordenadas de cámara

USO:
    python calibrate_reachable_zone.py --port COM3   (Windows)
    python calibrate_reachable_zone.py --port /dev/ttyUSB0  (Linux)

    O en modo simulado (sin brazo):
    python calibrate_reachable_zone.py --sim

QUÉ HACE:
    Mueve el brazo con el teclado y registra las posiciones extremas
    que el gripper alcanza en el frame de la cámara.
    Al terminar imprime el rectángulo REACHABLE_ZONE_PX para config.py.

CONTROLES:
    W/S  → hombro arriba/abajo
    A/D  → base izquierda/derecha
    Q/E  → codo sube/baja
    SPACE → registrar posición actual del gripper en el frame
    R     → resetear puntos registrados
    H     → ir a HOME
    ESC   → terminar y mostrar resultado
"""

import cv2
import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_interface import RobotInterface
from config import ACTION_STEPS

CAMERA_INDEX = 0
FRAME_W      = 640
FRAME_H      = 480

GRIPPER_COLOR = (0, 0, 255)

def draw_crosshair(frame, cx, cy, color=(0, 255, 255), size=20):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)

def draw_zone(frame, points):
    if len(points) < 2:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"({x1},{y1}) - ({x2},{y2})", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--sim",  action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    registered_points = []
    gripper_px = (FRAME_W // 2, FRAME_H // 2)

    KEY_MAP = {
        ord('w'): ("hombro",   +ACTION_STEPS),
        ord('s'): ("hombro",   -ACTION_STEPS),
        ord('a'): ("base",     -ACTION_STEPS),
        ord('d'): ("base",     +ACTION_STEPS),
        ord('q'): ("codo",     +ACTION_STEPS),
        ord('e'): ("codo",     -ACTION_STEPS),
    }

    print("Instrucciones:")
    print("  Mueve el gripper con W/S/A/D/Q/E al extremo de su rango.")
    print("  Presiona SPACE para registrar ese punto en el frame.")
    print("  Registra al menos 4 puntos: izquierda, derecha, arriba, abajo.")
    print("  Presiona ESC para ver el resultado final.\n")

    with RobotInterface(simulate=args.sim) as robot:
        robot.home()

        while True:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

            display = frame.copy()

            draw_crosshair(display, gripper_px[0], gripper_px[1])
            draw_zone(display, registered_points)

            for i, pt in enumerate(registered_points):
                cv2.circle(display, pt, 6, (0, 200, 0), -1)
                cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

            joints = robot.get_raw_positions()
            joint_str = "  ".join([f"{k}:{v}" for k, v in joints.items()])
            cv2.putText(display, joint_str, (8, FRAME_H - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            cv2.putText(display,
                        "W/S=hombro  A/D=base  Q/E=codo  SPACE=registrar  H=home  ESC=fin",
                        (8, FRAME_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            cv2.putText(display, f"Puntos registrados: {len(registered_points)}",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("Calibracion Zona Alcanzable", display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key == ord('h'):
                robot.home()
                print("HOME")
            elif key == ord('r'):
                registered_points.clear()
                print("Puntos reiniciados.")
            elif key == ord(' '):
                cx_px = gripper_px[0]
                cy_px = gripper_px[1]
                registered_points.append((cx_px, cy_px))
                print(f"  Punto {len(registered_points)} registrado: ({cx_px}, {cy_px})")
            elif key in KEY_MAP:
                axis, steps = KEY_MAP[key]
                moved = robot.move_joint(axis, steps)
                if not moved:
                    print(f"Limite alcanzado en {axis}")

    cap.release()
    cv2.destroyAllWindows()

    if len(registered_points) >= 2:
        xs = [p[0] for p in registered_points]
        ys = [p[1] for p in registered_points]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        print("\n" + "="*55)
        print("RESULTADO — pegar en config.py:")
        print("="*55)
        print(f"REACHABLE_ZONE_PX = ({x1}, {y1}, {x2}, {y2})")
        print("="*55)
    else:
        print("No se registraron suficientes puntos (mínimo 2).")

if __name__ == "__main__":
    main()
