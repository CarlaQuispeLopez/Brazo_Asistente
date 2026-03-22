"""
collect_demos.py — Recolección de demostraciones humanas para NutriBot
v2: una sola foto del plato al inicio, luego control manual sin visión en tiempo real.

FLUJO:
    1. Brazo va a HOME (cámara enfoca el plato)
    2. Se toma UNA snapshot → YOLO + Depth detecta el alimento
    3. Operador confirma la detección
    4. Grabación inicia: operador mueve el brazo manualmente
    5. Cada paso guarda: (estado_con_food_fijo, acción)
    6. Operador cierra la pinza cuando el gripper está sobre el alimento
    7. El sistema espera, levanta el brazo y pregunta si fue exitoso
    8. Se guarda el episodio

COMANDOS:
    CAM      → Tomar snapshot y detectar alimento
    I        → Iniciar grabación (después de CAM)
    F        → Finalizar episodio
    B+ / B-  → Base +/- pasos
    H+ / H-  → Hombro +/- pasos
    C+ / C-  → Codo +/- pasos
    G+ / G-  → Gripper sube / baja (stepper)
    GRIP+    → Cerrar pinza (y verificar agarre)
    GRIP-    → Abrir pinza
    STEPS N  → Cambiar tamaño de paso (default: 100)
    HOME     → Ir a HOME
    NUEVO    → Descartar episodio actual
    STATUS   → Ver estado del brazo
    ANALYZE  → Estadísticas del dataset
    EXIT     → Guardar y salir
"""

import os
import cv2
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from config import (
    DEMO_FILE, DEMO_DIR, ACTION_STEPS,
    GRIPPER_STABILIZE_SECS, LIFT_SUCCESS_STEPS,
)
from robot_interface import RobotInterface
from vision import VisionPipeline, FoodDetection

# ============================================================
# Mapa de comandos de teclado
# ============================================================
# Formato: cmd → (eje, dirección, action_idx)
#   eje = None para pinza
#   action_idx = -1 → no se graba
COMMAND_MAP = {
    "B+":    ("base",    +1, 0),
    "B-":    ("base",    -1, 1),
    "H+":    ("hombro",  +1, 2),
    "H-":    ("hombro",  -1, 3),
    "C+":    ("codo",    +1, 4),
    "C-":    ("codo",    -1, 5),
    "G+":    ("gripper", +1, 6),   # sube la garra (stepper)
    "G-":    ("gripper", -1, 7),   # baja la garra (stepper)
    "GRIP+": (None,       0, 8),   # cerrar pinza
    "GRIP-": (None,       0, -1),  # abrir pinza (no se graba como acción)
}


# ============================================================
# Colector de demostraciones
# ============================================================

class DemoCollector:

    def __init__(self, robot: RobotInterface, vision: VisionPipeline, step_size: int = ACTION_STEPS):
        self.robot      = robot
        self.vision     = vision
        self.step_size  = step_size

        self._food_target:  FoodDetection | None = None
        self._snapshot:     np.ndarray | None    = None
        self._cam_ok:       bool = False
        self._recording:    bool = False
        self._current_ep:   list = []

        Path(DEMO_DIR).mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    # ----------------------------------------------------------
    # Persistencia
    # ----------------------------------------------------------

    def _load(self) -> dict:
        if os.path.exists(DEMO_FILE):
            with open(DEMO_FILE, "rb") as f:
                d = pickle.load(f)
            print(f"[Demo] Dataset existente: {len(d.get('episodes', []))} episodios.")
            return d
        return {"episodes": [], "created": datetime.now().isoformat()}

    def _save(self):
        with open(DEMO_FILE, "wb") as f:
            pickle.dump(self.data, f)

    def _ep_count(self) -> int:
        return len(self.data.get("episodes", []))

    # ----------------------------------------------------------
    # Estado del sistema
    # ----------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """
        Construye el vector de estado usando la food_target fija del snapshot
        y las posiciones articulares actuales.
        """
        joints = self.robot.get_joint_positions()   # (4,): base, hombro, codo, gripper
        return self.vision.food_to_state_vector(
            self._food_target,
            joints,
            self.robot.is_gripper_closed(),
        )

    # ----------------------------------------------------------
    # Comandos principales
    # ----------------------------------------------------------

    def cmd_cam(self):
        """Toma snapshot y detecta el alimento. Muestra resultado para confirmación."""
        print("[Demo] Tomando snapshot del plato...")

        frame = self.vision.take_snapshot()
        if frame is None:
            print("[Demo] ERROR: No se pudo capturar frame.")
            return

        food = self.vision.detect_food(frame)

        # Mostrar al operador para que confirme
        try:
            confirmed = self.vision.show_snapshot_interactive(frame, food)
        except KeyboardInterrupt:
            print("[Demo] Abortado por operador.")
            return

        if not confirmed:
            print("[Demo] Foto rechazada. Escribe CAM para intentar de nuevo.")
            return

        if food is None:
            print("[Demo] No se detectó alimento. Reposiciona el plato y escribe CAM.")
            return

        self._snapshot    = frame
        self._food_target = food
        self._cam_ok      = True

        print(f"[Demo] Alimento confirmado: {food.label}")
        print(f"  posición: cx={food.center_norm[0]:.3f}  cy={food.center_norm[1]:.3f}")
        print(f"  depth_norm: {food.depth_norm:.3f}")
        print("[Demo] Escribe I para iniciar la grabación.")

    def cmd_iniciar(self):
        if not self._cam_ok:
            print("[Demo] Primero toma la foto con CAM.")
            return
        if self._recording:
            print("[Demo] Ya hay una grabación activa.")
            return
        self._recording  = True
        self._current_ep = []
        print(f"[Demo] Grabación iniciada. Episodio {self._ep_count() + 1}.")

    def cmd_finalizar(self):
        if not self._recording:
            print("[Demo] No hay grabación activa.")
            return
        self._recording = False
        n = len(self._current_ep)
        if n == 0:
            print("[Demo] Episodio vacío, descartado.")
            return

        print(f"[Demo] Grabación detenida. {n} pasos registrados.")
        print("¿El agarre fue exitoso? (s/n): ", end="", flush=True)
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        success = (ans == "s")

        episode = {
            "food_target": {
                "label":      self._food_target.label,
                "confidence": self._food_target.confidence,
                "cx_norm":    self._food_target.center_norm[0],
                "cy_norm":    self._food_target.center_norm[1],
                "depth_norm": self._food_target.depth_norm,
                "depth_raw":  self._food_target.depth_raw,
            },
            "steps":     list(self._current_ep),
            "success":   success,
            "timestamp": datetime.now().isoformat(),
            "n_steps":   n,
        }

        self.data.setdefault("episodes", []).append(episode)
        self._save()
        print(f"[Demo] Episodio {self._ep_count()} guardado ({'EXITOSO' if success else 'fallido'}).")

        # Resetear para el siguiente episodio
        self._food_target = None
        self._snapshot    = None
        self._cam_ok      = False
        self._current_ep  = []

    def cmd_mover(self, cmd: str, pasos: int | None = None):
        """Ejecuta un movimiento manual y, si está grabando, lo registra."""
        axis, direction, action_idx = COMMAND_MAP[cmd]
        steps   = pasos if pasos is not None else self.step_size
        state_before = self._get_state()

        if axis is None:
            # Es un comando de pinza
            if cmd == "GRIP+":
                self.robot.close_gripper()
                print(f"[Demo] Pinza cerrada.")
                # Esperar estabilización
                time.sleep(GRIPPER_STABILIZE_SECS)
                # Levantar el brazo para verificar si agarró
                print(f"[Demo] Levantando hombro {LIFT_SUCCESS_STEPS} pasos para verificar agarre...")
                ok = self.robot.move_joint("hombro", LIFT_SUCCESS_STEPS)
                if ok:
                    print("[Demo] ✓ Brazo levantó. ¿El gripper tiene el alimento? Escribe F para finalizar.")
                else:
                    print("[Demo] ⚠ Límite de articulación al levantar.")
            else:
                self.robot.open_gripper()
                print("[Demo] Pinza abierta.")
        else:
            ok = self.robot.move_joint(axis, direction * steps)
            if not ok:
                print(f"[Demo] Límite de articulación alcanzado en {axis}.")
                return
            pos   = self.robot.get_raw_positions().get(axis, 0)
            estado = "REC" if self._recording else "libre"
            print(f"[{estado}] {cmd} {steps}  eje={axis}  pos={pos}")

        # Grabar si está en modo grabación y es una acción válida
        if self._recording and action_idx >= 0:
            self._current_ep.append({
                "state":  state_before.tolist(),
                "action": action_idx,
                "cmd":    cmd,
            })

    def cmd_status(self):
        pos  = self.robot.get_raw_positions()
        norm = self.robot.get_joint_positions()
        axes = ["base", "hombro", "codo", "gripper"]
        print("[Demo] Estado actual:")
        for ax, nv in zip(axes, norm):
            print(f"  {ax:<10} {pos.get(ax, 0):+6d} pasos  ({nv:+.3f} norm)")
        print(f"  Pinza:     {'CERRADA' if self.robot.is_gripper_closed() else 'abierta'}")
        print(f"  Grabando:  {self._recording}")
        print(f"  Episodios: {self._ep_count()}")
        print(f"  Pasos/cmd: {self.step_size}")
        if self._food_target:
            print(
                f"  Objetivo:  {self._food_target.label} "
                f"cx={self._food_target.center_norm[0]:.3f} "
                f"depth={self._food_target.depth_norm:.3f}"
            )

    def cmd_analyze(self):
        episodes = self.data.get("episodes", [])
        if not episodes:
            print("[Demo] No hay episodios.")
            return
        total   = len(episodes)
        success = sum(1 for e in episodes if e.get("success"))
        steps   = [e["n_steps"] for e in episodes]
        print(f"\n[Demo] Dataset: {total} episodios  |  {success} exitosos  ({success/total*100:.0f}%)")
        print(f"  Pasos por episodio: min={min(steps)}  max={max(steps)}  prom={np.mean(steps):.1f}")

        # Distribución de alimentos
        labels = [e["food_target"]["label"] for e in episodes]
        unique = set(labels)
        print(f"  Alimentos: {', '.join(f'{l}×{labels.count(l)}' for l in sorted(unique))}")
        print()

    # ----------------------------------------------------------
    # Bucle principal de interacción
    # ----------------------------------------------------------

    def run(self):
        print("\n" + "=" * 60)
        print("  NutriBot — Recolección de Demostraciones")
        print("  Comandos: CAM  I  F  B+/B-  H+/H-  C+/C-  G+/G-")
        print("            GRIP+/GRIP-  STEPS N  HOME  STATUS  EXIT")
        print("=" * 60 + "\n")

        while True:
            ep  = self._ep_count() + 1
            rec = "REC" if self._recording else "---"
            cam = "CAM✓" if self._cam_ok else "CAM?"
            try:
                raw = input(f"EP{ep} [{rec}][{cam}] > ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                raw = "EXIT"

            if not raw:
                continue

            if raw == "EXIT":
                print(f"[Demo] Dataset guardado: {self._ep_count()} episodios.")
                break
            elif raw == "CAM":
                self.cmd_cam()
            elif raw == "I":
                self.cmd_iniciar()
            elif raw == "F":
                self.cmd_finalizar()
            elif raw == "HOME":
                self.robot.home()
                print("[Demo] HOME alcanzado.")
            elif raw == "NUEVO":
                self._recording   = False
                self._food_target = None
                self._snapshot    = None
                self._cam_ok      = False
                self._current_ep  = []
                print("[Demo] Episodio descartado.")
            elif raw == "STATUS":
                self.cmd_status()
            elif raw == "ANALYZE":
                self.cmd_analyze()
            elif raw.startswith("STEPS "):
                parts = raw.split()
                if len(parts) == 2 and parts[1].isdigit():
                    self.step_size = max(10, min(1600, int(parts[1])))
                    print(f"[Demo] Pasos por comando: {self.step_size}")
            else:
                parts = raw.split()
                cmd   = parts[0]
                if cmd in COMMAND_MAP:
                    pasos = None
                    if len(parts) == 2:
                        try:
                            pasos = abs(int(parts[1]))
                        except ValueError:
                            print(f"[Demo] Número inválido: {parts[1]}")
                            continue
                    self.cmd_mover(cmd, pasos)
                else:
                    print(f"[Demo] Comando no reconocido: {raw}")
                    print("[Demo] Comandos disponibles: CAM I F B+/B- H+/H- C+/C- G+/G- GRIP+/GRIP- STEPS N HOME NUEVO STATUS ANALYZE EXIT")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriBot — Recolección de demos")
    parser.add_argument("--sim",      action="store_true", help="Modo simulado")
    parser.add_argument("--steps",    type=int, default=ACTION_STEPS)
    parser.add_argument("--analyze",  action="store_true", help="Solo mostrar estadísticas")
    args = parser.parse_args()

    if args.analyze:
        # Solo analizar sin hardware
        data = {}
        if os.path.exists(DEMO_FILE):
            with open(DEMO_FILE, "rb") as f:
                data = pickle.load(f)
        episodes = data.get("episodes", [])
        print(f"Episodios: {len(episodes)}")
        for i, ep in enumerate(episodes):
            print(f"  {i+1:3d}. {ep['food_target']['label']:20s}  "
                  f"pasos={ep['n_steps']:4d}  "
                  f"{'EXITOSO' if ep['success'] else 'fallido'}")
        exit(0)

    with RobotInterface(simulate=args.sim) as robot:
        with VisionPipeline() as vision:
            robot.home()
            collector = DemoCollector(robot, vision, args.steps)
            try:
                collector.run()
            except KeyboardInterrupt:
                print("\n[Demo] Interrumpido. Dataset guardado.")