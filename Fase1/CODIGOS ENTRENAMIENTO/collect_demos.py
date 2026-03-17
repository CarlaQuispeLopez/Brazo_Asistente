"""
collect_demos.py — Recolección de demostraciones humanas para NutriBot

USO:
    python collect_demos.py [--sim] [--analyze]

CONTROLES durante la sesión:
    W / S       hombro arriba / abajo
    A / D       base izquierda / derecha
    Q / E       codo sube / baja
    R / F       muñeca arriba / abajo
    Z / C       rotación izquierda / derecha
    G           cerrar gripper  (registra acción + notifica a vision)
    O           abrir gripper   (no se graba como demo)
    N           nuevo episodio  (descarta el actual sin guardar)
    ENTER       finalizar episodio actual y marcar como exitoso
    H           ir a HOME
    ESC         salir y guardar todo

FLUJO DE UN EPISODIO:
    1. Coloca el alimento en el plato dentro de la zona alcanzable.
    2. Mueve el brazo con las teclas hasta que el gripper esté sobre el alimento.
    3. Presiona G para cerrar el gripper.
    4. Espera el mensaje AGARRE CONFIRMADO en pantalla.
    5. Presiona ENTER para guardar el episodio como exitoso.
    6. El brazo vuelve a HOME automáticamente.

IMPORTANTE:
    - Solo se graban pasos de las fases SEARCH y GRASP.
    - La fase DELIVERY NO se graba (la maneja el controlador proporcional).
    - Grabar mínimo 30 episodios exitosos con variedad de posiciones.
"""

import cv2
import pickle
import os
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

from config import (
    DEMO_FILE, DEMO_DIR, ACTION_STEPS,
    ACTION_DIM, STATE_DIM,
)
from robot_interface import RobotInterface
from vision import VisionPipeline, PipelinePhase

# ============================================================
# Mapeo teclado → (eje, dirección, índice_acción)
# ============================================================

KEY_ACTION_MAP = {
    ord('w'): ("hombro",    +1,  2),
    ord('s'): ("hombro",    -1,  3),
    ord('a'): ("base",      -1,  1),
    ord('d'): ("base",      +1,  0),
    ord('q'): ("codo",      +1,  4),
    ord('e'): ("codo",      -1,  5),
    ord('r'): ("muneca",    +1,  6),
    ord('f'): ("muneca",    -1,  7),
    ord('z'): ("rotacion",  -1,  9),
    ord('c'): ("rotacion",  +1,  8),
    ord('g'): ("gripper_close", 0, 10),
    ord('o'): ("gripper_open",  0, -1),
}

ACTION_NAMES = {
    0:  "base+",    1:  "base-",
    2:  "hombro+",  3:  "hombro-",
    4:  "codo+",    5:  "codo-",
    6:  "muneca+",  7:  "muneca-",
    8:  "rot+",     9:  "rot-",
    10: "gripper",
}

# ============================================================
# Recolector de demostraciones
# ============================================================

class DemoCollector:

    def __init__(self, robot: RobotInterface, vision: VisionPipeline):
        self.robot  = robot
        self.vision = vision

        Path(DEMO_DIR).mkdir(parents=True, exist_ok=True)
        self._data = self._load_existing()

        self._current_episode: list = []
        self._grasp_done_in_episode = False
        self._waiting_confirmation  = False

        ep_count   = len(self._data.get("episodes", []))
        step_count = sum(len(ep["steps"]) for ep in self._data.get("episodes", []))
        print(f"[DemoCollector] Demos existentes: {ep_count} episodios, {step_count} pasos")

    # ----------------------------------------------------------
    # Persistencia
    # ----------------------------------------------------------

    def _load_existing(self) -> dict:
        if os.path.exists(DEMO_FILE):
            with open(DEMO_FILE, "rb") as f:
                data = pickle.load(f)
            print(f"[DemoCollector] Cargado: {DEMO_FILE}")
            return data
        return {
            "episodes": [],
            "metadata": {
                "state_dim":  STATE_DIM,
                "action_dim": ACTION_DIM,
                "version":    1,
            },
        }

    def _save(self):
        with open(DEMO_FILE, "wb") as f:
            pickle.dump(self._data, f)
        ep  = len(self._data["episodes"])
        steps = sum(len(e["steps"]) for e in self._data["episodes"])
        print(f"[DemoCollector] Guardado: {ep} episodios, {steps} pasos → {DEMO_FILE}")

    # ----------------------------------------------------------
    # Gestión de episodios
    # ----------------------------------------------------------

    def _save_step(self, state: np.ndarray, action_idx: int, food_visible: bool):
        self._current_episode.append({
            "state":        state.copy(),
            "action":       action_idx,
            "food_visible": food_visible,
        })

    def _finish_episode(self, success: bool):
        if len(self._current_episode) == 0:
            print("[DemoCollector] Episodio vacío, descartado.")
            self._reset_episode_state()
            return

        steps_with_food = sum(1 for s in self._current_episode if s["food_visible"])
        ratio = steps_with_food / len(self._current_episode)
        if ratio < 0.5:
            print(f"[DemoCollector] Advertencia: solo {ratio*100:.0f}% de pasos con comida visible.")

        self._data["episodes"].append({
            "steps":   self._current_episode,
            "success": success,
            "grasp_included": self._grasp_done_in_episode,
        })
        ep_n = len(self._data["episodes"])
        mark = "EXITOSO" if success else "sin marcar"
        print(f"[DemoCollector] Episodio {ep_n} guardado ({mark}) "
              f"— {len(self._current_episode)} pasos, gripper={self._grasp_done_in_episode}")
        self._save()
        self._reset_episode_state()

    def _discard_episode(self):
        print(f"[DemoCollector] Episodio descartado ({len(self._current_episode)} pasos).")
        self._reset_episode_state()

    def _reset_episode_state(self):
        self._current_episode       = []
        self._grasp_done_in_episode = False
        self._waiting_confirmation  = False
        self.vision.notify_gripper_opened()
        self.robot.home()
        self.robot.open_gripper()

    # ----------------------------------------------------------
    # Bucle principal
    # ----------------------------------------------------------

    def run(self):
        self._print_controls()
        cv2.namedWindow("NutriBot — Demo Collector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("NutriBot — Demo Collector", 800, 600)

        while True:
            frame = self.vision.read_frame()
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Sin camara (modo simulado)",
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            joints = self.robot.get_joint_positions()
            state, phase, annotated = self.vision.get_state(
                frame, joints, robot_interface=self.robot
            )

            food_visible = (self.vision._last_food is not None)

            self._draw_collector_hud(annotated, phase, food_visible)
            cv2.imshow("NutriBot — Demo Collector", annotated)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break

            elif key == ord('h'):
                self.robot.home()
                self.vision.notify_gripper_opened()
                print("[DemoCollector] HOME")

            elif key == ord('n'):
                self._discard_episode()
                print("[DemoCollector] Episodio nuevo iniciado.")

            elif key == 13:
                if self._grasp_done_in_episode:
                    self._finish_episode(success=True)
                    print("[DemoCollector] Episodio guardado como EXITOSO. Volviendo a HOME...")
                else:
                    print("[DemoCollector] No se detectó acción de gripper en este episodio. "
                          "Usa G para cerrar el gripper antes de finalizar.")

            elif key in KEY_ACTION_MAP:
                axis, direction, action_idx = KEY_ACTION_MAP[key]

                if phase == PipelinePhase.DELIVERY:
                    print("[DemoCollector] Fase DELIVERY activa. El controlador P tiene el mando.")
                    continue

                state_before = self.vision.food_to_state_vector(
                    self.vision._last_food, self.robot.get_joint_positions()
                )
                food_visible_before = (self.vision._last_food is not None)

                if axis == "gripper_close":
                    depth_now = self.vision.get_current_food_depth()
                    self.robot.close_gripper()
                    self.vision.notify_gripper_closed(depth_now)
                    self._grasp_done_in_episode = True
                    self._waiting_confirmation  = True
                    self._save_step(state_before, action_idx, food_visible_before)
                    print(f"[+] gripper CERRADO  depth_antes={depth_now:.1f}cm  "
                          f"pasos_ep={len(self._current_episode)}")

                elif axis == "gripper_open":
                    self.robot.open_gripper()
                    self.vision.notify_gripper_opened()
                    self._waiting_confirmation = False
                    print("[DemoCollector] Gripper abierto (acción no guardada).")

                else:
                    steps = direction * ACTION_STEPS
                    moved = self.robot.move_joint(axis, steps)
                    if not moved:
                        print(f"[DemoCollector] Limite alcanzado en {axis}.")
                        continue

                    if action_idx >= 0:
                        self._save_step(state_before, action_idx, food_visible_before)
                        food_str = "SÍ" if food_visible_before else "NO"
                        print(f"[+] {axis:10s} ({ACTION_NAMES[action_idx]:8s})  "
                              f"pasos_ep={len(self._current_episode):3d}  "
                              f"food={food_str}")

        if self._current_episode:
            ans = input("\n¿Guardar episodio actual como exitoso? (s/n): ")
            self._finish_episode(success=(ans.lower() == 's'))

        cv2.destroyAllWindows()
        print(f"\n[DemoCollector] Sesión terminada.")
        print(f"Total: {len(self._data['episodes'])} episodios, "
              f"{sum(len(e['steps']) for e in self._data['episodes'])} pasos.")
        print(f"Archivo: {DEMO_FILE}")

    # ----------------------------------------------------------
    # HUD del colector
    # ----------------------------------------------------------

    def _draw_collector_hud(self, frame, phase: PipelinePhase, food_visible: bool):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        ep_total    = len(self._data["episodes"])
        steps_now   = len(self._current_episode)
        ep_success  = sum(1 for e in self._data["episodes"] if e.get("success", False))

        cv2.putText(frame,
                    f"Episodios guardados: {ep_total}  ({ep_success} exitosos)  |  "
                    f"Pasos episodio actual: {steps_now}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        grip_str  = "CERRADO" if self.vision._grasp_evidence.gripper_closed else "abierto"
        conf_str  = "CONFIRMADO" if self.vision._grasp_evidence.confirmed    else "pendiente"
        food_str  = "DETECTADA" if food_visible else "NO DETECTADA"
        food_col  = (0, 255, 0)  if food_visible else (0, 0, 200)

        cv2.putText(frame,
                    f"Gripper: {grip_str} ({conf_str})  |  Comida: ",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        x_food = 8 + cv2.getTextSize(
            f"Gripper: {grip_str} ({conf_str})  |  Comida: ",
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )[0][0]
        cv2.putText(frame, food_str, (x_food, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, food_col, 1)

        if self._waiting_confirmation and not self.vision._grasp_evidence.confirmed:
            cv2.putText(frame, "Esperando confirmacion de agarre...",
                        (w // 2 - 180, h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

        if self.vision._grasp_evidence.confirmed and self._waiting_confirmation:
            cv2.putText(frame, "AGARRE CONFIRMADO  —  Presiona ENTER para guardar",
                        (w // 2 - 260, h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        joints = self.robot.get_raw_positions()
        j_str  = "  ".join(f"{k[0].upper()}:{v:+d}" for k, v in joints.items())
        cv2.putText(frame, j_str, (8, h - 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 180), 1)

        cv2.putText(frame,
                    "W/S=hombro  A/D=base  Q/E=codo  R/F=muneca  Z/C=rot  "
                    "G=cerrar  O=abrir  ENTER=exito  N=descartar  H=home  ESC=salir",
                    (8, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

    @staticmethod
    def _print_controls():
        print("\n" + "=" * 58)
        print("  NUTRIBOT — RECOLECCIÓN DE DEMOSTRACIONES")
        print("=" * 58)
        print("  W/S   = hombro arriba/abajo")
        print("  A/D   = base izquierda/derecha")
        print("  Q/E   = codo sube/baja")
        print("  R/F   = muñeca arriba/abajo")
        print("  Z/C   = rotación izquierda/derecha")
        print("  G     = cerrar gripper (registra agarre)")
        print("  O     = abrir gripper  (no se graba)")
        print("  ENTER = finalizar episodio como EXITOSO")
        print("  N     = descartar episodio actual")
        print("  H     = HOME")
        print("  ESC   = salir")
        print("=" * 58)
        print("  FLUJO: mover → alinear → G → esperar CONFIRMADO → ENTER")
        print("=" * 58 + "\n")


# ============================================================
# Análisis del dataset
# ============================================================

def analyze_demos(demo_file: str = DEMO_FILE):
    if not os.path.exists(demo_file):
        print(f"No se encontró '{demo_file}'")
        return

    with open(demo_file, "rb") as f:
        data = pickle.load(f)

    episodes = data.get("episodes", [])
    if not episodes:
        print("No hay episodios guardados.")
        return

    all_steps = [s for ep in episodes for s in ep["steps"]]
    actions   = [s["action"]       for s in all_steps]
    states    = np.array([s["state"] for s in all_steps])
    food_vis  = [s.get("food_visible", True) for s in all_steps]

    print("\n" + "=" * 50)
    print("  ANÁLISIS DEL DATASET DE DEMOS")
    print("=" * 50)
    print(f"  Episodios totales:   {len(episodes)}")
    print(f"  Exitosos:            {sum(1 for e in episodes if e.get('success', False))}")
    print(f"  Con gripper:         {sum(1 for e in episodes if e.get('grasp_included', False))}")
    print(f"  Pasos totales:       {len(all_steps)}")
    print(f"  Pasos con comida:    {sum(food_vis)}  ({sum(food_vis)/len(food_vis)*100:.1f}%)")

    lengths = [len(ep["steps"]) for ep in episodes]
    print(f"  Longitud media ep.:  {np.mean(lengths):.1f} pasos")
    print(f"  Longitud min/max:    {min(lengths)} / {max(lengths)}")

    print("\n  Distribución de acciones:")
    cnt = Counter(actions)
    for a in range(ACTION_DIM):
        n   = cnt.get(a, 0)
        pct = n / len(actions) * 100 if actions else 0
        bar = "#" * int(pct / 2)
        print(f"    {ACTION_NAMES.get(a,'?'):10s} ({a:2d}): {n:4d} ({pct:4.1f}%) {bar}")

    feat_names = ["cx_norm", "cy_norm", "depth_norm", "j0", "j1", "j2", "j3", "j4"]
    print("\n  Estado — media ± std:")
    for i, name in enumerate(feat_names):
        print(f"    {name:12s}: {states[:, i].mean():+.3f} ± {states[:, i].std():.3f}")

    print("=" * 50)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolector de demostraciones NutriBot")
    parser.add_argument("--sim",     action="store_true", help="Modo simulado sin hardware")
    parser.add_argument("--analyze", action="store_true", help="Solo analizar demos existentes")
    parser.add_argument("--port",    default=None,        help="Puerto serial (sobreescribe config)")
    args = parser.parse_args()

    if args.analyze:
        analyze_demos()
    else:
        from config import SERIAL_PORT
        port = args.port if args.port else SERIAL_PORT

        with RobotInterface(simulate=args.sim, port=port) as robot:
            with VisionPipeline() as vision:
                robot.home()
                robot.open_gripper()
                collector = DemoCollector(robot, vision)
                collector.run()
                analyze_demos()