#!/usr/bin/env python
"""
nutribot_robot.py — Wrapper de NutriBot para LeRobot
"""

import time
import numpy as np
from typing import Optional, Dict, Any

# LeRobot imports
try:
    from lerobot.common.robot_devices.robots.utils import Robot
    from lerobot.common.robot_devices.motors import MotorConfig
except ImportError:
    raise ImportError("Instala LeRobot: pip install lerobot")

from robot_interface import RobotInterface
from config import JOINT_LIMITS, ACTION_STEPS, GRIPPER_OPEN, GRIPPER_CLOSE


class NutriBotRobot(Robot):
    """
    Adaptador de NutriBot para la interfaz Robot de LeRobot.
    Permite que los modelos de Hugging Face controlen tu brazo real.
    """

    def __init__(self, simulate: bool = False, port: str = "COM3"):
        super().__init__()
        self.simulate = simulate
        self.robot = RobotInterface(simulate=simulate, port=port)

        # Configuración de motores para LeRobot
        self.motor_names = ["base", "hombro", "codo", "muneca", "rotacion", "gripper"]
        self.motor_configs = {
            name: MotorConfig(
                index=i,
                name=name,
                # Límites normalizados [-1, 1] que LeRobot espera
                position_limits=(-1.0, 1.0),
            ) for i, name in enumerate(self.motor_names)
        }

        # Estado interno
        self._position = np.zeros(len(self.motor_names))
        self._velocity = np.zeros(len(self.motor_names))
        self._gripper_closed = False

    # ========================================================
    # Métodos obligatorios de la interfaz Robot de LeRobot
    # ========================================================

    def connect(self) -> None:
        """Conectar al hardware (ya lo hace robot_interface)"""
        pass  # robot_interface ya se conectó en __init__

    def disconnect(self) -> None:
        """Desconectar del hardware"""
        self.robot.close()

    def read_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado actual del robot.
        Formato esperado por LeRobot:
            - observation.state: vector de posiciones normalizadas
            - observation.image: frame de cámara (opcional)
        """
        # Posiciones normalizadas [-1, 1]
        joints = self.robot.get_joint_positions()  # [base, hombro, codo, muneca, rotacion]
        gripper_pos = 1.0 if self._gripper_closed else -1.0

        self._position = np.concatenate([joints, [gripper_pos]])

        # Leer frame de cámara (si quieres incluir imagen)
        # frame = self._read_camera()

        return {
            "observation.state": self._position.copy(),
            # "observation.image": frame,  # opcional
            "motor_names": self.motor_names,
        }

    def send_action(self, action: np.ndarray) -> None:
        """
        Recibe una acción de LeRobot y la ejecuta.
        action: vector de 6 dimensiones (5 motores + gripper) en [-1, 1]
        """
        if len(action) != 6:
            raise ValueError(f"Acción debe tener 6 dims, recibió {len(action)}")

        # Convertir de [-1,1] a pasos para cada motor
        for i, axis in enumerate(["base", "hombro", "codo", "muneca", "rotacion"]):
            lo, hi = JOINT_LIMITS[axis]
            rng = hi - lo
            # Normalizado [-1,1] → pasos absolutos
            target_steps = lo + (action[i] + 1.0) / 2.0 * rng
            current_steps = self.robot._pos[axis]
            delta_steps = int(target_steps - current_steps)

            if abs(delta_steps) > 10:  # umbral mínimo
                self.robot.move_joint(axis, delta_steps)

        # Control del gripper
        gripper_action = action[5]
        if gripper_action > 0.5 and not self._gripper_closed:
            self.robot.close_gripper()
            self._gripper_closed = True
        elif gripper_action < -0.5 and self._gripper_closed:
            self.robot.open_gripper()
            self._gripper_closed = False

    def teleop_step(self, *args, **kwargs):
        """No usado, pero necesario por la interfaz"""
        pass

    def capture_observation(self) -> Dict[str, Any]:
        """Captura observación completa (estado + imagen)"""
        return self.read_state()

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray):
        self._position = value

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self, value: np.ndarray):
        self._velocity = value

    # ========================================================
    # Helper privado para leer cámara (opcional)
    # ========================================================

    def _read_camera(self) -> np.ndarray:
        """Lee un frame de la cámara (implementa si quieres visión en LeRobot)"""
        try:
            from vision import VisionPipeline
            with VisionPipeline() as vision:
                frame = vision.read_frame()
                return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        except:
            return np.zeros((480, 640, 3), dtype=np.uint8)


# ============================================================
# Prueba rápida
# ============================================================

if __name__ == "__main__":
    robot = NutriBotRobot(simulate=True)
    state = robot.read_state()
    print(f"Estado inicial: {state['observation.state']}")

    # Enviar acción de prueba (todo cero = posición neutral)
    robot.send_action(np.zeros(6))
    print("Acción enviada") 