"""
robot_interface.py — Comunicación serial con Arduino/RAMPS 1.4
v2: protocolo actualizado a BASE/HOMBRO/CODO/GRIPPER/PINZA

Protocolo serial (Arduino lado):
    PC → Arduino : "BASE 200\n"   (pasos positivos o negativos)
                   "HOMBRO -100\n"
                   "CODO 500\n"
                   "GRIPPER 100\n"   (sube/baja la garra — stepper)
                   "PINZA ABRIR\n"
                   "PINZA CERRAR\n"
                   "PINZA 45\n"      (ángulo exacto 0-90°)
                   "PARAR\n"
                   "POSICION\n"
    Arduino → PC : "OK\n" al recibir
                   "DONE\n" al terminar movimiento
                   "ERROR msg\n" si falla
"""

import serial
import time
import threading
import numpy as np

from config import (
    SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT,
    JOINT_LIMITS, ACTION_STEPS, DEFAULT_SPEED,
    PINZA_OPEN_CMD, PINZA_CLOSE_CMD,
    HOME_POSITION,
    GRIPPER_STABILIZE_SECS,
    STATE_DIM,
)


class RobotInterface:
    """
    Interfaz de alto nivel para controlar el brazo robótico NutriBot.

    Ejes disponibles: base, hombro, codo, gripper (sube/baja la garra)
    Pinza (abrir/cerrar): servo controlado por PINZA ABRIR / PINZA CERRAR

    Uso recomendado como context manager:
        with RobotInterface(simulate=False) as robot:
            robot.home()
            robot.move_joint("hombro", 200)
    """

    # Ejes de motores paso a paso activos
    AXES = ["base", "hombro", "codo", "gripper"]

    # Mapa eje → comando Arduino
    AXIS_CMD = {
        "base":    "BASE",
        "hombro":  "HOMBRO",
        "codo":    "CODO",
        "gripper": "GRIPPER",
    }

    def __init__(
        self,
        port:     str  = SERIAL_PORT,
        baud:     int  = SERIAL_BAUD,
        simulate: bool = False,
    ):
        self.simulate = simulate
        self._lock    = threading.Lock()

        self._pos: dict[str, int] = {ax: 0 for ax in self.AXES}
        self._pinza_closed: bool  = False
        self._connected:    bool  = False
        self._ser: serial.Serial | None = None

        if not simulate:
            self._connect(port, baud)
        else:
            print("[RobotInterface] Modo SIMULADO activado.")

    # ----------------------------------------------------------
    # Conexión
    # ----------------------------------------------------------

    def _connect(self, port: str, baud: int):
        try:
            self._ser = serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)
            time.sleep(2.0)   # esperar reset del Arduino
            self._connected = True
            print(f"[RobotInterface] Conectado a {port} @ {baud} baud.")
        except serial.SerialException as e:
            print(f"[RobotInterface] ERROR al conectar: {e}")
            print("[RobotInterface] Cambiando a modo simulado.")
            self.simulate   = True
            self._connected = False

    # ----------------------------------------------------------
    # Comunicación serial
    # ----------------------------------------------------------

    def _send(self, cmd: str) -> str:
        """Envía un comando al Arduino y retorna la primera línea de respuesta."""
        if self.simulate:
            return "OK"
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write((cmd + "\n").encode())
            try:
                resp = self._ser.readline().decode().strip()
            except Exception as e:
                print(f"[RobotInterface] Error de lectura serial: {e}")
                resp = "ERROR lectura"
        return resp

    def _wait_done(self, timeout: float = 10.0) -> bool:
        """Espera la confirmación DONE del Arduino tras un movimiento."""
        if self.simulate:
            time.sleep(0.05)
            return True

        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self._ser.readline().decode().strip()
            except Exception:
                return False
            if line == "DONE":
                return True
            if line.startswith("ERROR"):
                print(f"[Arduino] {line}")
                return False
        print(f"[RobotInterface] Timeout esperando DONE ({timeout}s).")
        return False

    # ----------------------------------------------------------
    # Control de ejes (motores paso a paso)
    # ----------------------------------------------------------

    def move_joint(self, axis: str, steps: int) -> bool:
        """
        Mueve un eje un número de pasos.
            steps > 0  → dirección positiva
            steps < 0  → dirección negativa

        Envía al Arduino: "BASE 200" o "HOMBRO -100" etc.
        Verifica límites antes de enviar.
        Retorna True si el movimiento fue ejecutado.
        """
        if axis not in JOINT_LIMITS:
            print(f"[RobotInterface] Eje desconocido: '{axis}'")
            return False

        lo, hi  = JOINT_LIMITS[axis]
        new_pos = self._pos[axis] + steps

        if not (lo <= new_pos <= hi):
            print(
                f"[RobotInterface] Límite en '{axis}': "
                f"{self._pos[axis]} + {steps} = {new_pos}  fuera de [{lo}, {hi}]"
            )
            return False

        cmd_name = self.AXIS_CMD[axis]
        cmd      = f"{cmd_name} {steps}"

        resp = self._send(cmd)
        if resp in ("OK", ""):
            self._wait_done()
            self._pos[axis] = new_pos
            return True

        print(f"[RobotInterface] Respuesta inesperada de Arduino: '{resp}'")
        return False

    # ----------------------------------------------------------
    # Control de pinza (servo)
    # ----------------------------------------------------------

    def open_gripper(self):
        """Abre la pinza (servo)."""
        self._send(PINZA_OPEN_CMD)
        self._pinza_closed = False
        if self.simulate:
            time.sleep(0.3)

    def close_gripper(self):
        """Cierra la pinza (servo)."""
        self._send(PINZA_CLOSE_CMD)
        self._pinza_closed = True
        if self.simulate:
            time.sleep(0.3)

    def set_gripper_angle(self, angle: int):
        """Mueve la pinza a un ángulo exacto (0–90°)."""
        angle = int(np.clip(angle, 0, 90))
        self._send(f"PINZA {angle}")
        self._pinza_closed = (angle < 45)
        if self.simulate:
            time.sleep(0.3)

    def is_gripper_closed(self) -> bool:
        return self._pinza_closed

    # ----------------------------------------------------------
    # HOME
    # ----------------------------------------------------------

    def home(self):
        """
        Lleva el brazo a la posición HOME funcional.
        HOME = base:0, hombro:100, codo:1000, gripper:0

        Secuencia:
          1. Abrir pinza
          2. Mover cada eje a su posición HOME desde la posición actual
        """
        print("[RobotInterface] Yendo a HOME...")
        self.open_gripper()

        for axis in self.AXES:
            target  = HOME_POSITION.get(axis, 0)
            current = self._pos[axis]
            delta   = target - current
            if delta != 0:
                self.move_joint(axis, delta)

        print("[RobotInterface] HOME alcanzado.")

    def go_to_zero(self):
        """
        Lleva todos los ejes al paso 0 (origen mecánico).
        Útil para calibración inicial o apagado.
        """
        print("[RobotInterface] Volviendo a cero mecánico...")
        for axis in self.AXES:
            current = self._pos[axis]
            if current != 0:
                self.move_joint(axis, -current)
        self.open_gripper()
        print("[RobotInterface] Posición cero alcanzada.")

    def stop(self):
        """Envía comando PARAR al Arduino."""
        self._send("PARAR")

    def query_position(self) -> str:
        """Pide al Arduino las posiciones actuales (respuesta de texto)."""
        if self.simulate:
            return "SIMULADO"
        return self._send("POSICION")

    # ----------------------------------------------------------
    # Ejecución de acción discreta (usada por el agente RL)
    # ----------------------------------------------------------

    def execute_action(self, action_idx: int, steps: int = ACTION_STEPS) -> bool:
        """
        Mapea un índice de acción discreta (0–8) al movimiento físico.
        Retorna True si la acción fue ejecutada sin error.

        Índices:
            0  base+        4  codo+
            1  base-        5  codo-
            2  hombro+      6  gripper+   (sube la garra)
            3  hombro-      7  gripper-   (baja la garra)
                            8  cerrar_pinza
        """
        action_map = {
            0: ("base",    +steps),
            1: ("base",    -steps),
            2: ("hombro",  +steps),
            3: ("hombro",  -steps),
            4: ("codo",    +steps),
            5: ("codo",    -steps),
            6: ("gripper", +steps),
            7: ("gripper", -steps),
            8: None,   # cerrar pinza
        }

        act = action_map.get(action_idx)

        if act is None:
            self.close_gripper()
            return True

        axis, s = act
        return self.move_joint(axis, s)

    # ----------------------------------------------------------
    # Estado del robot
    # ----------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """
        Devuelve las posiciones de los 4 ejes normalizadas a [-1, 1].
        Orden: base, hombro, codo, gripper
        """
        result = []
        for ax in self.AXES:
            lo, hi = JOINT_LIMITS[ax]
            rng    = hi - lo
            norm   = (self._pos[ax] - lo) / rng * 2.0 - 1.0
            result.append(float(np.clip(norm, -1.0, 1.0)))
        return np.array(result, dtype=np.float32)

    def get_raw_positions(self) -> dict:
        return dict(self._pos)

    def set_raw_positions(self, pos_dict: dict):
        """Restaura el estado interno sin mover hardware (útil para reset RL)."""
        for ax, val in pos_dict.items():
            if ax in self._pos:
                self._pos[ax] = int(val)

    # ----------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------

    def status(self):
        print("\n[RobotInterface] Estado actual:")
        for ax in self.AXES:
            lo, hi = JOINT_LIMITS[ax]
            pos    = self._pos[ax]
            pct    = (pos - lo) / (hi - lo) * 100
            print(f"  {ax:10s}: {pos:+6d} pasos  ({pct:5.1f}% del rango [{lo}, {hi}])")
        print(f"  pinza     : {'CERRADA' if self._pinza_closed else 'abierta'}")
        print(f"  modo      : {'SIMULADO' if self.simulate else 'HARDWARE'}")
        print(f"  HOME ref  : {HOME_POSITION}\n")

    def close(self):
        if not self.simulate and self._connected and self._ser is not None:
            self._ser.close()
            print("[RobotInterface] Conexión serial cerrada.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()