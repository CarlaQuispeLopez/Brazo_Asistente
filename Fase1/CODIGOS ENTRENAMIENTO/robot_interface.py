"""
robot_interface.py — Comunicación serial con Arduino/RAMPS 1.4

Protocolo serial:
    PC → Arduino : "<CMD arg1 arg2 ...>\n"
    Arduino → PC : "OK\n" | "DONE\n" | "ERROR msg\n"

Comandos implementados en el firmware Arduino:
    MOVE <eje> <pasos> <dir> <vel_us>   mueve un eje N pasos
    GRIPPER <grados>                     mueve el servo del gripper
    HOME                                 lleva todos los ejes a posición 0
    ENABLE <0|1>                         habilita o deshabilita los drivers
    QUERY_POS                            responde posiciones actuales (no implementado en hw)
"""

import serial
import time
import threading
import numpy as np

from config import (
    SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT,
    JOINT_LIMITS, ACTION_STEPS, DEFAULT_SPEED,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    STATE_DIM,
)


class RobotInterface:
    """
    Interfaz de alto nivel para controlar el brazo robótico NutriBot.

    Uso recomendado como context manager:
        with RobotInterface(simulate=False) as robot:
            robot.home()
            robot.move_joint("hombro", 200)
    """

    AXES = ["base", "hombro", "codo", "muneca", "rotacion"]

    def __init__(
        self,
        port: str     = SERIAL_PORT,
        baud: int     = SERIAL_BAUD,
        simulate: bool = False,
    ):
        self.simulate = simulate
        self._lock    = threading.Lock()

        self._pos: dict[str, int] = {ax: 0 for ax in self.AXES}
        self._gripper_angle: int  = GRIPPER_OPEN
        self._connected: bool     = False
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
            time.sleep(2.0)
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
        """
        Espera la confirmación DONE del Arduino tras un movimiento.
        Retorna True si llega DONE, False si hay timeout o ERROR.
        """
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
    # Control de movimiento
    # ----------------------------------------------------------

    def move_joint(
        self,
        axis:     str,
        steps:    int,
        speed_us: int = DEFAULT_SPEED,
    ) -> bool:
        """
        Mueve un eje un número de pasos.
            steps > 0  dirección positiva
            steps < 0  dirección negativa

        Verifica límites antes de enviar el comando.
        Retorna True si el movimiento fue ejecutado, False si fue bloqueado.
        """
        if axis not in JOINT_LIMITS:
            print(f"[RobotInterface] Eje desconocido: '{axis}'")
            return False

        lo, hi      = JOINT_LIMITS[axis]
        new_pos     = self._pos[axis] + steps

        if not (lo <= new_pos <= hi):
            print(
                f"[RobotInterface] Límite en '{axis}': "
                f"{self._pos[axis]} + {steps} = {new_pos}  fuera de [{lo}, {hi}]"
            )
            return False

        direction = 1 if steps > 0 else 0
        abs_steps = abs(steps)
        cmd       = f"MOVE {axis} {abs_steps} {direction} {speed_us}"

        resp = self._send(cmd)
        if resp in ("OK", ""):
            self._wait_done()
            self._pos[axis] = new_pos
            return True

        print(f"[RobotInterface] Respuesta inesperada de Arduino: '{resp}'")
        return False

    def set_gripper(self, angle: int):
        """Mueve el servo del gripper al ángulo indicado (0–180°)."""
        angle = int(np.clip(angle, 0, 180))
        self._send(f"GRIPPER {angle}")
        self._gripper_angle = angle
        if self.simulate:
            time.sleep(0.3)

    def open_gripper(self):
        self.set_gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self.set_gripper(GRIPPER_CLOSE)

    def home(self):
        """Lleva todos los ejes a la posición de referencia (pasos = 0)."""
        print("[RobotInterface] Yendo a HOME...")
        self._send("HOME")
        self._wait_done(timeout=30.0)
        self._pos = {ax: 0 for ax in self.AXES}
        self.open_gripper()
        print("[RobotInterface] HOME alcanzado.")

    def enable_drivers(self, enable: bool = True):
        """Habilita o deshabilita los drivers de los motores paso a paso."""
        val = 1 if enable else 0
        self._send(f"ENABLE {val}")

    # ----------------------------------------------------------
    # Ejecución de acción discreta (usada por el agente RL)
    # ----------------------------------------------------------

    def execute_action(self, action_idx: int, steps: int = ACTION_STEPS) -> bool:
        """
        Mapea un índice de acción discreta (0–10) al movimiento físico.
        Retorna True si la acción fue ejecutada sin error.

        Índices:
            0  base+        1  base-
            2  hombro+      3  hombro-
            4  codo+        5  codo-
            6  muneca+      7  muneca-
            8  rotacion+    9  rotacion-
            10 cerrar gripper
        """
        action_map = {
            0:  ("base",      +steps),
            1:  ("base",      -steps),
            2:  ("hombro",    +steps),
            3:  ("hombro",    -steps),
            4:  ("codo",      +steps),
            5:  ("codo",      -steps),
            6:  ("muneca",    +steps),
            7:  ("muneca",    -steps),
            8:  ("rotacion",  +steps),
            9:  ("rotacion",  -steps),
            10: None,
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
        Devuelve las posiciones de los 5 ejes normalizadas a [-1, 1].
        El orden es: base, hombro, codo, muneca, rotacion.
        """
        result = []
        for ax in self.AXES:
            lo, hi = JOINT_LIMITS[ax]
            rng    = hi - lo
            norm   = (self._pos[ax] - lo) / rng * 2.0 - 1.0
            result.append(float(np.clip(norm, -1.0, 1.0)))
        return np.array(result, dtype=np.float32)

    def get_raw_positions(self) -> dict:
        """Devuelve las posiciones en pasos (sin normalizar) de cada eje."""
        return dict(self._pos)

    def set_raw_positions(self, pos_dict: dict):
        """
        Restaura el estado interno de posiciones.
        Útil para el reset de episodio en el entorno RL.
        No mueve el hardware, solo actualiza el registro interno.
        """
        for ax, val in pos_dict.items():
            if ax in self._pos:
                self._pos[ax] = int(val)

    def get_gripper_angle(self) -> int:
        """Retorna el ángulo actual del servo del gripper."""
        return self._gripper_angle

    def is_gripper_closed(self) -> bool:
        """Retorna True si el gripper está en posición de cierre."""
        return self._gripper_angle <= GRIPPER_CLOSE

    # ----------------------------------------------------------
    # Utilidades
    # ----------------------------------------------------------

    def status(self):
        """Imprime el estado actual del robot en consola."""
        print("\n[RobotInterface] Estado actual:")
        for ax in self.AXES:
            lo, hi = JOINT_LIMITS[ax]
            pos    = self._pos[ax]
            pct    = (pos - lo) / (hi - lo) * 100
            print(f"  {ax:10s}: {pos:+6d} pasos  ({pct:5.1f}% del rango [{lo}, {hi}])")
        print(f"  gripper   : {self._gripper_angle}°  "
              f"({'CERRADO' if self.is_gripper_closed() else 'abierto'})")
        print(f"  modo      : {'SIMULADO' if self.simulate else 'HARDWARE'}\n")

    def close(self):
        if not self.simulate and self._connected and self._ser is not None:
            self.enable_drivers(False)
            self._ser.close()
            print("[RobotInterface] Conexión serial cerrada.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ============================================================
# Firmware de referencia para Arduino (esqueleto)
# ============================================================

ARDUINO_FIRMWARE_HINT = """
/* ============================================================
   FIRMWARE ARDUINO — NUTRIBOT
   Pegar en el .ino del Arduino Mega 2560 con RAMPS 1.4
   Requiere: AccelStepper, Servo
   ============================================================ */

#include <AccelStepper.h>
#include <Servo.h>

// Pines según RAMPS 1.4
// base=X, hombro=Y, codo=Z, muneca=E0, rotacion=E1
AccelStepper steppers[5] = {
  AccelStepper(AccelStepper::DRIVER, 54, 55),  // base      STEP=54 DIR=55
  AccelStepper(AccelStepper::DRIVER, 60, 61),  // hombro    STEP=60 DIR=61
  AccelStepper(AccelStepper::DRIVER, 46, 48),  // codo      STEP=46 DIR=48
  AccelStepper(AccelStepper::DRIVER, 26, 28),  // muneca    STEP=26 DIR=28
  AccelStepper(AccelStepper::DRIVER, 36, 34),  // rotacion  STEP=36 DIR=34
};

int EN_PINS[5] = {38, 56, 62, 24, 30};

Servo gripper;

void setup() {
  Serial.begin(115200);
  gripper.attach(9);
  gripper.write(180);  // abierto al inicio
  for (int i = 0; i < 5; i++) {
    pinMode(EN_PINS[i], OUTPUT);
    digitalWrite(EN_PINS[i], LOW);  // LOW = habilitar driver
    steppers[i].setMaxSpeed(2000);
    steppers[i].setAcceleration(500);
  }
}

int axisIndex(String name) {
  if (name == "base")     return 0;
  if (name == "hombro")   return 1;
  if (name == "codo")     return 2;
  if (name == "muneca")   return 3;
  if (name == "rotacion") return 4;
  return -1;
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\\n');
    cmd.trim();

    if (cmd.startsWith("MOVE")) {
      // Formato: MOVE <eje> <pasos> <dir 0|1> <vel_us>
      // Ejemplo: MOVE hombro 200 1 800
      int i1 = cmd.indexOf(' ');
      int i2 = cmd.indexOf(' ', i1 + 1);
      int i3 = cmd.indexOf(' ', i2 + 1);
      int i4 = cmd.indexOf(' ', i3 + 1);

      String axName = cmd.substring(i1 + 1, i2);
      long   nSteps = cmd.substring(i2 + 1, i3).toInt();
      int    dir    = cmd.substring(i3 + 1, i4).toInt();
      int    velUs  = cmd.substring(i4 + 1).toInt();

      int idx = axisIndex(axName);
      if (idx < 0) { Serial.println("ERROR eje desconocido"); return; }

      long target = dir == 1 ? nSteps : -nSteps;
      float speed = 1000000.0 / velUs;  // convierte µs/paso → pasos/s

      steppers[idx].setMaxSpeed(speed);
      steppers[idx].move(target);
      Serial.println("OK");

      while (steppers[idx].distanceToGo() != 0) {
        steppers[idx].run();
      }
      Serial.println("DONE");

    } else if (cmd.startsWith("GRIPPER")) {
      int ang = cmd.substring(8).toInt();
      ang = constrain(ang, 0, 180);
      gripper.write(ang);
      Serial.println("OK");

    } else if (cmd == "HOME") {
      for (int i = 0; i < 5; i++) {
        steppers[i].moveTo(0);
      }
      bool moving = true;
      while (moving) {
        moving = false;
        for (int i = 0; i < 5; i++) {
          if (steppers[i].distanceToGo() != 0) {
            steppers[i].run();
            moving = true;
          }
        }
      }
      gripper.write(180);  // abrir gripper en HOME
      Serial.println("DONE");

    } else if (cmd.startsWith("ENABLE")) {
      int val = cmd.charAt(7) - '0';
      for (int i = 0; i < 5; i++) {
        digitalWrite(EN_PINS[i], val ? LOW : HIGH);
      }
      Serial.println("OK");

    } else {
      Serial.println("ERROR comando desconocido");
    }
  }
}
"""