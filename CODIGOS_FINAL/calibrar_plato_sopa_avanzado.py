import argparse
import pickle
import time
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

try:
    import serial
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False
    print("[ADVERTENCIA] pyserial no instalado. Modo simulación activado.")

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
TIMEOUT_SEG = 60

HOME_POSITION = {"base": 0, "hombro": 400, "codo": 400, "muneca": 0, "rotacion": 0}

POSICION_PLATO_RELATIVA = {
    "base":     950,
    "hombro":   1250,
    "codo":     1700,
    "muneca":   0,
    "rotacion": 0
}

AXIS_CMD = {
    "base":     "BASE",
    "hombro":   "HOMBRO",
    "codo":     "CODO",
    "muneca":   "GRIPPER",
    "rotacion": "GIRO",
}

CAMARA_INDEX    = 2
MODELO_YOLO_SEG = "yolov8n-seg.pt"
CLASES_PLATO    = {"bowl", "cup", "plate", "dish"}
EROSION_BORDE_PX = 12
WARMUP_FRAMES   = 20

FALLBACK_RADIO_MIN = 80
FALLBACK_RADIO_MAX = 400
FALLBACK_BLUR_K    = 11

TOL_H = 15
TOL_S = 40
TOL_V = 85

SOMBRA_S_MAX = 30
SOMBRA_V_MAX = 80

UMBRAL_VACIO_PCT  = 5.0
UMBRAL_RESIDUO_PCT = 10.0

CARPETA_SALIDA = Path("calibracion_sopa")
ARCHIVO_PKL    = CARPETA_SALIDA / "calibracion.pkl"


class RobotInterface:
    def __init__(self, simulate=False):
        self.simulate = simulate
        self._pos = dict(HOME_POSITION)
        self._ser = None

    def conectar(self):
        if self.simulate or not SERIAL_OK:
            print("[Robot] Modo SIMULADO")
            return True
        try:
            print(f"[Robot] Conectando {SERIAL_PORT} @ {SERIAL_BAUD}...")
            self._ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=5)
            time.sleep(2)
            self._ser.flushInput()
            deadline = time.time() + 15
            while time.time() < deadline:
                line = self._ser.readline().decode(errors="ignore").strip()
                if line == "READY":
                    print("[Robot] Arduino listo.")
                    return True
            print("[Robot] No se recibió READY, continuando...")
            return True
        except Exception as e:
            print(f"[Robot] Error de conexión: {e}")
            print("[Robot] Entrando en modo simulación.")
            self.simulate = True
            return True

    def _send_and_wait(self, cmd, timeout=TIMEOUT_SEG):
        if self.simulate:
            print(f"    [SIM] {cmd}")
            time.sleep(0.3)
            return "OK"
        self._ser.flushInput()
        self._ser.write((cmd + "\n").encode())
        start = time.time()
        while time.time() - start < timeout:
            if self._ser.in_waiting:
                line = self._ser.readline().decode(errors="ignore").strip()
                if line == "OK":
                    return "OK"
                if line == "ERR":
                    print(f"    [ERR] Arduino reportó error en: {cmd}")
                    return "ERR"
        print(f"    [TIMEOUT] No hubo respuesta para: {cmd}")
        return "TIMEOUT"

    def mover_relativo(self, eje, pasos):
        if pasos == 0:
            return True
        cur = self._pos[eje]
        nuevo = cur + pasos
        print(f"    {eje:10s}: {cur:+5d} → {nuevo:+5d}  (Δ{pasos:+d})")
        resp = self._send_and_wait(f"{AXIS_CMD[eje]} {pasos}")
        if resp == "OK":
            self._pos[eje] = nuevo
            return True
        print(f"    [FALLO] No se completó el movimiento de {eje}")
        return False

    def ir_home(self):
        print("\n[Robot] ── Yendo a HOME ──────────────────────────────")
        orden_seguro = ["hombro", "codo", "muneca", "rotacion", "base"]
        for eje in orden_seguro:
            objetivo = HOME_POSITION[eje]
            diff = objetivo - self._pos[eje]
            if diff != 0:
                ok = self.mover_relativo(eje, diff)
                if not ok:
                    print(f"    [AVISO] Error moviendo {eje}, continuando...")
        print("[Robot] HOME alcanzado.\n")

    def mover_a_plato(self):
        print("[Robot] ── Moviéndose directo a posición del plato ──────────")

        print("    BASE +950...")
        if not self.mover_relativo("base", POSICION_PLATO_RELATIVA["base"]):
            return False

        print("    HOMBRO +1250...")
        if not self.mover_relativo("hombro", POSICION_PLATO_RELATIVA["hombro"]):
            return False

        print("    CODO +1700...")
        if not self.mover_relativo("codo", POSICION_PLATO_RELATIVA["codo"]):
            return False

        print("[Robot] Posición del plato alcanzada.\n")
        return True

    def desconectar(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[Robot] Puerto serie cerrado.")


def segmentar_plato_yolo(frame, model):
    fh, fw = frame.shape[:2]
    mascara = np.zeros((fh, fw), dtype=np.uint8)
    results = model(frame, verbose=False, device="cpu")[0]
    if results.masks is None:
        return mascara
    for i in range(len(results.masks.data)):
        cls_id = int(results.boxes.cls[i])
        if cls_id not in model.names:
            continue
        if model.names[cls_id].lower() not in CLASES_PLATO:
            continue
        mask_raw   = results.masks.data[i].cpu().numpy()
        mask_frame = cv2.resize(mask_raw, (fw, fh), interpolation=cv2.INTER_LINEAR)
        mascara    = cv2.bitwise_or(mascara, (mask_frame * 255).astype(np.uint8))
    if np.any(mascara) and EROSION_BORDE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (EROSION_BORDE_PX*2+1, EROSION_BORDE_PX*2+1)
        )
        mascara = cv2.erode(mascara, k, iterations=1)
    return mascara


def segmentar_plato_circular(frame):
    fh, fw = frame.shape[:2]
    mascara = np.zeros((fh, fw), dtype=np.uint8)
    gris    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris    = cv2.GaussianBlur(gris, (FALLBACK_BLUR_K, FALLBACK_BLUR_K), 0)
    circulos = cv2.HoughCircles(
        gris, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=fh // 3,
        param1=80,
        param2=40,
        minRadius=FALLBACK_RADIO_MIN,
        maxRadius=FALLBACK_RADIO_MAX
    )
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        mejor = sorted(circulos[0], key=lambda c: c[2], reverse=True)[0]
        cx, cy, r = mejor
        r_eros = max(0, int(r) - EROSION_BORDE_PX)
        cv2.circle(mascara, (int(cx), int(cy)), r_eros, 255, -1)
    return mascara


def segmentar_plato(frame, model):
    mascara = segmentar_plato_yolo(frame, model)
    if np.any(mascara):
        return mascara, "yolo"
    return segmentar_plato_circular(frame), "circular"


def capturar_hsv_referencia(frame, mascara):
    if not np.any(mascara):
        return None
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixeles = hsv[mascara > 0].astype(np.float32)
    if len(pixeles) == 0:
        return None
    ref = {
        "H_mean": float(np.mean(pixeles[:, 0])),
        "S_mean": float(np.mean(pixeles[:, 1])),
        "V_mean": float(np.mean(pixeles[:, 2])),
        "H_std":  float(np.std(pixeles[:, 0])),
        "S_std":  float(np.std(pixeles[:, 1])),
        "V_std":  float(np.std(pixeles[:, 2])),
        "n_pixels": len(pixeles)
    }
    return ref


def pantalla_calibrar(cap, model, titulo, instrucciones, color_overlay=(0, 200, 0)):
    print("\n" + "="*60)
    print(titulo)
    for linea in instrucciones:
        print(f"  {linea}")
    print("  → ESPACIO para capturar   |   ESC para cancelar")
    print("="*60 + "\n")

    ventana = f"{titulo} — ESPACIO capturar, ESC cancelar"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, 1280, 720)

    ref_hsv   = None
    mascara_g = None
    frame_g   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        mascara, metodo = segmentar_plato(frame, model)
        display = frame.copy()

        if np.any(mascara):
            overlay = np.zeros_like(frame)
            overlay[mascara > 0] = color_overlay
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
            contornos, _ = cv2.findContours(
                mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(display, contornos, -1, color_overlay, 2)
            cv2.putText(display, "ESPACIO = capturar", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_overlay, 2)
            color_met = (0, 255, 100) if metodo == "yolo" else (0, 200, 255)
            cv2.putText(display, f"[{metodo.upper()}]", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_met, 2)
        else:
            cv2.putText(display, "Plato NO detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display, "Ajusta FALLBACK_RADIO_MIN/MAX si persiste", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 130, 255), 1)

        cv2.imshow(ventana, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key == 32 and np.any(mascara):
            ref_hsv = capturar_hsv_referencia(frame, mascara)
            if ref_hsv:
                mascara_g = mascara.copy()
                frame_g   = frame.copy()
                print(f"  ✅ Capturado  H:{ref_hsv['H_mean']:.1f}  "
                      f"S:{ref_hsv['S_mean']:.1f}  V:{ref_hsv['V_mean']:.1f}  "
                      f"({ref_hsv['n_pixels']} píxeles)")
                break

    cv2.destroyWindow(ventana)
    return ref_hsv, mascara_g, frame_g


def guardar_calibracion(ref_plato, ref_sombra, mascara, frame):
    CARPETA_SALIDA.mkdir(exist_ok=True)
    datos = {
        "referencia_plato":  ref_plato,
        "referencia_sombra": ref_sombra,
        "tolerancias":  {"H": TOL_H, "S": TOL_S, "V": TOL_V},
        "anti_sombra":  {"S_max": SOMBRA_S_MAX, "V_max": SOMBRA_V_MAX},
        "umbral_vacio":   UMBRAL_VACIO_PCT,
        "umbral_residuo": UMBRAL_RESIDUO_PCT,
        "mascara_plato":  mascara,
        "frame_referencia": frame,
    }
    with open(ARCHIVO_PKL, "wb") as f:
        pickle.dump(datos, f)
    print(f"\n✅ Calibración guardada en: {ARCHIVO_PKL}")
    print(f"   Plato  → H:{ref_plato['H_mean']:.1f} S:{ref_plato['S_mean']:.1f} V:{ref_plato['V_mean']:.1f}")
    if ref_sombra:
        print(f"   Sombra → H:{ref_sombra['H_mean']:.1f} S:{ref_sombra['S_mean']:.1f} V:{ref_sombra['V_mean']:.1f}")
    else:
        print("   Sombra → No calibrada (se usarán umbrales por defecto)")


def cargar_calibracion():
    if not ARCHIVO_PKL.exists():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_PKL}.\n"
            "Ejecuta primero: python 1_calibrar_sopa.py"
        )
    with open(ARCHIVO_PKL, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Calibración del sistema de detección de sopa")
    parser.add_argument("--sim",         action="store_true", help="Simulación sin hardware")
    parser.add_argument("--cam",         type=int, default=CAMARA_INDEX, help="Índice de cámara")
    parser.add_argument("--skip-homing", action="store_true", help="Omite movimientos del brazo")
    parser.add_argument("--solo-plato",  action="store_true", help="Solo calibra el plato vacío")
    parser.add_argument("--solo-sombra", action="store_true", help="Solo recalibra la sombra")
    args = parser.parse_args()

    robot = RobotInterface(simulate=args.sim)
    robot.conectar()

    if not args.skip_homing:
        robot.ir_home()
        robot.mover_a_plato()
    else:
        print("[INFO] Movimientos del brazo omitidos (--skip-homing)\n")

    print("[VISIÓN] Cargando modelo YOLO en CPU...")
    model = YOLO(MODELO_YOLO_SEG)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {args.cam}")
        robot.desconectar()
        sys.exit(1)

    print(f"[VISIÓN] Calentando cámara ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        cap.read()
    print("[VISIÓN] Cámara lista.\n")

    ref_plato  = None
    ref_sombra = None
    mascara_ref = None
    frame_ref   = None

    if args.solo_sombra:
        try:
            datos = cargar_calibracion()
            ref_plato   = datos["referencia_plato"]
            mascara_ref = datos.get("mascara_plato")
            frame_ref   = datos.get("frame_referencia")
            print("[INFO] Plato ya calibrado cargado desde archivo.")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            cap.release()
            robot.desconectar()
            sys.exit(1)

        ref_sombra, _, _ = pantalla_calibrar(
            cap, model,
            titulo="CALIBRACIÓN DE SOMBRA",
            instrucciones=["Provoca la sombra y presiona ESPACIO"],
            color_overlay=(0, 200, 200)
        )

    else:
        ref_plato, mascara_ref, frame_ref = pantalla_calibrar(
            cap, model,
            titulo="CALIBRACIÓN — PLATO VACÍO",
            instrucciones=[
                "Asegúrate de que el plato esté COMPLETAMENTE VACÍO y LIMPIO",
                "Cuando la máscara cubra bien el plato → ESPACIO",
            ],
            color_overlay=(0, 200, 0)
        )

        if ref_plato is None:
            print("\n[CANCELADO] Saliendo.")
            cap.release()
            robot.desconectar()
            sys.exit(0)

        if not args.solo_plato:
            ref_sombra, _, _ = pantalla_calibrar(
                cap, model,
                titulo="CALIBRACIÓN — SOMBRA (ESC para omitir)",
                instrucciones=["Provoca la sombra y presiona ESPACIO"],
                color_overlay=(0, 200, 200)
            )

    if ref_plato is not None:
        guardar_calibracion(ref_plato, ref_sombra, mascara_ref, frame_ref)

    cap.release()
    cv2.destroyAllWindows()
    robot.desconectar()
    print("\n[FIN] Calibración terminada.")

if __name__ == "__main__":
    main()