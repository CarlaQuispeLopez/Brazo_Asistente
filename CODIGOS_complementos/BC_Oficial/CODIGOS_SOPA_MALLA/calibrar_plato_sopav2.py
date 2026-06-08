import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# =================================================================
#  CONFIGURACION — edita segun tu hardware
# =================================================================

SERIAL_PORT    = "COM3"
SERIAL_BAUD    = 115200
CAMERA_1_INDEX = 2

HOME_POSITION = {
    "base":     0,
    "hombro": 400,
    "codo":   400,
    "muneca":   0,
    "rotacion": 0,
}

POSICION_CAMARA_PLATO = {
    "base":     1100,
    "hombro":   -600,
    "codo":      900,
    "muneca":      0,
    "rotacion":    0,
}

AXIS_CMD = {
    "base":     "BASE",
    "hombro":   "HOMBRO",
    "codo":     "CODO",
    "muneca":   "GRIPPER",
    "rotacion": "GIRO",
}

ORDEN_MOVER  = ["hombro", "codo", "muneca", "rotacion", "base"]

WARMUP_FRAMES   = 20
MODELO_YOLO_SEG = "yolov8n-seg.pt"
CLASES_PLATO = {"bowl", "cup", "plate", "dish"}
EROSION_BORDE_PX = 12

TOLERANCIA_H  = 15
TOLERANCIA_S  = 40
TOLERANCIA_V  = 85

UMBRAL_LLENO_PCT  = 10.0
UMBRAL_VACIO_PCT  =  5.0

# -----------------------------------------------------------------
#  NUEVOS PARAMETROS — deteccion de residuos y estancamiento
# -----------------------------------------------------------------

# Que fraccion del radio del plato define la "zona central"
# Pixeles de sopa dentro de este radio = sopa real
# Pixeles de sopa fuera de este radio  = posible residuo
RADIO_CENTRAL_FRACCION = 0.70   # 70% interior = zona liquida real

# Si el porcentaje de sopa en zona CENTRAL baja de este valor,
# y la zona periferica tiene mas sopa que la central, son residuos
UMBRAL_RESIDUO_CENTRAL_PCT = 8.0

# Estancamiento: si el porcentaje no baja mas de esto en X segundos -> fin
ESTANCAMIENTO_DELTA_PCT  = 2.0   # Cambio minimo para NO considerar estancado
ESTANCAMIENTO_SEGUNDOS   = 6.0   # Tiempo sin cambio para declarar estancamiento

CARPETA_SALIDA     = Path("calibracion_sopa")
ARCHIVO_REFERENCIA = CARPETA_SALIDA / "calibracion_plato.pkl"


# =================================================================
#  SERIAL
# =================================================================

def conectar_serial(port, baud):
    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial no instalado. Ejecuta: pip install pyserial")
        sys.exit(1)
    print(f"[Serial] Conectando {port} @ {baud} bps...")
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2.0)
    ser.flushInput()
    print("[Serial] Esperando confirmacion del Arduino...")
    deadline = time.time() + 20
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line in ("READY", "BRAZO LISTO"):
                print(f"[Serial] Arduino listo ('{line}').")
                return ser
    print("[Serial] Sin confirmacion en 20s — continuando de todas formas.")
    return ser


def _enviar(ser, cmd, simulate=False, timeout=25):
    if simulate:
        print(f"  [SIM] -> {cmd}")
        time.sleep(0.06)
        return "OK"
    ser.flushInput()
    ser.write((cmd + "\n").encode())
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line == "OK":
                return "OK"
            if line == "ERR":
                return "ERR"
    print(f"  [TIMEOUT] Sin respuesta para: {cmd}")
    return "TIMEOUT"


def obtener_posicion_arduino(ser, simulate=False):
    pos = {ax: 0 for ax in HOME_POSITION}
    if simulate:
        pos = dict(HOME_POSITION)
        print("  [SIM] Posicion asumida: HOME.")
        return pos
    ser.flushInput()
    ser.write(b"POSICION\n")
    time.sleep(0.35)
    deadline = time.time() + 2.5
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if ":" not in line:
                continue
            clave, _, valor = line.partition(":")
            clave  = clave.strip().upper()
            valor  = valor.strip()
            if not valor.lstrip("-").isdigit():
                continue
            if clave == "BASE":       pos["base"]     = int(valor)
            elif clave == "HOMBRO":   pos["hombro"]   = int(valor)
            elif clave == "CODO":     pos["codo"]     = int(valor)
            elif clave == "GRIPPER":  pos["muneca"]   = int(valor)
            elif clave == "GIRO":     pos["rotacion"] = int(valor)
    print(f"  [Serial] Posicion actual consultada: {pos}")
    return pos


def mover_a_posicion(ser, posicion_objetivo, simulate=False):
    print(f"\n[Brazo] Consultando posicion actual...")
    pos_actual = obtener_posicion_arduino(ser, simulate=simulate)
    nombre_destino = (
        "POSICION CAMARA PLATO"
        if posicion_objetivo == POSICION_CAMARA_PLATO
        else "POSICION OBJETIVO"
    )
    print(f"[Brazo] Moviendo a {nombre_destino}...")
    movio = False
    for ax in ORDEN_MOVER:
        tgt  = posicion_objetivo[ax]
        cur  = pos_actual[ax]
        diff = tgt - cur
        if diff == 0:
            print(f"  {ax:10s}: ya en {cur:+d} (sin movimiento).")
            continue
        print(f"  {ax:10s}: {cur:+d} -> {tgt:+d}  (delta {diff:+d} pasos)")
        resp = _enviar(ser, f"{AXIS_CMD[ax]} {diff}", simulate=simulate)
        if resp != "OK":
            print(f"  [Advertencia] Respuesta inesperada en {ax}: {resp}")
        movio = True
        time.sleep(0.15)
    if not movio:
        print(f"[Brazo] Ya estaba en {nombre_destino}.")
    else:
        print(f"[Brazo] {nombre_destino} alcanzada.")


# =================================================================
#  CAMARA
# =================================================================

def abrir_camara(idx, warmup):
    print(f"\n[Cam1] Abriendo camara indice {idx}...")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {idx}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[Cam1] Descartando {warmup} frames de warmup...")
    for i in range(warmup):
        cap.read()
        sys.stdout.write(f"  {i+1}/{warmup}\r")
        sys.stdout.flush()
    print("\n[Cam1] Camara lista.")
    return cap


# =================================================================
#  YOLO — segmentacion dinamica del plato
# =================================================================

def segmentar_plato(frame, model, erosion_px=EROSION_BORDE_PX):
    fh, fw = frame.shape[:2]
    mascara_total = np.zeros((fh, fw), dtype=np.uint8)

    results = model(frame, verbose=False)[0]

    if results.masks is None:
        return mascara_total

    for i in range(len(results.masks.data)):
        cls_id     = int(results.boxes.cls[i])
        cls_nombre = model.names[cls_id].lower()

        if cls_nombre not in CLASES_PLATO:
            continue

        mask_raw   = results.masks.data[i].cpu().numpy()
        mask_frame = cv2.resize(mask_raw, (fw, fh), interpolation=cv2.INTER_LINEAR)
        mask_uint8 = (mask_frame * 255).astype(np.uint8)
        mascara_total = cv2.bitwise_or(mascara_total, mask_uint8)

    if erosion_px > 0 and np.any(mascara_total):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1)
        )
        mascara_total = cv2.erode(mascara_total, kernel, iterations=1)

    return mascara_total


# =================================================================
#  MASCARA MANUAL
# =================================================================

_roi_pts   = []
_dibujando = False

def _mouse_roi(event, x, y, flags, param):
    global _roi_pts, _dibujando
    if event == cv2.EVENT_LBUTTONDOWN:
        _dibujando = True
        _roi_pts   = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and _dibujando:
        if len(_roi_pts) == 1:
            _roi_pts.append((x, y))
        else:
            _roi_pts[1] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and _dibujando:
        _dibujando = False
        if len(_roi_pts) == 2:
            param["roi"] = (
                min(_roi_pts[0][0], _roi_pts[1][0]),
                min(_roi_pts[0][1], _roi_pts[1][1]),
                max(_roi_pts[0][0], _roi_pts[1][0]),
                max(_roi_pts[0][1], _roi_pts[1][1]),
            )


def mascara_desde_roi(shape, roi):
    m = np.zeros(shape[:2], dtype=np.uint8)
    if roi and (roi[2] - roi[0]) > 10 and (roi[3] - roi[1]) > 10:
        cv2.rectangle(m, (roi[0], roi[1]), (roi[2], roi[3]), 255, -1)
        if EROSION_BORDE_PX > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (EROSION_BORDE_PX * 2 + 1, EROSION_BORDE_PX * 2 + 1)
            )
            m = cv2.erode(m, k, iterations=1)
    return m


# =================================================================
#  ANALISIS DE COLOR
# =================================================================

def capturar_color_referencia(frame, mascara):
    if not np.any(mascara):
        return None
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixeles = hsv[mascara > 0].astype(np.float32)
    if len(pixeles) == 0:
        return None
    return {
        "H_mean": float(np.mean(pixeles[:, 0])),
        "H_std":  float(np.std( pixeles[:, 0])),
        "S_mean": float(np.mean(pixeles[:, 1])),
        "S_std":  float(np.std( pixeles[:, 1])),
        "V_mean": float(np.mean(pixeles[:, 2])),
        "V_std":  float(np.std( pixeles[:, 2])),
        "n_pixels": int(len(pixeles)),
    }


def calcular_porcentaje_sopa(frame, mascara, referencia,
                              tol_h=TOLERANCIA_H,
                              tol_s=TOLERANCIA_S,
                              tol_v=TOLERANCIA_V):
    pixels_totales = int(np.sum(mascara > 0))
    if pixels_totales == 0 or referencia is None:
        return 0.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    h_ref = referencia["H_mean"]
    s_ref = referencia["S_mean"]
    v_ref = referencia["V_mean"]

    dif_h = np.abs(hsv[:, :, 0] - h_ref)
    dif_h = np.minimum(dif_h, 180.0 - dif_h)
    dif_s = np.abs(hsv[:, :, 1] - s_ref)
    dif_v = np.abs(hsv[:, :, 2] - v_ref)

    es_sopa = (dif_h > tol_h) | (dif_s > tol_s) | (dif_v > tol_v)

    pixels_sopa = int(np.sum(es_sopa & (mascara > 0)))
    return float((pixels_sopa / pixels_totales) * 100.0)


# =================================================================
#  NUEVA LOGICA — filtro de residuos por distribucion radial
# =================================================================

def calcular_centro_mascara(mascara):
    """
    Calcula el centroide geometrico de la mascara del plato.
    Retorna (cx, cy) en pixeles.
    """
    M = cv2.moments(mascara)
    if M["m00"] == 0:
        h, w = mascara.shape
        return w // 2, h // 2
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def calcular_radio_mascara(mascara, cx, cy):
    """
    Estima el radio promedio de la mascara del plato desde su centroide.
    Usa los pixeles activos para calcular la distancia media al borde.
    """
    ys, xs = np.where(mascara > 0)
    if len(xs) == 0:
        return 1.0
    distancias = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    # Usamos el percentil 90 para evitar outliers de la forma irregular
    return float(np.percentile(distancias, 90))


def filtrar_residuos_radial(mascara_sopa_raw, mascara_plato,
                             radio_central_frac=RADIO_CENTRAL_FRACCION,
                             umbral_central_pct=UMBRAL_RESIDUO_CENTRAL_PCT):
    """
    Analiza si los pixeles de sopa detectados corresponden a sopa real
    o son residuos/rastros en las paredes del plato.

    Logica:
      - Calcula el centro y radio del plato.
      - Divide la zona del plato en zona CENTRAL (radio_central_frac del radio)
        y zona PERIFERIA (el anillo exterior).
      - Si la sopa esta mayormente en la periferia y el centro esta vacio
        -> son residuos -> retorna mascara corregida sin ellos.
      - Si la sopa ocupa bien la zona central -> es sopa real -> no toca nada.

    Retorna:
      mascara_sopa_filtrada : mascara corregida (residuos eliminados)
      es_residuo            : True si se detecto patron de residuo
      info                  : dict con metricas para el HUD
    """
    pixels_plato = int(np.sum(mascara_plato > 0))
    pixels_sopa  = int(np.sum(mascara_sopa_raw > 0))

    if pixels_plato == 0 or pixels_sopa == 0:
        return mascara_sopa_raw, False, {"pct_central": 0.0, "pct_periferia": 0.0}

    # Centro y radio del plato
    cx, cy = calcular_centro_mascara(mascara_plato)
    radio  = calcular_radio_mascara(mascara_plato, cx, cy)
    radio_central = radio * radio_central_frac

    # Mapa de distancias al centro para toda la imagen
    h, w = mascara_plato.shape
    ys, xs = np.mgrid[0:h, 0:w]
    dist_map = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    # Zonas dentro del plato
    zona_central   = (mascara_plato > 0) & (dist_map <= radio_central)
    zona_periferia = (mascara_plato > 0) & (dist_map >  radio_central)

    # Cuantos pixeles de sopa hay en cada zona
    sopa_central   = int(np.sum(mascara_sopa_raw[zona_central]   > 0))
    sopa_periferia = int(np.sum(mascara_sopa_raw[zona_periferia] > 0))

    px_zona_central   = int(np.sum(zona_central))
    px_zona_periferia = int(np.sum(zona_periferia))

    pct_central   = (sopa_central   / px_zona_central)   * 100.0 if px_zona_central   > 0 else 0.0
    pct_periferia = (sopa_periferia / px_zona_periferia) * 100.0 if px_zona_periferia > 0 else 0.0

    info = {
        "pct_central":    round(pct_central,   1),
        "pct_periferia":  round(pct_periferia, 1),
        "cx": cx, "cy": cy,
        "radio": round(radio, 1),
        "radio_central": round(radio_central, 1),
    }

    # Condicion de residuo:
    # La zona central tiene muy poca sopa Y la periferia tiene bastante
    es_residuo = (pct_central < umbral_central_pct) and (pct_periferia > pct_central * 1.5)

    if es_residuo:
        # Devolvemos mascara vacia: los rastros no cuentan como sopa
        mascara_filtrada = np.zeros_like(mascara_sopa_raw)
    else:
        mascara_filtrada = mascara_sopa_raw

    return mascara_filtrada, es_residuo, info


# =================================================================
#  NUEVA LOGICA — detector de estancamiento temporal
# =================================================================

class DetectorEstancamiento:
    """
    Detecta cuando el nivel de sopa deja de bajar durante varios segundos,
    lo que indica que el brazo ya no puede extraer mas liquido aunque
    queden rastros visibles.
    """

    def __init__(self, delta_pct=ESTANCAMIENTO_DELTA_PCT,
                 segundos=ESTANCAMIENTO_SEGUNDOS):
        self.delta_pct  = delta_pct
        self.segundos   = segundos
        self._pct_ref   = None      # Porcentaje en el momento que empezo el timer
        self._t_inicio  = None      # Cuando empezo el periodo sin cambio
        self.estancado  = False

    def actualizar(self, pct_actual):
        """
        Llama en cada frame con el porcentaje actual de sopa.
        Retorna True si el nivel lleva mas de `segundos` sin bajar mas de `delta_pct`.
        """
        ahora = time.time()

        if self._pct_ref is None:
            # Primera llamada: inicializar
            self._pct_ref  = pct_actual
            self._t_inicio = ahora
            self.estancado = False
            return False

        cambio = abs(pct_actual - self._pct_ref)

        if cambio >= self.delta_pct:
            # Hubo cambio significativo: reiniciar timer
            self._pct_ref  = pct_actual
            self._t_inicio = ahora
            self.estancado = False
        else:
            # Sin cambio: verificar si supero el tiempo limite
            if (ahora - self._t_inicio) >= self.segundos:
                self.estancado = True

        return self.estancado

    def resetear(self):
        self._pct_ref  = None
        self._t_inicio = None
        self.estancado = False


# =================================================================
#  GUARDAR / CARGAR REFERENCIA
# =================================================================

def guardar_referencia(referencia_hsv, mascara_ref, frame_ref, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)

    datos = {
        "timestamp":      datetime.now().isoformat(),
        "referencia_hsv": referencia_hsv,
        "mascara_ref":    mascara_ref,
        "frame_ref":      frame_ref,
        "tolerancias": {
            "H": TOLERANCIA_H,
            "S": TOLERANCIA_S,
            "V": TOLERANCIA_V,
        },
        "umbrales_pct": {
            "lleno":  UMBRAL_LLENO_PCT,
            "vacio":  UMBRAL_VACIO_PCT,
        },
        "erosion_borde_px": EROSION_BORDE_PX,
    }

    pkl_path = carpeta / "calibracion_plato.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(datos, f)
    print(f"[Guardado] PKL -> {pkl_path}")

    datos_json = {
        "timestamp":      datos["timestamp"],
        "referencia_hsv": referencia_hsv,
        "tolerancias":    datos["tolerancias"],
        "umbrales_pct":   datos["umbrales_pct"],
        "erosion_borde_px": EROSION_BORDE_PX,
        "nota": (
            "Referencia capturada con el plato VACIO. "
            "El sistema compara cada frame contra este color de fondo. "
            "Si >UMBRAL_LLENO_PCT% de pixels difieren del fondo -> hay sopa. "
            "Si <UMBRAL_VACIO_PCT% de pixels difieren del fondo -> plato vacio."
        ),
    }
    json_path = carpeta / "calibracion_plato_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(datos_json, f, indent=2, ensure_ascii=False)
    print(f"[Guardado] JSON -> {json_path}")

    vis = frame_ref.copy()
    contornos, _ = cv2.findContours(mascara_ref, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contornos, -1, (0, 255, 0), 2)
    hsv_str = (f"H={referencia_hsv['H_mean']:.1f}  "
               f"S={referencia_hsv['S_mean']:.1f}  "
               f"V={referencia_hsv['V_mean']:.1f}")
    cv2.putText(vis, "REFERENCIA PLATO VACIO", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(vis, hsv_str, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    png_path = carpeta / "calibracion_plato_ref.png"
    cv2.imwrite(str(png_path), vis)
    print(f"[Guardado] PNG  -> {png_path}")

    return pkl_path


def cargar_referencia(ruta=ARCHIVO_REFERENCIA):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro la referencia en '{ruta}'.\n"
            f"Ejecuta primero: python calibrar_plato_sopa.py --calibrar"
        )
    with open(ruta, "rb") as f:
        datos = pickle.load(f)
    return datos


def imprimir_resumen_referencia(referencia_hsv, datos):
    print(f"\n{'=' * 60}")
    print(f"  REFERENCIA PLATO VACIO CAPTURADA")
    print(f"{'=' * 60}")
    print(f"  H (tono)       : {referencia_hsv['H_mean']:.1f}  "
          f"(std ± {referencia_hsv['H_std']:.1f})")
    print(f"  S (saturacion) : {referencia_hsv['S_mean']:.1f}  "
          f"(std ± {referencia_hsv['S_std']:.1f})")
    print(f"  V (brillo)     : {referencia_hsv['V_mean']:.1f}  "
          f"(std ± {referencia_hsv['V_std']:.1f})")
    print(f"  Pixels en ROI  : {referencia_hsv['n_pixels']}")
    print(f"{'=' * 60}")
    print(f"  Plato vacio tipico: S bajo (~10-30), V alto (~180-240).")
    print(f"  Si S_mean > 50 la calibracion puede tener restos de sopa.")
    print(f"{'=' * 60}\n")


# =================================================================
#  BUCLE DE CALIBRACION
# =================================================================

def bucle_calibracion(cap, model, carpeta):
    ventana = "CALIBRAR plato VACIO — ESPACIO=guardar  M=manual  ESC=cancelar"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)

    estado_roi = {"roi": None}
    cv2.setMouseCallback(ventana, _mouse_roi, estado_roi)

    modo_manual    = False
    ref_hsv_actual = None
    mascara_actual = None
    frame_actual   = None

    print(f"\n[Calib] Mostrando cam1. Espera a que YOLO detecte el plato.")
    print(f"        Presiona ESPACIO cuando la deteccion cubra bien el plato.")
    print(f"        Si YOLO no detecta el plato presiona M y dibuja el area.")
    print(f"        ESC para cancelar.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fh, fw = frame.shape[:2]

        if modo_manual and estado_roi["roi"]:
            mascara = mascara_desde_roi(frame.shape, estado_roi["roi"])
        else:
            mascara = segmentar_plato(frame, model)
            if np.any(mascara):
                modo_manual = False

        ref_actual = capturar_color_referencia(frame, mascara)

        if np.any(mascara) and ref_actual is not None:
            ref_hsv_actual = ref_actual
            mascara_actual = mascara.copy()
            frame_actual   = frame.copy()

        display = frame.copy()

        if np.any(mascara):
            capa = np.zeros_like(frame)
            capa[mascara > 0] = (50, 220, 50)
            display = cv2.addWeighted(display, 0.6, capa, 0.4, 0)

            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contornos, -1, (0, 255, 0), 2)

            area_px = int(np.sum(mascara > 0))
            cv2.putText(display, f"Plato detectado  {area_px} px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            if ref_actual:
                cv2.putText(display,
                            f"HSV ref: H={ref_actual['H_mean']:.0f} "
                            f"S={ref_actual['S_mean']:.0f} "
                            f"V={ref_actual['V_mean']:.0f}",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 220, 255), 2)
                if ref_actual["S_mean"] > 50:
                    cv2.putText(display, "ADVERTENCIA: S alto — hay restos de sopa?",
                                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 80, 255), 2)
        else:
            sin_det = "YOLO: sin deteccion" if not modo_manual else "Dibuja el plato con el mouse"
            cv2.putText(display, sin_det,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2)

        if _dibujando and len(_roi_pts) == 2:
            cv2.rectangle(display, _roi_pts[0], _roi_pts[1], (255, 255, 0), 1)

        modo_str = "[MANUAL]" if modo_manual else "[YOLO]"
        cv2.putText(display, f"{modo_str}  ESPACIO=guardar  M=manual  ESC=cancelar",
                    (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                    (180, 180, 180), 1)

        cv2.imshow(ventana, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("\n[Calib] Cancelado por el usuario.")
            cv2.destroyWindow(ventana)
            return False
        elif key in (ord("m"), ord("M")):
            modo_manual       = True
            estado_roi["roi"] = None
            _roi_pts.clear()
            print("[Calib] Modo manual: dibuja el rectangulo sobre el plato.")
        elif key == 32:
            if mascara_actual is not None and ref_hsv_actual is not None:
                imprimir_resumen_referencia(ref_hsv_actual, {})
                guardar_referencia(ref_hsv_actual, mascara_actual, frame_actual, carpeta)
                confirm = frame_actual.copy()
                cv2.putText(confirm, "CALIBRACION GUARDADA",
                            (10, confirm.shape[0] // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.imshow(ventana, confirm)
                cv2.waitKey(2000)
                cv2.destroyWindow(ventana)
                return True
            else:
                print("[Calib] Sin deteccion valida aun.")

    return False


# =================================================================
#  BUCLE DE DETECCION — con filtro de residuos y estancamiento
# =================================================================

def bucle_deteccion(cap, model, referencia):
    ref_hsv     = referencia["referencia_hsv"]
    tolerancias = referencia.get("tolerancias", {
        "H": TOLERANCIA_H, "S": TOLERANCIA_S, "V": TOLERANCIA_V
    })
    tol_h = tolerancias.get("H", TOLERANCIA_H)
    tol_s = tolerancias.get("S", TOLERANCIA_S)
    tol_v = tolerancias.get("V", TOLERANCIA_V)
    umbral_vacio = referencia.get("umbrales_pct", {}).get("vacio", UMBRAL_VACIO_PCT)

    # Instancia del detector de estancamiento
    detector_estanc = DetectorEstancamiento()

    ventana = "DETECCION en tiempo real — ESC para salir"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)

    print(f"\n[Detect] Mostrando deteccion en tiempo real.")
    print(f"         Referencia: H={ref_hsv['H_mean']:.1f} "
          f"S={ref_hsv['S_mean']:.1f} V={ref_hsv['V_mean']:.1f}")
    print(f"         Umbral vacio: < {umbral_vacio:.1f}%")
    print(f"         Filtro residuos: zona central {RADIO_CENTRAL_FRACCION*100:.0f}% del radio")
    print(f"         Estancamiento: sin cambio de {ESTANCAMIENTO_DELTA_PCT}% "
          f"en {ESTANCAMIENTO_SEGUNDOS}s -> detener")
    print(f"         ESC para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fh, fw = frame.shape[:2]
        display = frame.copy()

        mascara = segmentar_plato(frame, model)

        pct            = 0.0
        es_residuo     = False
        info_radial    = {}
        estancado      = False

        if np.any(mascara):
            # 1. Calcular mapa de diferencia HSV (igual que antes)
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            dif_h = np.abs(hsv[:, :, 0] - ref_hsv["H_mean"])
            dif_h = np.minimum(dif_h, 180.0 - dif_h)
            dif_s = np.abs(hsv[:, :, 1] - ref_hsv["S_mean"])
            dif_v = np.abs(hsv[:, :, 2] - ref_hsv["V_mean"])
            es_sopa_mapa = ((dif_h > tol_h) | (dif_s > tol_s) | (dif_v > tol_v)).astype(np.uint8) * 255

            # Mascara cruda de sopa (solo dentro del plato)
            mascara_sopa_raw = cv2.bitwise_and(es_sopa_mapa, es_sopa_mapa, mask=mascara)

            # 2. NUEVO: filtrar residuos por distribucion radial
            mascara_sopa_filtrada, es_residuo, info_radial = filtrar_residuos_radial(
                mascara_sopa_raw, mascara
            )

            # 3. Calcular porcentaje con la mascara ya filtrada
            pixels_totales = int(np.sum(mascara > 0))
            pixels_sopa    = int(np.sum(mascara_sopa_filtrada > 0))
            pct = (pixels_sopa / pixels_totales * 100.0) if pixels_totales > 0 else 0.0

            # 4. NUEVO: detectar estancamiento
            estancado = detector_estanc.actualizar(pct)

            # ── Overlays visuales ──────────────────────────────────
            zona_plato = mascara > 0
            es_sopa_bool = mascara_sopa_filtrada > 0

            capa_sopa     = np.zeros_like(frame)
            capa_ceramica = np.zeros_like(frame)
            capa_residuo  = np.zeros_like(frame)

            capa_sopa[zona_plato & es_sopa_bool]      = (0,  50, 220)   # Rojo  = sopa real
            capa_ceramica[zona_plato & ~es_sopa_bool] = (50, 180,  50)  # Verde = ceramica
            # Residuo: lo que la mascara cruda tenia pero la filtrada descarto
            zona_residuo = (mascara_sopa_raw > 0) & (~es_sopa_bool) & zona_plato
            capa_residuo[zona_residuo] = (0, 200, 200)                  # Amarillo = residuo descartado

            display = cv2.addWeighted(display, 0.55, capa_sopa,     0.30, 0)
            display = cv2.addWeighted(display, 1.00, capa_ceramica,  0.15, 0)
            display = cv2.addWeighted(display, 1.00, capa_residuo,   0.25, 0)

            # Dibujar circulo de zona central (referencia visual)
            if info_radial:
                cx = info_radial["cx"]
                cy = info_radial["cy"]
                rc = int(info_radial["radio_central"])
                cv2.circle(display, (cx, cy), rc, (200, 200, 0), 1)  # circulo zona central

            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contornos, -1, (0, 255, 0), 2)

        else:
            # Sin plato: resetear estancamiento
            detector_estanc.resetear()

        # ── Barra de porcentaje ────────────────────────────────────
        barra_w    = int(fw * 0.6)
        barra_fill = int(barra_w * min(pct, 100.0) / 100.0)
        barra_y    = 85
        cv2.rectangle(display, (10, barra_y), (10 + barra_w, barra_y + 18),
                      (60, 60, 60), -1)
        color_barra = (0, 50, 220) if pct > umbral_vacio else (50, 200, 50)
        if barra_fill > 0:
            cv2.rectangle(display, (10, barra_y),
                          (10 + barra_fill, barra_y + 18), color_barra, -1)

        # ── HUD de texto ───────────────────────────────────────────
        cv2.putText(display, f"Sopa detectada: {pct:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Estado principal
        if estancado:
            estado      = "--- SIN CAMBIO: DETENER ---"
            color_estado = (0, 140, 255)   # Naranja
        elif es_residuo:
            estado      = "RESIDUO DETECTADO (ignorado)"
            color_estado = (0, 200, 200)   # Amarillo
        elif pct <= umbral_vacio:
            estado      = "--- VACIO ---"
            color_estado = (50, 220, 50)   # Verde
        else:
            estado      = "CON SOPA"
            color_estado = (0, 50, 220)    # Rojo

        cv2.putText(display, estado, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_estado, 2)

        # Info radial (zona central vs periferia)
        if info_radial:
            cv2.putText(display,
                        f"Central: {info_radial['pct_central']:.1f}%  "
                        f"Periferia: {info_radial['pct_periferia']:.1f}%",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1)

        cv2.putText(display, f"Umbral vacio < {umbral_vacio:.0f}%  |  ESC=salir",
                    (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)

        cv2.imshow(ventana, display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow(ventana)


# =================================================================
#  MODO SIMULACION
# =================================================================

def ejecutar_simulacion_calibracion(carpeta):
    print("\n[SIM] Ejecutando calibracion simulada...")
    fh, fw = 480, 640
    mascara_sim = np.zeros((fh, fw), dtype=np.uint8)
    cv2.ellipse(mascara_sim, (fw // 2, fh // 2), (180, 160), 0, 0, 360, 255, -1)
    if EROSION_BORDE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (EROSION_BORDE_PX * 2 + 1, EROSION_BORDE_PX * 2 + 1)
        )
        mascara_sim = cv2.erode(mascara_sim, k)
    frame_sim = np.full((fh, fw, 3), (215, 215, 220), dtype=np.uint8)
    cv2.putText(frame_sim, "SIMULACION PLATO VACIO", (80, fh // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    ref_hsv_sim = {
        "H_mean": 102.0, "H_std": 18.0,
        "S_mean":  12.0, "S_std":  5.0,
        "V_mean": 215.0, "V_std":  8.0,
        "n_pixels": int(np.sum(mascara_sim > 0)),
    }
    imprimir_resumen_referencia(ref_hsv_sim, {})
    guardar_referencia(ref_hsv_sim, mascara_sim, frame_sim, carpeta)
    print(f"[SIM] Listo. Archivos en: {carpeta.resolve()}\n")


# =================================================================
#  MAIN
# =================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calibracion y deteccion de sopa con YOLO segmentacion.\n"
            "Usa --calibrar para capturar la referencia del plato vacio.\n"
            "Usa --detectar para ver el porcentaje de llenado en tiempo real."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    accion = parser.add_mutually_exclusive_group(required=True)
    accion.add_argument("--calibrar", action="store_true",
                        help="Captura la referencia del plato VACIO (una sola vez).")
    accion.add_argument("--detectar", action="store_true",
                        help="Muestra el porcentaje de sopa en tiempo real.")

    parser.add_argument("--cam1",   type=int, default=CAMERA_1_INDEX)
    parser.add_argument("--warmup", type=int, default=WARMUP_FRAMES)
    parser.add_argument("--yolo",   default=MODELO_YOLO_SEG)
    parser.add_argument("--salida", default=str(CARPETA_SALIDA))
    args = parser.parse_args()

    carpeta = Path(args.salida)

    print(f"\n[YOLO] Cargando {args.yolo}...")
    model = YOLO(args.yolo)
    print("[YOLO] Listo.")

    cap = abrir_camara(args.cam1, warmup=args.warmup)

    if args.calibrar:
        print(f"\n{'=' * 60}")
        print(f"  CALIBRACION - Plato VACIO")
        print(f"  Asegurate de que el plato este completamente VACIO.")
        print(f"{'=' * 60}")
        exito = bucle_calibracion(cap, model, carpeta)
        if exito:
            print(f"\nCalibracion completada. Archivos en: {carpeta.resolve()}")
            print(f"Para verificar: python calibrar_plato_sopa.py --detectar\n")
        else:
            print("\nCalibracion no completada.\n")

    elif args.detectar:
        print(f"\n{'=' * 60}")
        print(f"  DETECCION - Verificacion en tiempo real")
        print(f"{'=' * 60}")
        try:
            referencia = cargar_referencia(carpeta / "calibracion_plato.pkl")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            cap.release()
            sys.exit(1)
        bucle_deteccion(cap, model, referencia)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n\n[!] Interrumpido por el usuario (Ctrl+C).")
        sys.exit(0)
    finally:
        cv2.destroyAllWindows()