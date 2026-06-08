import argparse
import json
import os
import sys
import time
import signal
import cv2
import numpy as np

DIR_BASE    = os.path.dirname(os.path.abspath(__file__))
ESTADO_JSON = os.path.join(DIR_BASE, "estado.json")

CAM2_FRAME_TMP       = os.path.join(DIR_BASE, "nutribot_cam2_frame.npy")
CAM2_FRAME_TMP_WRITE = os.path.join(DIR_BASE, "nutribot_cam2_frame.tmp.npy")

sys.path.insert(0, DIR_BASE)

ARCHIVO_CALIB_SOPA = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_FINAL\calibracion.pkl"
MODELO_YOLO_SEG    = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_FINAL\yolov8n-seg.pt"
UMBRAL_VACIO_PORC  = 85.0
MAX_CUCHARADAS     = 25

CAMARA_SOPA_INDEX      = 2
CAMARA_MEDIAPIPE_INDEX = 1

ARUCO_ID_BUSCADO = 1
ARUCO_DICT       = cv2.aruco.DICT_4X4_50
ARUCO_ESPERA_SEG = 3.0
ARUCO_REINTENTOS = 2

AGARRE_BASE   = -195
AGARRE_CODO   = +1430
AGARRE_HOMBRO = +1200
BAJAR_HOMBRO  = +1340
SUBIR_HOMBRO  = -1200
SUBIR_BASE    = +400

HOME2_ABSOLUTO = {"base": 950, "hombro": 1650, "codo": 2100, "muneca": 0}

PLATO_BASE     = -550
PLATO_HOMBRO   = +110
PLATO_CODO     = -580
PLATO_MUNECA   = -1000
SUMERGE_HOMBRO = +350
LEVANTA_CODO   = +400
LEVANTA_HOMBRO = -500
LEVANTA_BASE   = -425

LIMITE_CODO_BOCA        = 4000     # tope físico del codo (solo afecta la búsqueda; el posicionamiento ya no mueve el codo)
# --- Búsqueda de rostro (F0): igual que modo sólido, 20 intentos ---
CODO_BUSQUEDA_PASOS_SOPA = 700     # codo +700 en cada intento
GRIPPER_BUSQUEDA_SOPA    = -300    # gripper (muneca) -300 en cada intento
CODO_BUSQUEDA_MAX_SOPA   = 20      # hasta 20 veces, como modo sólido

# --- Posicionamiento (FA): tras detectar rostro, UN solo movimiento hombro/gripper ---
HOMBRO_POS_SOPA      = -300        # hombro -300 (una vez)
GRIPPER_POS_SOPA     = -200        # gripper (muneca) -200 (una vez)
EATING_CM_SOPA       = 13
FAR_CM_SOPA          = 25
EATING_WAIT_SOPA     = 6.0
BOCA_CONFIRM_SOPA    = 1.5
TIMEOUT_ENTREGA_SOPA = 90

CODO_BUSQUEDA_PASOS_SOL = 200
CODO_BUSQUEDA_MAX_SOL   = 20
APPROACH_SEQ_SOL        = [("hombro", +400), ("codo", +600)]
MAX_APPROACH_MOVES_SOL  = 30
ALARM_RAISE_SOL         = 500
EATING_WAIT_SOL         = 5.0
BOCA_AUSENTE_FIN_SOL    = 3.0
TIMEOUT_ENTREGA_SOL     = 90

def leer_estado():
    for intento in range(3):
        try:
            with open(ESTADO_JSON, "r", encoding="utf-8") as f:
                contenido = f.read()
            if contenido.strip():
                return json.loads(contenido)
        except (json.JSONDecodeError, ValueError):
            if intento < 2:
                time.sleep(0.005)
        except Exception:
            break
    return {}

def _guardar_estado(estado):
    tmp = ESTADO_JSON + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=2, ensure_ascii=False)
        os.replace(tmp, ESTADO_JSON)
    except Exception as e:
        print(f"[ORQ] Error guardando estado: {e}", flush=True)

def escribir_fase(fase, ciclo=None, agarres_ok=None):
    estado = leer_estado() or {}
    estado["fase_actual"] = fase
    if ciclo is not None:
        estado["ciclo"] = ciclo
    if agarres_ok is not None:
        estado["agarres_ok"] = agarres_ok
    _guardar_estado(estado)

def escribir_voz(texto):
    estado = leer_estado() or {}
    estado["voz"] = texto
    _guardar_estado(estado)
    print(f"[VOZ] >>> '{texto}'", flush=True)

def escribir_estado_campo(campo, valor):
    estado = leer_estado() or {}
    estado[campo] = valor
    _guardar_estado(estado)

def hay_comando(cmd):
    return leer_estado().get("comando", "") == cmd

def esperar_si_pausa():
    while hay_comando("PAUSA"):
        time.sleep(0.5)

def debe_apagar():
    return leer_estado().get("comando", "") == "DETENER"

def debe_detener():
    return leer_estado().get("comando", "") in ("DETENER", "PARAR")

def importar_modulo_principal():
    try:
        import importlib
        m = importlib.import_module("auto_brazo_completo_PRESENTACION")
        print("[ORQ] Módulo 'auto_brazo_completo_PRESENTACION' importado.", flush=True)
        return m
    except ModuleNotFoundError:
        print("[ORQ] AVISO: auto_brazo_completo_PRESENTACION.py no encontrado.",
              flush=True)
        return None
    except Exception as e:
        print(f"[ORQ] Error importando módulo: {e}", flush=True)
        return None

def _mover_clamped(robot, eje, pasos, limite_codo=None):
    if pasos == 0:
        return
    if eje == "codo" and limite_codo is not None:
        cur   = robot._pos.get("codo", 0)
        nuevo = cur + pasos
        if nuevo > limite_codo:
            pasos = limite_codo - cur
            if pasos <= 0:
                print(f"    [LÍMITE] codo ya en {cur}", flush=True)
                return
            print(f"    [LÍMITE] codo -> {cur + pasos}", flush=True)
    robot.mover_eje(eje, pasos)

def _mover_absoluto(robot, eje, objetivo, limite_codo=None):
    delta = objetivo - robot._pos.get(eje, 0)
    if delta == 0:
        print(f"    {eje}: ya en {objetivo}", flush=True)
        return
    _mover_clamped(robot, eje, delta, limite_codo=limite_codo)

def _ir_a_home2(robot):
    print("\n[ORQ] -- Volviendo a HOME2 ------------------------------",
          flush=True)
    print(f"  _pos actual : {robot._pos}", flush=True)
    print(f"  objetivo    : {HOME2_ABSOLUTO}", flush=True)
    _mover_absoluto(robot, "muneca",  HOME2_ABSOLUTO["muneca"])
    _mover_absoluto(robot, "codo",    HOME2_ABSOLUTO["codo"])
    _mover_absoluto(robot, "hombro",  HOME2_ABSOLUTO["hombro"])
    _mover_absoluto(robot, "base",    HOME2_ABSOLUTO["base"])
    print(f"[ORQ] HOME2 OK. _pos={robot._pos}", flush=True)

class CamaraCompartida:
    def __init__(self, index):
        self.index = index
        self.cap   = None

    def abrir(self, ancho=1280, alto=720, warmup=5):
        self.cerrar()
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        if not self.cap.isOpened():
            print(f"[CAM-SOPA] ERROR al abrir índice {self.index}", flush=True)
            return False
        for _ in range(warmup):
            self.cap.read()
        print(f"[CAM-SOPA] Cámara {self.index} abierta.", flush=True)
        return True

    def leer(self):
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None

    def cerrar(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            print(f"[CAM-SOPA] Camara {self.index} cerrada.", flush=True)

def _esperar_cam1_disponible(timeout=10.0):
    print("[ORQ] Esperando cam1_owner=cuchara...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if leer_estado().get("cam1_owner", "") == "cuchara":
            break
        time.sleep(0.2)
    time.sleep(1.0)
    print("[ORQ] Cámara disponible.", flush=True)

class ArucoDetector:
    def __init__(self, camara: CamaraCompartida):
        self.camara   = camara
        aruco_dict    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        params        = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def _detectar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, ids, _ = self.detector.detectMarkers(gray)
        return ids is not None and ARUCO_ID_BUSCADO in ids

    def verificar_presencia(self, espera_seg=2.0):
        print(f"[ArUco] Buscando ID{ARUCO_ID_BUSCADO} durante {espera_seg}s...",
              flush=True)
        t0 = time.time()
        while time.time() - t0 < espera_seg:
            ret, frame = self.camara.leer()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            if self._detectar(frame):
                print(f"[ArUco] [OK] ID{ARUCO_ID_BUSCADO} presente.", flush=True)
                return True
            time.sleep(0.05)
        print(f"[ArUco] [FALLA] ID{ARUCO_ID_BUSCADO} NO detectado.", flush=True)
        return False

    def verificar_ausencia(self, espera_seg=ARUCO_ESPERA_SEG):
        print(f"[ArUco] Verificando ausencia de ID{ARUCO_ID_BUSCADO}...",
              flush=True)
        t0    = time.time()
        t_sin = None
        while time.time() - t0 < espera_seg:
            ret, frame = self.camara.leer()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            if self._detectar(frame):
                t_sin = None
            else:
                if t_sin is None:
                    t_sin = time.time()
                elif time.time() - t_sin >= 1.0:
                    print("[ArUco] [OK] Ausencia confirmada.", flush=True)
                    return True
            time.sleep(0.05)
        print("[ArUco] [FALLA] Marcador sigue visible.", flush=True)
        return False

def _ir_a_verificacion_aruco(robot):
    print("[ORQ] Moviendo a posición ArUco: BASE-195, CODO+1430, HOMBRO+1200",
          flush=True)
    robot.mover_eje("base",   AGARRE_BASE)
    robot.mover_eje("codo",   AGARRE_CODO)
    robot.mover_eje("hombro", AGARRE_HOMBRO)
    print("[ORQ] Posición ArUco alcanzada.", flush=True)

def verificar_agarre_cuchara_post_subida(camara: CamaraCompartida):
    aruco = ArucoDetector(camara)
    print("\n[CUCHARA][5.5] Verificando agarre sin volver a HOME...", flush=True)
    time.sleep(0.5)
    return aruco.verificar_ausencia(espera_seg=ARUCO_ESPERA_SEG)

def _cargar_sopa_detector(camara: CamaraCompartida):
    try:
        import pickle
        from pathlib import Path
        from ultralytics import YOLO

        calib_path = Path(ARCHIVO_CALIB_SOPA)
        if not calib_path.exists():
            print(f"[SOPA] Calibración no encontrada: {ARCHIVO_CALIB_SOPA}",
                  flush=True)
            return None

        with open(calib_path, "rb") as f:
            datos = pickle.load(f)

        ref_plato  = datos["referencia_plato"]
        ref_sombra = datos.get("referencia_sombra", None)
        tol        = datos["tolerancias"]
        anti       = datos["anti_sombra"]
        model_seg  = YOLO(MODELO_YOLO_SEG)

        class _Detector:
            def medir_nivel(self_d):
                ret, frame = camara.leer()
                if not ret or frame is None:
                    return 0.0
                fh, fw  = frame.shape[:2]
                mascara = np.zeros((fh, fw), dtype=np.uint8)
                results = model_seg(frame, verbose=False, device="cpu")[0]
                if results.masks is not None:
                    for i in range(len(results.masks.data)):
                        cls_id = int(results.boxes.cls[i])
                        nombre = model_seg.names.get(cls_id, "").lower()
                        if nombre in {"bowl", "cup", "plate", "dish"}:
                            mk = results.masks.data[i].cpu().numpy()
                            mk = cv2.resize(mk, (fw, fh))
                            mascara = cv2.bitwise_or(
                                mascara, (mk * 255).astype(np.uint8))
                if not np.any(mascara):
                    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gris = cv2.GaussianBlur(gris, (11, 11), 0)
                    circ = cv2.HoughCircles(
                        gris, cv2.HOUGH_GRADIENT, 1.2, fh // 3,
                        param1=80, param2=40, minRadius=80, maxRadius=400)
                    if circ is not None:
                        cx, cy, r = circ[0][0]
                        cv2.circle(mascara, (int(cx), int(cy)),
                                   max(0, int(r) - 12), 255, -1)
                if not np.any(mascara):
                    return 0.0
                k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                mascara = cv2.erode(mascara, k, iterations=1)
                hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                h_c, s_c, v_c = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
                dh = np.minimum(
                    np.abs(h_c - ref_plato["H_mean"]),
                    180 - np.abs(h_c - ref_plato["H_mean"]))
                ds = np.abs(s_c - ref_plato["S_mean"])
                dv = np.abs(v_c - ref_plato["V_mean"])
                diferente = (dh > tol["H"]) | (ds > tol["S"]) | (dv > tol["V"])
                sombra = ((s_c < anti["S_max"]) & (v_c < anti["V_max"])
                          if ref_sombra is not None
                          else np.zeros_like(diferente))
                sopa  = (mascara > 0) & diferente & (~sombra)
                tot   = int(np.sum(mascara > 0))
                return 0.0 if tot == 0 else 100.0 * np.sum(sopa) / tot

        det = _Detector()
        print("[SOPA] Detector inicializado.", flush=True)
        return det
    except Exception as e:
        print(f"[SOPA] ERROR cargando detector: {e}", flush=True)
        return None

def _estimar_dist_sopa(landmarks, shape):
    h, w     = shape[:2]
    le       = landmarks.landmark[33]
    re       = landmarks.landmark[263]
    eye_dist = np.hypot((re.x - le.x) * w, (re.y - le.y) * h)
    nose     = landmarks.landmark[1]
    chin     = landmarks.landmark[152]
    nc       = np.hypot((chin.x - nose.x) * w, (chin.y - nose.y) * h)
    if eye_dist > 30:
        d = 5000.0 / eye_dist
    elif nc > 0:
        d = 2500.0 / nc
    else:
        d = 50.0
    return float(np.clip(d, 3.0, 65.0))

def _mediapipe_worker(cam_index, pipe_in, pipe_out):
    import sys, os, json, time, threading
    import cv2, numpy as np
    try:
        import mediapipe as mp
    except Exception:
        pipe_out.write('{"error": "mediapipe no instalado"}\n')
        pipe_out.flush()
        return

    base_dir       = os.path.dirname(os.path.abspath(__file__))
    frame_tmp       = os.path.join(base_dir, "nutribot_cam2_frame.npy")
    frame_tmp_write = os.path.join(base_dir, "nutribot_cam2_frame.tmp.npy")

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        pipe_out.write('{"error": "no se pudo abrir camara"}\n')
        pipe_out.flush()
        return

    ok_frames = 0
    for _ in range(40):
        ret, _f = cap.read()
        if ret and _f is not None:
            ok_frames += 1
        if ok_frames >= 15:
            break

    mp_face   = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    estado = {"dist": None, "fase": "buscando-rostro", "run": True}
    lock   = threading.Lock()

    pipe_out.write('{"ready": true}\n')
    pipe_out.flush()

    def _lector_stdin():
        for linea in pipe_in:
            linea = (linea or "").strip()
            if not linea:
                continue
            try:
                msg = json.loads(linea)
            except Exception:
                continue
            cmd = msg.get("cmd")
            if cmd == "quit":
                with lock:
                    estado["run"] = False
                return
            if cmd == "fase":
                with lock:
                    estado["fase"] = msg.get("v", estado["fase"])
            elif cmd == "dist":
                with lock:
                    d = estado["dist"]
                try:
                    pipe_out.write(json.dumps({"dist": d}) + "\n")
                    pipe_out.flush()
                except Exception:
                    with lock:
                        estado["run"] = False
                    return

    def _calc_dist(lm, shape):
        h, w = shape[:2]
        le   = lm.landmark[33];  re = lm.landmark[263]
        ed   = np.hypot((re.x - le.x) * w, (re.y - le.y) * h)
        nose = lm.landmark[1];   chin = lm.landmark[152]
        nc   = np.hypot((chin.x - nose.x) * w, (chin.y - nose.y) * h)
        if ed > 30:
            d = 5000.0 / ed
        elif nc > 0:
            d = 2500.0 / nc
        else:
            d = 50.0
        return float(np.clip(d, 3.0, 65.0))

    def _hud(frame, dist, fase):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 34), (32, 32, 32), -1)
        txt = "MediaPipe: " + fase
        if dist is not None:
            txt += "  |  dist=%.1f cm" % dist
        cv2.putText(frame, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _publicar(frame):
        try:
            np.save(frame_tmp_write, frame)
            os.replace(frame_tmp_write, frame_tmp)
        except Exception:
            pass

    hilo = threading.Thread(target=_lector_stdin, daemon=True)
    hilo.start()

    try:
        while True:
            with lock:
                if not estado["run"]:
                    break
                fase = estado["fase"]
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            dist = None
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm   = res.multi_face_landmarks[0]
                    dist = _calc_dist(lm, frame.shape)
                    h_f, w_f = frame.shape[:2]
                    ul = lm.landmark[13]; ll = lm.landmark[14]
                    bx = int((ul.x + ll.x) / 2 * w_f)
                    by = int((ul.y + ll.y) / 2 * h_f)
                    cv2.circle(frame, (bx, by), 10, (0, 255, 255), -1)
            except Exception:
                dist = None
            with lock:
                estado["dist"] = dist
            _publicar(_hud(frame, dist, fase))
    finally:
        try:
            face_mesh.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        for ruta in (frame_tmp, frame_tmp_write):
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
            except Exception:
                pass

class _MediaPipeProxy:

    _WORKER_SCRIPT = None

    @classmethod
    def _get_worker_script(cls):
        if cls._WORKER_SCRIPT is not None:
            import os
            if os.path.exists(cls._WORKER_SCRIPT):
                return cls._WORKER_SCRIPT
        import tempfile, os
        script = (
            "import sys, os\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            "# Importar directamente para no depender de nombre de modulo\n"
            "import importlib.util, types\n"
            "spec = importlib.util.spec_from_file_location('orq', sys.argv[1])\n"
            "mod  = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "mod._mediapipe_worker(int(sys.argv[2]), sys.stdin, sys.stdout)\n"
        )
        fd, path = tempfile.mkstemp(suffix=".py", prefix="mp_worker_")
        with os.fdopen(fd, "w") as f:
            f.write(script)
        cls._WORKER_SCRIPT = path
        return path

    def __init__(self, cam_index):
        import subprocess, sys, os
        self.cam_index = cam_index
        self._proc     = None
        self._ok       = False
        try:
            orq_path = os.path.abspath(__file__)
            script   = self._get_worker_script()
            self._proc = subprocess.Popen(
                [sys.executable, script, orq_path, str(cam_index)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            import time as _t, json as _j
            t0 = _t.time()
            while _t.time() - t0 < 8.0:
                if self._proc.poll() is not None:
                    break
                linea = self._proc.stdout.readline()
                if not linea:
                    _t.sleep(0.05)
                    continue
                msg = _j.loads(linea.strip())
                if msg.get("ready"):
                    self._ok = True
                    print(f"[MP-PROXY] Worker listo (cam {cam_index}).", flush=True)
                    break
                if msg.get("error"):
                    print(f"[MP-PROXY] Error: {msg['error']}", flush=True)
                    break
        except Exception as e:
            print(f"[MP-PROXY] No se pudo lanzar worker: {e}", flush=True)

    def set_fase(self, label):
        if not self._ok or self._proc is None or self._proc.poll() is not None:
            return
        try:
            import json as _j
            self._proc.stdin.write(_j.dumps({"cmd": "fase", "v": label}) + "\n")
            self._proc.stdin.flush()
        except Exception:
            self._ok = False

    def get_dist(self, timeout=0.30):
        import json as _j, time as _t
        if not self._ok or self._proc is None or self._proc.poll() is not None:
            self._ok = False
            return None
        try:
            self._proc.stdin.write('{"cmd": "dist"}\n')
            self._proc.stdin.flush()
            t0 = _t.time()
            while _t.time() - t0 < timeout:
                linea = self._proc.stdout.readline()
                if linea:
                    try:
                        return _j.loads(linea.strip()).get("dist")
                    except Exception:
                        return None
                _t.sleep(0.01)
        except Exception:
            self._ok = False
        return None

    def close(self):
        p = self._proc
        self._proc = None
        self._ok   = False
        if p is None:
            return

        try:
            if p.poll() is None:
                try:
                    p.stdin.write('{"cmd": "quit"}\n')
                    p.stdin.flush()
                except Exception:
                    pass
                try:
                    p.wait(timeout=3.0)
                except Exception:
                    pass
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
                try:
                    p.wait(timeout=2.0)
                except Exception:
                    pass
        finally:
            for s in (getattr(p, "stdin", None), getattr(p, "stdout", None)):
                try:
                    if s:
                        s.close()
                except Exception:
                    pass

def fase_entrega_sopa(robot):
    print("\n[SOPA][10] == Fase de entrega adaptativa ==", flush=True)

    print("[SOPA][10] Solicitando cam2 a la interfaz...", flush=True)
    escribir_estado_campo("cam2_interfaz_lista", False)
    escribir_estado_campo("cam2_owner", "orquestador")

    t_cam2 = time.time()
    while time.time() - t_cam2 < 2.5:
        try:
            est = leer_estado()
            if est.get("cam2_owner") == "orquestador" and                not est.get("cam2_interfaz_lista", True):
                break
        except Exception:
            pass
        time.sleep(0.05)
    time.sleep(0.15)
    print("[SOPA][10] cam2 confirmada libre - abriendo worker MediaPipe.", flush=True)

    def _devolver_cam2():
        time.sleep(1.2)
        escribir_estado_campo("cam2_owner", "interface")
        print("[SOPA][10] cam2 devuelta a la interfaz.", flush=True)

        t_dev = time.time()
        while time.time() - t_dev < 4.0:
            try:
                if leer_estado().get("cam2_interfaz_lista", False):
                    print("[SOPA][10] cam2 lista en interfaz.", flush=True)
                    break
            except Exception:
                pass
            time.sleep(0.05)

    proxy = _MediaPipeProxy(CAMARA_MEDIAPIPE_INDEX)

    if not proxy._ok:
        print("[SOPA][10] Worker MediaPipe no disponible - entrega sin deteccion.", flush=True)
        proxy.close()
        _devolver_cam2()

        time.sleep(EATING_WAIT_SOPA + 2.0)
        return True

    def leer_dist():
        return proxy.get_dist()

    print(f"  [F0] Busqueda de cara (max {CODO_BUSQUEDA_MAX_SOPA} subidas)", flush=True)
    for intento in range(CODO_BUSQUEDA_MAX_SOPA + 1):
        if debe_detener():
            proxy.close(); _devolver_cam2(); return False
        if leer_dist() is not None:
            print(f"  [OK] Cara en intento {intento}", flush=True)
            break
        if intento < CODO_BUSQUEDA_MAX_SOPA:
            print(f"  Sin cara (intento {intento+1}/{CODO_BUSQUEDA_MAX_SOPA}) "
                  f"-> CODO +{CODO_BUSQUEDA_PASOS_SOPA}, GRIPPER {GRIPPER_BUSQUEDA_SOPA} ...", flush=True)
            _mover_clamped(robot, "codo", CODO_BUSQUEDA_PASOS_SOPA,
                           limite_codo=LIMITE_CODO_BOCA)
            robot.mover_eje("muneca", GRIPPER_BUSQUEDA_SOPA)
            time.sleep(0.3)

    print(f"  [FA] Posicionamiento: HOMBRO {HOMBRO_POS_SOPA}, GRIPPER {GRIPPER_POS_SOPA} (una sola vez)", flush=True)
    if debe_detener():
        proxy.close(); _devolver_cam2(); return False
    robot.mover_eje("hombro", HOMBRO_POS_SOPA)
    robot.mover_eje("muneca", GRIPPER_POS_SOPA)

    escribir_voz("Listo para comer")

    print(f"  [FB] Esperando boca < {EATING_CM_SOPA} cm", flush=True)
    t0_b     = time.time()
    comiendo = False
    t_sin_cara = None

    while time.time() - t0_b < TIMEOUT_ENTREGA_SOPA:
        if debe_detener():
            proxy.close(); _devolver_cam2(); return False

        if not proxy._ok:
            print("  [AVISO] Worker MediaPipe caido - continuando sin deteccion.", flush=True)
            break
        dist = leer_dist()
        if dist is None:
            if t_sin_cara is None:
                t_sin_cara = time.time()
            elif time.time() - t_sin_cara >= 1.5:
                print("  [OK] Cara ausente 1.5 s -> boca asumida cerca", flush=True)
                comiendo = True
                break
        else:
            t_sin_cara = None
            if dist < EATING_CM_SOPA:
                print(f"  [OK] dist={dist:.1f} cm -> comiendo", flush=True)
                comiendo = True
                break
        time.sleep(0.05)

    if not comiendo:
        print("  [TIMEOUT] Timeout FB - continuando.", flush=True)

    print(f"  [FC] Delay {EATING_WAIT_SOPA}s + alejamiento", flush=True)
    t_comiendo     = time.time()
    t_cara_ausente = None

    while True:
        if debe_detener():
            proxy.close(); _devolver_cam2(); return False
        elapsed = time.time() - t_comiendo
        if elapsed < EATING_WAIT_SOPA:
            time.sleep(0.1)
            continue

        if not proxy._ok:
            print("  [AVISO] Worker caido en FC - saliendo.", flush=True)
            break
        dist = leer_dist()
        if dist is None:
            if t_cara_ausente is None:
                t_cara_ausente = time.time()
            elif time.time() - t_cara_ausente >= 3.0:
                print("  [OK] Cara ausente > 3 s - cucharada completa.", flush=True)
                break
        else:
            t_cara_ausente = None
            if dist > FAR_CM_SOPA:
                print(f"  [OK] Persona alejada {dist:.1f} cm.", flush=True)
                break
        if time.time() - t_comiendo > TIMEOUT_ENTREGA_SOPA:
            print("  [TIMEOUT] Timeout entrega.", flush=True)
            break
        time.sleep(0.1)

    proxy.close()
    _devolver_cam2()
    print(f"[SOPA][10] Entrega OK. _pos={robot._pos}", flush=True)
    return True

def _pedir_cam2_a_interfaz():
    print("[ENTREGA] Solicitando cam2 a la interfaz...", flush=True)
    escribir_estado_campo("cam2_interfaz_lista", False)
    escribir_estado_campo("cam2_owner", "orquestador")
    t0 = time.time()
    while time.time() - t0 < 3.0:
        est = leer_estado()
        if est.get("cam2_owner") == "orquestador" and not est.get("cam2_interfaz_lista", True):
            break
        time.sleep(0.05)
    time.sleep(0.5)
    print("[ENTREGA] cam2 confirmada libre.", flush=True)

def _devolver_cam2_a_interfaz():
    time.sleep(1.2)
    escribir_estado_campo("cam2_owner", "interface")
    print("[ENTREGA] cam2 devuelta a la interfaz.", flush=True)
    t0 = time.time()
    while time.time() - t0 < 4.0:
        if leer_estado().get("cam2_interfaz_lista", False):
            break
        time.sleep(0.05)

def fase_entrega_solido(robot, simulate, threshold_cm, eating_cm, far_cm):
    print("\n[ENTREGA][SOLIDO] == Fase de entrega adaptativa ==", flush=True)

    if simulate:
        print("  [SIM] Entrega simulada.", flush=True)
        time.sleep(3.0)
        return True

    _pedir_cam2_a_interfaz()

    proxy = _MediaPipeProxy(CAMARA_MEDIAPIPE_INDEX)
    if not proxy._ok:
        print("[ENTREGA][SOLIDO] Worker MediaPipe no disponible - entrega ciega.",
              flush=True)
        proxy.close()
        _devolver_cam2_a_interfaz()
        escribir_voz("Listo para comer")
        time.sleep(EATING_WAIT_SOL + 2.0)
        return True

    def leer_dist():
        return proxy.get_dist()

    proxy.set_fase("buscando-rostro")
    print(f"  [F0] Busqueda de rostro (max {CODO_BUSQUEDA_MAX_SOL} subidas).", flush=True)
    for intento in range(CODO_BUSQUEDA_MAX_SOL + 1):
        if debe_detener():
            proxy.close(); _devolver_cam2_a_interfaz(); return False
        d = None
        for _ in range(6):
            d = leer_dist()
            if d is not None:
                break
            time.sleep(0.08)
        if d is not None:
            print(f"  [OK] Rostro detectado a {d:.1f} cm (intento {intento}).", flush=True)
            break
        if intento < CODO_BUSQUEDA_MAX_SOL:
            print(f"  Sin rostro ({intento+1}/{CODO_BUSQUEDA_MAX_SOL}) - CODO +{CODO_BUSQUEDA_PASOS_SOL}.",
                  flush=True)
            robot.mover_eje("codo", CODO_BUSQUEDA_PASOS_SOL)
            time.sleep(0.4)

    proxy.set_fase("aproximando")
    print(f"  [FA] Aproximacion (parar si dist <= {threshold_cm} cm).", flush=True)
    approach_idx    = 0
    last_voz        = 0.0
    alarm_triggered = False
    while approach_idx < MAX_APPROACH_MOVES_SOL:
        if debe_detener():
            proxy.close(); _devolver_cam2_a_interfaz(); return False
        dist = leer_dist()
        if dist is None:
            print("  Sin rostro - brazo quieto, esperando cara...", flush=True)
            time.sleep(0.15)
            continue
        if dist <= threshold_cm:
            alarm_triggered = True
            if time.time() - last_voz > 1.5:
                escribir_voz("Listo para comer")
                last_voz = time.time()
                print(f"  [ALERTA] dist={dist:.1f}cm <= {threshold_cm}cm - BRAZO DETENIDO.",
                      flush=True)
            print(f"  Subiendo CODO {ALARM_RAISE_SOL} pasos para alcanzar la boca...",
                  flush=True)
            robot.mover_eje("codo", ALARM_RAISE_SOL)
            print("  Elevacion completada. Brazo en posicion de entrega.", flush=True)
            break
        eje, pasos = APPROACH_SEQ_SOL[approach_idx % len(APPROACH_SEQ_SOL)]
        print(f"  dist={dist:.1f}cm - moviendo {eje} {pasos:+d} (paso {approach_idx+1}).",
              flush=True)
        robot.mover_eje(eje, pasos)
        approach_idx += 1
        time.sleep(0.05)

    if not alarm_triggered:
        escribir_voz("Listo para comer")

    proxy.set_fase("alarma")
    print(f"  [FB] Esperando boca < {eating_cm} cm.", flush=True)
    t0_b       = time.time()
    t_sin_cara = None
    while time.time() - t0_b < TIMEOUT_ENTREGA_SOL:
        if debe_detener():
            proxy.close(); _devolver_cam2_a_interfaz(); return False
        if not proxy._ok:
            print("  [AVISO] Worker MediaPipe caido - continuando sin deteccion.", flush=True)
            break
        dist = leer_dist()
        if dist is None:
            if t_sin_cara is None:
                t_sin_cara = time.time()
            elif time.time() - t_sin_cara >= 1.5:
                print("  [OK] Cara ausente 1.5 s -> boca asumida cerca.", flush=True)
                break
        else:
            t_sin_cara = None
            if dist < eating_cm:
                print(f"  [OK] dist={dist:.1f} cm -> comiendo.", flush=True)
                break
        time.sleep(0.05)

    proxy.set_fase("comiendo")
    print(f"  [FC] Espera {EATING_WAIT_SOL}s + alejamiento.", flush=True)
    t_comiendo     = time.time()
    t_cara_ausente = None
    while True:
        if debe_detener():
            break
        elapsed = time.time() - t_comiendo
        if elapsed < EATING_WAIT_SOL:
            time.sleep(0.1)
            continue
        if not proxy._ok:
            print("  [AVISO] Worker caido en FC - saliendo.", flush=True)
            break
        dist = leer_dist()
        if dist is None:
            if t_cara_ausente is None:
                t_cara_ausente = time.time()
            elif time.time() - t_cara_ausente >= BOCA_AUSENTE_FIN_SOL:
                print("  [OK] Cara ausente -> entrega completa.", flush=True)
                break
        else:
            t_cara_ausente = None
            if dist > far_cm:
                print(f"  [OK] Persona alejada {dist:.1f} cm.", flush=True)
                break
        if time.time() - t_comiendo > TIMEOUT_ENTREGA_SOL:
            print("  [TIMEOUT] Timeout entrega.", flush=True)
            break
        time.sleep(0.1)

    escribir_voz("Parando alimentacion")
    proxy.close()
    _devolver_cam2_a_interfaz()
    print(f"[ENTREGA][SOLIDO] Entrega OK. _pos={robot._pos}", flush=True)
    return True

def ejecutar_cucharada(robot, sopa_detector, camara: CamaraCompartida, num):
    print(f"\n{'='*55}", flush=True)
    print(f"  Cucharada {num}", flush=True)
    print(f"{'='*55}", flush=True)

    print("\n[SOPA][6] HOME2...", flush=True)
    _ir_a_home2(robot)
    time.sleep(1.0)

    if not camara.abrir():
        print("[SOPA][6] ERROR cámara sopa.", flush=True)
        return False
    if sopa_detector is not None:
        nivel = sopa_detector.medir_nivel()
        print(f"[SOPA] Nivel: {nivel:.1f}%", flush=True)
        camara.cerrar()
        if nivel < UMBRAL_VACIO_PORC:
            escribir_voz("La sopa está casi vacía, terminando servicio")
            return False
    else:
        camara.cerrar()

    if debe_detener():
        return False

    print("\n[SOPA][7] Dentro del plato...", flush=True)
    robot.mover_eje("base",   PLATO_BASE)
    robot.mover_eje("hombro", PLATO_HOMBRO)
    robot.mover_eje("codo",   PLATO_CODO)
    robot.mover_eje("muneca", PLATO_MUNECA)

    if debe_detener():
        return False

    print("\n[SOPA][8] Sumergiendo...", flush=True)
    robot.mover_eje("hombro", SUMERGE_HOMBRO)

    if debe_detener():
        return False

    print("\n[SOPA][9] Levantando y centrando...", flush=True)
    robot.mover_eje("codo",   LEVANTA_CODO)
    robot.mover_eje("hombro", LEVANTA_HOMBRO)
    robot.mover_eje("base",   LEVANTA_BASE)

    if debe_detener():
        return False

    print("\n[SOPA][10] Iniciando entrega...", flush=True)
    if not fase_entrega_sopa(robot):
        return False

    return True

def bucle_sopa(robot, simulate, PINZA_MIN, PINZA_MAX):
    print("\n" + "="*60, flush=True)
    print("  SERVICIO DE SOPA - Múltiples cucharadas", flush=True)
    print("="*60, flush=True)

    if simulate:
        print("[SOPA] Modo SIMULADO.", flush=True)
        time.sleep(3)
        robot.go_home()
        return

    _esperar_cam1_disponible(timeout=10.0)

    camara = CamaraCompartida(index=CAMARA_SOPA_INDEX)

    sopa_detector = _cargar_sopa_detector(camara)
    if sopa_detector is None:
        escribir_voz("Error de calibración de sopa")
        robot.go_home()
        return

    cucharadas = 0
    while cucharadas < MAX_CUCHARADAS:
        if debe_detener():
            escribir_voz("Parando servicio de sopa")
            break

        esperar_si_pausa()
        cucharadas += 1
        print(f"\n[CICLO] Cucharada {cucharadas}/{MAX_CUCHARADAS}", flush=True)

        if not ejecutar_cucharada(robot, sopa_detector, camara, cucharadas):
            break

        time.sleep(1.0)

        if cucharadas % 3 == 0:
            if camara.abrir():
                nivel = sopa_detector.medir_nivel()
                camara.cerrar()
                print(f"[SOPA] Control periódico: {nivel:.1f}%", flush=True)
                if nivel < UMBRAL_VACIO_PORC:
                    escribir_voz("La sopa está casi vacía")
                    break

        if cucharadas >= 20:
            escribir_voz("¿Quieres seguir comiendo sopa?")
            for _ in range(30):
                if debe_detener():
                    break
                time.sleep(0.5)
            if debe_detener():
                escribir_voz("Terminando servicio")
                break

    escribir_voz("Terminando servicio de sopa")
    print("[SOPA] Fin - volviendo a HOME...", flush=True)
    robot.go_home()
    time.sleep(0.5)
    robot.set_gripper(90, "retire cuchara")
    time.sleep(0.5)
    escribir_voz("Retire la cuchara")
    print("[SOPA] >>> RETIRE LA CUCHARA <<<", flush=True)
    print("[SOPA] Esperando 20 s para retirar cuchara (sin voz ni YOLO)...", flush=True)
    for _ in range(20):
        if debe_detener():
            break
        time.sleep(1.0)
    robot.set_gripper(PINZA_MIN, "pinza cerrada tras sopa")
    time.sleep(0.3)
    camara.cerrar()
    print(f"[OK] Sopa completada. Cucharadas: {cucharadas}", flush=True)

def esperar_objetivo(timeout=120):
    print("[ORQ] Esperando objetivo (CAM1)...", flush=True)
    inicio = time.time()
    while True:
        if debe_detener():
            return "DETENER"
        esperar_si_pausa()
        estado = leer_estado()
        obj    = estado.get("objetivo")
        if obj and obj.get("listo") and obj.get("cx_norm") is not None:
            print(f"[ORQ] Objetivo recibido: cx={obj['cx_norm']:.4f} "
                  f"cy={obj['cy_norm']:.4f}", flush=True)
            return obj
        if time.time() - inicio > timeout:
            print(f"[ORQ] Timeout esperando objetivo.", flush=True)
            return None
        time.sleep(0.4)

def esperar_modo(robot, PINZA_MIN, PINZA_MAX):
    escribir_estado_campo("yolo_activo", False)
    escribir_estado_campo("modo_liquido_activo", False)
    escribir_fase("REPOSO")
    print("[REPOSO] HOME + pinza abierta. Esperando QUIERO COMER / QUIERO SOPA...",
          flush=True)
    robot.go_home()
    robot.set_gripper(PINZA_MAX, "reposo pinza abierta")
    escribir_voz("Sistema listo. Diga quiero comer o quiero sopa.")
    cmd0 = leer_estado().get("comando", "")
    if cmd0 in ("INICIAR_CICLO", "INICIAR_CICLO_LIQUIDO", "PARAR"):
        escribir_estado_campo("comando", "NINGUNO")
    while True:
        if debe_apagar():
            return "DETENER"
        cmd = leer_estado().get("comando", "")
        if cmd == "INICIAR_CICLO":
            print("[REPOSO] -> MODO SOLIDO.", flush=True)
            escribir_estado_campo("comando", "NINGUNO")
            return "SOLIDO"
        if cmd == "INICIAR_CICLO_LIQUIDO":
            print("[REPOSO] -> MODO LIQUIDO.", flush=True)
            escribir_estado_campo("comando", "NINGUNO")
            return "LIQUIDO"
        time.sleep(0.2)

def agarrar_cuchara_y_verificar(robot, simulate):
    print("[LIQUIDO] Tomando la cuchara...", flush=True)
    escribir_fase("AGARRE_CUCHARA")
    _ir_a_verificacion_aruco(robot)
    print("[LIQUIDO] Abriendo pinza...", flush=True)
    robot.set_gripper(90, "abrir pinza cuchara")
    time.sleep(0.5)
    print("[LIQUIDO] Bajando HOMBRO a la cuchara...", flush=True)
    robot.mover_eje("hombro", BAJAR_HOMBRO)
    print("[LIQUIDO] Cerrando pinza...", flush=True)
    robot.set_gripper(0, "cerrar pinza cuchara")
    time.sleep(0.6)
    robot.mover_eje("hombro", SUBIR_HOMBRO)
    robot.mover_eje("base",   SUBIR_BASE)
    escribir_estado_campo("cam1_owner",          "cuchara")
    escribir_estado_campo("modo_liquido_activo", True)
    time.sleep(1.5)
    if simulate:
        return True
    _esperar_cam1_disponible(timeout=10.0)
    camara_verif = CamaraCompartida(index=CAMARA_SOPA_INDEX)
    if not camara_verif.abrir():
        print("[LIQUIDO] ERROR: no se pudo abrir CAM1 para verificar agarre.",
              flush=True)
        return False
    agarre_ok = verificar_agarre_cuchara_post_subida(camara_verif)
    camara_verif.cerrar()
    return agarre_ok

def ejecutar_modo_liquido(robot, simulate, args, PINZA_MIN, PINZA_MAX):
    escribir_voz("Modo sopa.")
    ok = agarrar_cuchara_y_verificar(robot, simulate)
    if not ok:
        print("[LIQUIDO] Agarre de cuchara fallido.", flush=True)
        escribir_voz("No se pudo tomar la cuchara")
        robot.set_gripper(90, "soltar cuchara")
        time.sleep(0.5)
        robot.go_home()
        escribir_estado_campo("cam1_owner",          "interface")
        escribir_estado_campo("modo_liquido_activo", False)
        return
    print("[LIQUIDO] Agarre verificado. Sirviendo sopa.", flush=True)
    escribir_fase("AGARRE_CUCHARA_OK")
    bucle_sopa(robot, simulate, PINZA_MIN, PINZA_MAX)
    escribir_estado_campo("cam1_owner",          "interface")
    escribir_estado_campo("modo_liquido_activo", False)

def ejecutar_modo_solido(robot, predictor, cuadricula, yolo_model,
                         simulate, args, PINZA_MIN, PINZA_MAX):
    escribir_estado_campo("yolo_activo", True)
    escribir_voz("Modo solido. Buscando comida.")
    ciclo      = 0
    agarres_ok = 0
    while not debe_detener():
        esperar_si_pausa()
        ciclo += 1
        print(f"\n{'='*60}\n  CICLO SOLIDO {ciclo}\n{'='*60}", flush=True)
        escribir_fase("HOME", ciclo=ciclo, agarres_ok=agarres_ok)
        robot.go_home()
        robot.set_gripper(PINZA_MIN, "inicio ciclo")
        time.sleep(0.4)

        escribir_fase("DETECCION", ciclo=ciclo)
        objetivo = esperar_objetivo(timeout=30)
        if objetivo == "DETENER" or debe_detener():
            break
        if objetivo is None:
            print("[SOLIDO] No hay mas comida en el plato -> volviendo a reposo.",
                  flush=True)
            escribir_voz("No hay mas comida. Volviendo a reposo.")
            break

        obj_cx = objetivo["cx_norm"]
        obj_cy = objetivo["cy_norm"]
        escribir_estado_campo("objetivo", {"listo": False})

        agarre_exitoso = False
        intento        = 0
        while not agarre_exitoso:
            if debe_detener():
                break
            intento += 1
            print(f"\n  INTENTO #{intento}", flush=True)
            escribir_fase("PREDICCION MLP", ciclo=ciclo)
            pred = predictor.predecir(obj_cx, obj_cy)
            print(f"[MLP] Prediccion: {pred}", flush=True)
            escribir_fase("AGARRE", ciclo=ciclo)
            escribir_voz("Iniciando agarre")
            robot.set_gripper(PINZA_MAX, "pre-agarre")
            time.sleep(0.5)
            robot.ejecutar_prediccion(pred)
            robot.set_gripper(PINZA_MIN, "agarrando")
            time.sleep(0.8)
            robot.go_home()
            time.sleep(0.3)
            escribir_fase("VERIFICACION", ciclo=ciclo)
            print("[SOLIDO] Verificando agarre...", flush=True)
            time.sleep(2.0)
            est       = leer_estado()
            obj_nuevo = est.get("objetivo")
            if obj_nuevo and obj_nuevo.get("listo"):
                dist_n = ((obj_nuevo.get("cx_norm", 0) - obj_cx) ** 2 +
                          (obj_nuevo.get("cy_norm", 0) - obj_cy) ** 2) ** 0.5
                if dist_n < 0.15:
                    print("[SOLIDO] Trozo presente -> agarre fallido, reintento.",
                          flush=True)
                    robot.set_gripper(PINZA_MAX, "soltando-reintento")
                    time.sleep(0.5)
                    robot.set_gripper(PINZA_MIN, "listo-reintento")
                    robot.go_home()
                    continue
            agarre_exitoso = True
            escribir_voz("Llevando alimento")
            print("[SOLIDO] Trozo no detectado -> agarre exitoso.", flush=True)

        if debe_detener():
            break

        agarres_ok += 1
        escribir_fase("ENTREGA", ciclo=ciclo, agarres_ok=agarres_ok)
        robot.go_home()
        fase_entrega_solido(
            robot,
            simulate=simulate,
            threshold_cm=args.threshold,
            eating_cm=args.eating,
            far_cm=args.far,
        )

        if debe_detener():
            break

        escribir_fase("CICLO COMPLETO", ciclo=ciclo, agarres_ok=agarres_ok)
        robot.go_home()
        robot.set_gripper(PINZA_MAX, "soltando")
        time.sleep(0.8)
        robot.set_gripper(PINZA_MIN, "listo-siguiente")
        time.sleep(0.4)
        print(f"\n  Ciclo solido {ciclo} completado. Agarres: {agarres_ok}",
              flush=True)
        time.sleep(0.5)

    print(f"[SOLIDO] Modo solido finalizado. Agarres: {agarres_ok}", flush=True)

def bucle_principal(robot, predictor, cuadricula, yolo_model, simulate, args):
    from auto_brazo_completo_PRESENTACION import PINZA_MIN, PINZA_MAX

    while True:
        modo = esperar_modo(robot, PINZA_MIN, PINZA_MAX)
        if modo == "DETENER":
            print("[ORQ] DETENER - apagando sistema.", flush=True)
            break

        if modo == "SOLIDO":
            ejecutar_modo_solido(robot, predictor, cuadricula, yolo_model,
                                 simulate, args, PINZA_MIN, PINZA_MAX)
        elif modo == "LIQUIDO":
            ejecutar_modo_liquido(robot, simulate, args, PINZA_MIN, PINZA_MAX)

        if debe_apagar():
            print("[ORQ] DETENER tras modo - apagando.", flush=True)
            break

        escribir_estado_campo("comando",             "NINGUNO")
        escribir_estado_campo("cam1_owner",          "interface")
        escribir_estado_campo("modo_liquido_activo", False)
        print("[ORQ] PARAR - volviendo a reposo (esperando modo).", flush=True)

    robot.go_home()
    robot.set_gripper(PINZA_MIN, "apagado")
    escribir_estado_campo("comando",     "NINGUNO")
    escribir_estado_campo("cam1_owner",  "interface")
    escribir_fase("INACTIVO")

def main():
    parser = argparse.ArgumentParser(
        description="Orquestador UNIFICADO del brazo robótico"
    )
    parser.add_argument("--modo",      default="trozo",
                        choices=["trozo", "cuchara"])
    parser.add_argument("--sim",       action="store_true")
    parser.add_argument("--cam1",      type=int,   default=3)
    parser.add_argument("--cam2",      type=int,   default=1)
    parser.add_argument("--puerto",    default="COM3")
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--eating",    type=float, default=15.0)
    parser.add_argument("--far",       type=float, default=20.0)
    parser.add_argument("--modelo",    default="")
    parser.add_argument("--calib",     default="")
    args = parser.parse_args()

    simulate = args.sim

    print(f"[ORQ] Orquestador UNIFICADO.", flush=True)
    print(f"[ORQ] Modo: {args.modo.upper()}", flush=True)
    print(f"[ORQ] Simulación: {simulate}", flush=True)
    print(f"[ORQ] Puerto: {args.puerto}", flush=True)

    modulo = importar_modulo_principal()
    if modulo is None and not simulate:
        print("[ORQ] ERROR: módulo principal no encontrado.", flush=True)
        sys.exit(1)

    if modulo:
        modulo.SERIAL_PORT    = args.puerto
        modulo.CAMERA_1_INDEX = args.cam1
        modulo.CAMERA_2_INDEX = args.cam2
        if args.modelo:
            modulo.MODELO_PT  = args.modelo
        if args.calib:
            modulo.CALIB_JSON = args.calib

    print("[ORQ] Inicializando componentes...", flush=True)
    escribir_fase("INICIANDO")

    if modulo:
        from auto_brazo_completo_PRESENTACION import (
            MLPPredictor, Cuadricula, RobotInterface, PINZA_MIN, PINZA_MAX
        )
        try:
            from ultralytics import YOLO
            yolo_model = YOLO("yolov8x-worldv2.pt")
            yolo_model.set_classes(modulo.YOLO_CLASSES)
            print("[YOLO] Modelo cargado.", flush=True)
        except Exception as e:
            print(f"[YOLO] Error: {e}", flush=True)
            yolo_model = None
            if not simulate:
                sys.exit(1)

        try:
            predictor  = MLPPredictor(args.modelo, simulate=simulate)
            cuadricula = Cuadricula(args.calib)
        except Exception as e:
            print(f"[ORQ] Error MLP/Cuadricula: {e}", flush=True)
            if not simulate:
                sys.exit(1)
            predictor  = MLPPredictor("", simulate=True)
            cuadricula = type("C", (), {
                "punto_dentro": lambda *_: True,
                "info_celda":   lambda *_: {"celda":1,"fila":1,"columna":1},
                "disponible":   False,
            })()
    else:

        from auto_brazo_completo_PRESENTACION import PINZA_MIN, PINZA_MAX

        _pos0 = {"base":0,"hombro":400,"codo":400,"muneca":0,"rotacion":0}

        class _SimRobot:
            simulate = True
            _pos     = dict(_pos0)
            _pinza   = 0
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def go_home(self):
                print("  [SIM] HOME", flush=True)
                self._pos = dict(_pos0)
            def set_gripper(self, a, l=""):
                print(f"  [SIM] PINZA {a}", flush=True)
            def mover_eje(self, e, p):
                print(f"  [SIM] {e} {p:+d}", flush=True)
                self._pos[e] = self._pos.get(e, 0) + p
            def ejecutar_prediccion(self, p):
                print(f"  [SIM] Pred:{p}", flush=True)

        RobotInterface = _SimRobot
        predictor  = type("P", (), {
            "predecir": lambda self,x,y: {"base":100,"codo":200,"hombro":150}
        })()
        cuadricula = type("C", (), {
            "punto_dentro": lambda *_: True,
            "info_celda":   lambda *_: {"celda":1,"fila":1,"columna":1},
            "disponible":   False,
        })()
        yolo_model = None

    print(f"[ORQ] Lanzando bucle principal (arranque en REPOSO)...", flush=True)

    with RobotInterface(simulate=simulate) as robot:

        escribir_estado_campo("comando", "NINGUNO")
        bucle_principal(robot, predictor, cuadricula, yolo_model, simulate, args)

    escribir_fase("INACTIVO")
    escribir_estado_campo("modo_liquido_activo", False)
    print("[ORQ] Programa terminado.", flush=True)

def _sigint(sig, frame):
    print("\n[ORQ] Ctrl+C.", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint)

if __name__ == "__main__":
    main()