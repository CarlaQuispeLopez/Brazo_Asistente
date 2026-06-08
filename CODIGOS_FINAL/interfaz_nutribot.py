import tkinter as tk
import threading
import subprocess
import json
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import speech_recognition as sr
    VOICE_OK = True
except ImportError:
    VOICE_OK = False
    print("[VOZ] speech_recognition no instalado. pip install SpeechRecognition pyaudio")

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False
    print("[TTS] pyttsx3 no instalado. pip install pyttsx3")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[YOLO] ultralytics no instalado. pip install ultralytics")

DIR_BASE    = os.path.dirname(os.path.abspath(__file__))
ESTADO_JSON = os.path.join(DIR_BASE, "estado.json")
ORQUESTADOR = os.path.join(DIR_BASE, "orquestador.py")

CAM2_FRAME_TMP = os.path.join(DIR_BASE, "nutribot_cam2_frame.npy")
CAM2_FRAME_TMP_WRITE = os.path.join(DIR_BASE, "nutribot_cam2_frame.tmp.npy")

MODELO_PT  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_FINAL\modelo_bc.pt"
CALIB_JSON = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_FINAL\calibracion_cuadricula.json"

CAM1_INDEX  = 2
CAM2_INDEX  = 1
SERIAL_PORT = "COM3"

MIC_INDEX = None

CAM_ANCHO  = 410
CAM_ALTO   = 230

YOLO_CONF        = 0.20
INTERVALO_DETECT = 1.5

YOLO_CLASSES = [
    "apple", "pear", "peach", "banana", "grape", "orange slice",
    "mango", "tomato", "carrot piece", "broccoli floret",
    "cucumber slice", "potato chunk", "chicken piece", "beef piece",
    "pork piece", "meatball", "shrimp", "fish piece",
    "boiled egg", "tofu cube", "pasta piece", "noodle",
    "rice ball", "bread piece", "food piece", "fruit piece",
    "vegetable piece", "meat piece",
]

BG_VENTANA      = "#F2F2F2"
COLOR_TITULO    = "#1A1A1A"
COLOR_TEXTO     = "#1A1A1A"
COLOR_SEPARADOR = "#D0D0D0"
COLOR_BARRA     = "#E6E6E6"
COLOR_BARRA_TXT = "#666666"
COLOR_CAM       = "#000000"

VACIO_OFF    = "#F0C8C0"
VACIO_ON     = "#E07060"
SOLIDO_OFF   = "#EDE0B0"
SOLIDO_ON    = "#C8A000"
LIQUIDO_OFF  = "#C0D4EC"
LIQUIDO_ON   = "#5090D0"

ALRM_A_OFF   = "#C4E0C4"
ALRM_A_ON    = "#40B840"
ALRM_B_OFF   = "#EDE0B0"
ALRM_B_ON    = "#C8A000"
ALRM_C_OFF   = "#C8DFF0"
ALRM_C_ON    = "#2E86C1"
ALRM_D_OFF   = "#F0D0C4"
ALRM_D_ON    = "#D06040"

BTN_VERDE    = "#90CC78"
BTN_VERDE_H  = "#70AA58"
BTN_ROJO     = "#D86060"
BTN_ROJO_H   = "#B84040"
BTN_TEXTO    = "#FFFFFF"
IND_TEXTO_OFF = "#888888"
IND_TEXTO_ON  = "#FFFFFF"

import queue as _queue

_tts_cola = _queue.Queue()

def _hilo_tts_worker():
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")

        voices = speaker.GetVoices()
        for i in range(voices.Count):
            desc = voices.Item(i).GetDescription().lower()
            if any(k in desc for k in ("spanish", "español", "helena", "sabina", "pablo")):
                speaker.Voice = voices.Item(i)
                break
        speaker.Rate   = 0
        speaker.Volume = 100
        print("[TTS] Motor SAPI inicializado correctamente.", flush=True)
    except Exception as e:
        print(f"[TTS] Error al inicializar SAPI: {e}", flush=True)
        return

    while True:
        texto = _tts_cola.get()
        if texto is None:
            break
        try:
            print(f"[TTS-WORKER] Pronunciando: '{texto}'", flush=True)
            speaker.Speak(texto)
            print(f"[TTS-WORKER] Fin: '{texto}'", flush=True)
        except Exception as e:
            print(f"[TTS] Error al hablar '{texto}': {e}", flush=True)

def inicializar_tts():
    t = threading.Thread(target=_hilo_tts_worker, daemon=True)
    t.name = "TTS-Worker"
    t.start()

def hablar(texto):
    print(f"[TTS] hablar('{texto}') — cola size antes: {_tts_cola.qsize()}", flush=True)
    _tts_cola.put(texto)

def _en_hilo(func):
    threading.Thread(target=func, daemon=True).start()

def alarma_iniciando_agarre():
    hablar("Iniciando agarre")

def alarma_llevando_alimento():
    hablar("Llevando alimento")

def alarma_listo_para_comer():
    hablar("Listo para comer")

def alarma_parando_agarre():
    hablar("Parando alimentación")

ESTADO_INICIAL = {
    "comando":     "NINGUNO",
    "modo":        "trozo",
    "simulacion":  False,
    "fase_actual": "INACTIVO",
    "ciclo":       0,
    "agarres_ok":  0,
    "cam2_owner":           "interface",
    "cam2_interfaz_lista":  True,
    "yolo_activo":          True,
    "cam1_owner":  "interface",
    "config": {
        "puerto":       SERIAL_PORT,
        "baud":         115200,
        "cam1":         CAM1_INDEX,
        "cam2":         CAM2_INDEX,
        "threshold_cm": 20.0,
        "eating_cm":    15.0,
        "far_cm":       20.0,
        "modelo_pt":    MODELO_PT,
        "calib_json":   CALIB_JSON,
    }
}

def leer_estado():
    for intento in range(3):
        try:
            with open(ESTADO_JSON, "r", encoding="utf-8") as f:
                contenido = f.read()
            if contenido.strip():
                return json.loads(contenido)
        except (json.JSONDecodeError, ValueError):

            if intento < 2:
                import time as _t; _t.sleep(0.005)
        except Exception:
            break
    return {}

def escribir_estado_campo(campo, valor):
    try:
        estado = leer_estado()
        if not estado:
            estado = dict(ESTADO_INICIAL)
        estado[campo] = valor
        tmp = ESTADO_JSON + ".tmp.gui"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=2, ensure_ascii=False)
        os.replace(tmp, ESTADO_JSON)
    except Exception as e:
        print(f"[GUI] Error estado.json: {e}", flush=True)

def enviar_comando(cmd):
    escribir_estado_campo("comando", cmd)

class HiloVoz(threading.Thread):

    FRASE_INICIAR = ["quiero comer", "comer", "solido", "trozo"]
    FRASE_SOPA    = ["sopa", "liquido", "caldo"]
    FRASE_DETENER = ["ya no quiero", "no quiero comer", "ya termine", "parar"]

    def __init__(self, cb_iniciar, cb_detener, cb_estado, cb_sopa=None):
        super().__init__(daemon=True)
        self.cb_iniciar = cb_iniciar
        self.cb_detener = cb_detener
        self.cb_estado  = cb_estado
        self.cb_sopa    = cb_sopa
        self.activo     = True

    def run(self):
        if not VOICE_OK:
            self.cb_estado("[VOZ] No disponible. pip install SpeechRecognition pyaudio")
            return

        import unicodedata

        def _norm(s):
            s = unicodedata.normalize("NFKD", s)
            s = s.encode("ascii", "ignore").decode("ascii")
            return s.lower().strip()

        try:
            nombres = sr.Microphone.list_microphone_names()
            print("[VOZ] Microfonos disponibles:", flush=True)
            for i, n in enumerate(nombres):
                print(f"   [{i}] {n}", flush=True)
        except Exception as e:
            print(f"[VOZ] No se pudo listar microfonos: {e}", flush=True)

        rec = sr.Recognizer()
        rec.pause_threshold          = 0.8
        rec.energy_threshold         = 300
        rec.dynamic_energy_threshold = True

        det = [_norm(f) for f in self.FRASE_DETENER]
        sop = [_norm(f) for f in self.FRASE_SOPA]
        ini = [_norm(f) for f in self.FRASE_INICIAR]

        while self.activo:
            try:
                mic = (sr.Microphone(device_index=MIC_INDEX)
                       if MIC_INDEX is not None else sr.Microphone())
            except Exception as e:
                self.cb_estado(f"[VOZ] Error al abrir microfono: {e}")
                time.sleep(3)
                continue

            try:

                with mic as fuente:
                    rec.adjust_for_ambient_noise(fuente, duration=1.0)
                    self.cb_estado("[VOZ] Escuchando: di 'Quiero comer' o 'Quiero sopa'")
                    print(f"[VOZ] Listo. Umbral energia = {rec.energy_threshold:.0f}",
                          flush=True)

                    while self.activo:
                        try:
                            audio = rec.listen(fuente, timeout=6, phrase_time_limit=5)
                        except sr.WaitTimeoutError:
                            continue

                        self.cb_estado("[VOZ] Procesando...")
                        try:
                            texto = rec.recognize_google(audio, language="es-ES")
                        except sr.UnknownValueError:
                            self.cb_estado("[VOZ] No se entendio, intenta de nuevo.")
                            continue
                        except sr.RequestError as e:
                            self.cb_estado(f"[VOZ] Sin conexion para reconocer: {e}")
                            time.sleep(2)
                            continue
                        except Exception as e:
                            print(f"[VOZ] Error reconocimiento: {e}", flush=True)
                            continue

                        n = _norm(texto)
                        print(f"[VOZ] Reconocido: '{texto}'", flush=True)
                        self.cb_estado(f"[VOZ] Reconocido: {texto}")

                        if any(f in n for f in det):
                            self.cb_detener()
                        elif self.cb_sopa and any(f in n for f in sop):
                            self.cb_sopa()
                        elif any(f in n for f in ini):
                            self.cb_iniciar()
            except OSError as e:
                print(f"[VOZ] Microfono perdido ({e}) - reintentando.", flush=True)
                self.cb_estado("[VOZ] Microfono perdido - reintentando...")
                time.sleep(2)
            except Exception as e:
                print(f"[VOZ] Error en bucle de escucha: {e}", flush=True)
                time.sleep(2)

    def detener(self):
        self.activo = False

class NutribotGUI:

    MODO_VACIO   = "VACIO"
    MODO_SOLIDO  = "SOLIDO"
    MODO_LIQUIDO = "LIQUIDO"

    ALARMA_NINGUNA   = ""
    ALARMA_INICIANDO = "INICIANDO"
    ALARMA_LLEVANDO  = "LLEVANDO"
    ALARMA_LISTO     = "LISTO"
    ALARMA_PARANDO   = "PARANDO"

    def __init__(self, root):
        self.root = root

        self.sistema_activo = False
        self.proc           = None
        self.modo_actual    = self.MODO_VACIO
        self.modo_liquido_activo = False
        self.alarma_actual  = self.ALARMA_NINGUNA
        self.ultimo_detect  = 0.0
        self._ultima_voz    = ""
        self._camaras_activas = False

        self.frame_cam1 = None
        self.frame_cam2 = None
        self._lock_cam1 = threading.Lock()
        self._lock_cam2 = threading.Lock()
        self._photo1    = None
        self._photo2    = None

        self.yolo_model = None
        self.hilo_voz   = None

        self._escribir_estado_inicial()
        inicializar_tts()
        self._construir_ventana()
        self._iniciar_camaras()
        self._cargar_yolo_en_hilo()
        self._iniciar_voz()
        self._ciclo_actualizacion()

    def _escribir_estado_inicial(self):
        try:
            tmp = ESTADO_JSON + ".tmp.gui"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(dict(ESTADO_INICIAL), f, indent=2, ensure_ascii=False)
            os.replace(tmp, ESTADO_JSON)
        except Exception:
            pass

        for ruta in (CAM2_FRAME_TMP, CAM2_FRAME_TMP_WRITE,
                     CAM2_FRAME_TMP.replace(".npy", ".jpg"),
                     CAM2_FRAME_TMP.replace(".npy", ".jpg") + ".tmp.jpg"):
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
            except Exception:
                pass

    def _construir_ventana(self):
        self.root.title("Sistema NUTRIBOT")
        self.root.configure(bg=BG_VENTANA)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._cerrar)
        ancho = CAM_ANCHO * 2 + 100
        self.root.geometry(f"{ancho}x570")

        self._construir_encabezado()
        self._construir_camaras()
        self._separador()
        self._construir_modos()
        self._construir_alarmas()
        self._construir_barra_estado()

    def _construir_encabezado(self):
        frame = tk.Frame(self.root, bg=BG_VENTANA)
        frame.pack(fill="x", padx=28, pady=(16, 6))

        tk.Label(
            frame,
            text="SISTEMA NUTRIBOT",
            font=("Arial", 14),
            bg=BG_VENTANA,
            fg=COLOR_TITULO,
        ).pack(side="left", expand=True)

        self.btn_sistema = tk.Button(
            frame,
            text="INICIAR",
            font=("Arial", 9, "bold"),
            bg=BTN_VERDE,
            fg=BTN_TEXTO,
            activebackground=BTN_VERDE_H,
            activeforeground=BTN_TEXTO,
            relief="flat",
            bd=0,
            padx=18, pady=5,
            cursor="hand2",
            command=self._toggle_sistema,
        )
        self.btn_sistema.pack(side="right")

    def _construir_camaras(self):
        frame_cams = tk.Frame(self.root, bg=BG_VENTANA)
        frame_cams.pack(padx=28, pady=(6, 0))

        for idx, attr, etiqueta in (
            (0, "lbl_cam1", "CAM 1"),
            (1, "lbl_cam2", "CAM 2"),
        ):
            col = tk.Frame(frame_cams, bg=BG_VENTANA)
            col.pack(side="left", padx=(0, 20) if idx == 0 else 0)

            marco = tk.Frame(col, bg=COLOR_CAM,
                             width=CAM_ANCHO, height=CAM_ALTO)
            marco.pack_propagate(False)
            marco.pack()

            lbl = tk.Label(marco, bg=COLOR_CAM)
            lbl.pack(fill="both", expand=True)
            setattr(self, attr, lbl)

            tk.Label(
                col, text=etiqueta,
                font=("Arial", 9),
                bg=BG_VENTANA, fg=COLOR_TEXTO,
            ).pack(pady=(5, 0))

    def _construir_modos(self):
        frame = tk.Frame(self.root, bg=BG_VENTANA)
        frame.pack(fill="x", padx=28, pady=(10, 6))

        tk.Label(
            frame,
            text="MODO:",
            font=("Arial", 10, "bold"),
            bg=BG_VENTANA,
            fg=COLOR_TEXTO,
            width=9, anchor="w",
        ).pack(side="left")

        self.ind_vacio   = self._indicador(frame, "VACIO",
                                            VACIO_OFF,   VACIO_ON)
        self.ind_solido  = self._indicador(frame, "SOLIDO",
                                            SOLIDO_OFF,  SOLIDO_ON)
        self.ind_liquido = self._indicador(frame, "LIQUIDO",
                                            LIQUIDO_OFF, LIQUIDO_ON)

    def _construir_alarmas(self):
        frame = tk.Frame(self.root, bg=BG_VENTANA)
        frame.pack(fill="x", padx=28, pady=(4, 8))

        tk.Label(
            frame,
            text="ALARMA:",
            font=("Arial", 10, "bold"),
            bg=BG_VENTANA,
            fg=COLOR_TEXTO,
            width=9, anchor="w",
        ).pack(side="left")

        self.ind_iniciando = self._indicador(
            frame, "INICIANDO\nAGARRE",  ALRM_A_OFF, ALRM_A_ON, alto=2)
        self.ind_llevando  = self._indicador(
            frame, "LLEVANDO\nALIMENTO", ALRM_B_OFF, ALRM_B_ON, alto=2)
        self.ind_listo     = self._indicador(
            frame, "LISTO PARA\nCOMER",  ALRM_C_OFF, ALRM_C_ON, alto=2)
        self.ind_parando   = self._indicador(
            frame, "PARANDO\nALIMENTACIÓN", ALRM_D_OFF, ALRM_D_ON, alto=2)

        frame_r = tk.Frame(self.root, bg=BG_VENTANA)
        frame_r.pack(fill="x", padx=28, pady=(2, 8))
        self.lbl_reposo = tk.Label(frame_r, text="REPOSO:",
            font=("Arial", 10, "bold"), bg=BG_VENTANA, fg="#888888",
            width=9, anchor="w")
        self.lbl_reposo.pack(side="left")
        self.btn_comer = tk.Button(frame_r, text="QUIERO COMER",
            font=("Arial", 9, "bold"), bg="#78C878", fg="white",
            activebackground="#58A858", activeforeground="white",
            relief="flat", bd=0, padx=14, pady=5, cursor="hand2",
            command=self._btn_quiero_comer, state="disabled")
        self.btn_comer.pack(side="left", padx=(0, 8))
        self.btn_sopa = tk.Button(frame_r, text="QUIERO COMER SOPA",
            font=("Arial", 9, "bold"), bg="#5090D0", fg="white",
            activebackground="#3070B0", activeforeground="white",
            relief="flat", bd=0, padx=14, pady=5, cursor="hand2",
            command=self._btn_quiero_sopa, state="disabled")
        self.btn_sopa.pack(side="left")

    def _construir_barra_estado(self):
        barra = tk.Frame(self.root, bg=COLOR_BARRA)
        barra.pack(fill="x", side="bottom")

        self.lbl_estado = tk.Label(
            barra,
            text="Sistema listo. Presione INICIAR o diga 'Quiero COMER'.",
            font=("Arial", 8),
            bg=COLOR_BARRA,
            fg=COLOR_BARRA_TXT,
            anchor="w",
        )
        self.lbl_estado.pack(side="left", padx=10, pady=4)

        self.lbl_ciclos = tk.Label(
            barra,
            text="Ciclos: 0  |  Entregas: 0",
            font=("Arial", 8),
            bg=COLOR_BARRA,
            fg=COLOR_BARRA_TXT,
        )
        self.lbl_ciclos.pack(side="right", padx=10)

    def _separador(self):
        tk.Frame(self.root, bg=COLOR_SEPARADOR, height=1).pack(
            fill="x", padx=20, pady=6)

    def _indicador(self, parent, texto, c_off, c_on, ancho=12, alto=1):
        lbl = tk.Label(
            parent,
            text=texto,
            font=("Arial", 8, "bold"),
            bg=c_off,
            fg=IND_TEXTO_OFF,
            relief="groove",
            bd=1,
            width=ancho,
            height=alto,
            padx=4, pady=5,
        )
        lbl.pack(side="left", padx=5)
        lbl._c_off = c_off
        lbl._c_on  = c_on
        return lbl

    def _set_ind(self, lbl, encendido):
        lbl.config(
            bg=lbl._c_on  if encendido else lbl._c_off,
            fg=IND_TEXTO_ON if encendido else IND_TEXTO_OFF,
        )

    def _iniciar_camaras(self):
        threading.Thread(
            target=self._hilo_cam,
            args=(CAM1_INDEX, self._lock_cam1, 1, 1280, 720),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._hilo_cam2,
            daemon=True,
        ).start()

    def _hilo_cam(self, indice, lock, num, ancho, alto):
        cap = None
        cam1_owner_cache = "interface"

        while True:

            if not self._camaras_activas:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                cam1_owner_cache = "interface"
                time.sleep(0.1)
                continue

            if num == 1:
                try:
                    estado = leer_estado()
                    if isinstance(estado, dict) and "cam1_owner" in estado:
                        cam1_owner_cache = estado["cam1_owner"]
                except Exception:
                    pass

                if cam1_owner_cache == "cuchara":

                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        print("[CAM1] Cedida al modulo cuchara.", flush=True)
                    time.sleep(0.1)
                    continue
                elif cam1_owner_cache == "interface" and cap is None:

                    print("[CAM1] Recuperada por interfaz — warmup.", flush=True)
                    time.sleep(0.5)

            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ancho)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    with lock:
                        if num == 1:
                            self.frame_cam1 = frame
            else:
                try:
                    cap.open(indice)
                except Exception:
                    pass
            time.sleep(0.030)

    def _frame_cam2_espera(self, texto="CAM 2 esperando señal"):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, texto, (55, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 180, 180), 2)
        cv2.putText(frame, "La imagen de MediaPipe aparecerá aquí", (80, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 120, 120), 1)
        return frame

    def _hilo_cam2(self):
        cap                = None
        ultimo_ts          = 0.0
        owner_prev         = "interface"
        owner_cache        = "interface"
        iter_count         = 0
        ultimo_frame_sum   = -1
        frames_congelados  = 0
        ultimo_intento     = 0.0
        cooldown           = 1.5
        fallos_abrir       = 0
        t_abierta          = 0.0

        def _ok(f):
            return (f is not None and f.ndim == 3 and f.shape[2] == 3
                    and f.shape[0] > 9 and f.shape[1] > 9
                    and f.dtype == np.uint8 and f.max() >= 3)

        def _abrir():
            nonlocal cap, ultimo_frame_sum, frames_congelados
            if cap is not None:
                try: cap.release()
                except Exception: pass
                cap = None
                time.sleep(0.8)
            print(f"[CAM2] Abriendo indice {CAM2_INDEX}...", flush=True)

            _c = None
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, None):
                try:
                    cand = (cv2.VideoCapture(CAM2_INDEX, backend)
                            if backend is not None else cv2.VideoCapture(CAM2_INDEX))
                except Exception:
                    cand = None
                if cand is not None and cand.isOpened():
                    _c = cand
                    break
                if cand is not None:
                    try: cand.release()
                    except Exception: pass
                time.sleep(0.25)
            if _c is None:
                print("[CAM2] No se pudo abrir (todos los backends).", flush=True)
                return None
            _c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            _c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            _c.set(cv2.CAP_PROP_FPS,          30)
            _c.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            ok = 0
            for _ in range(30):
                r, f = _c.read()
                if r and f is not None:
                    ok += 1
                if ok >= 10:
                    break
            print("[CAM2] Abierta.", flush=True)
            ultimo_frame_sum = -1; frames_congelados = 0
            return _c

        while True:
            iter_count += 1

            if not self._camaras_activas:
                if cap is not None:
                    try: cap.release()
                    except Exception: pass
                    cap = None
                with self._lock_cam2:
                    self.frame_cam2 = None
                owner_prev = "interface"; owner_cache = "interface"
                ultimo_frame_sum = -1; frames_congelados = 0; iter_count = 0
                fallos_abrir = 0; cooldown = 1.5
                time.sleep(0.1); continue

            if iter_count % 10 == 1:
                try:
                    est = leer_estado()
                    if isinstance(est, dict) and "cam2_owner" in est:
                        owner_cache = est["cam2_owner"]
                except Exception:
                    pass

            if owner_cache == "orquestador":
                if cap is not None:
                    try: cap.release()
                    except Exception: pass
                    cap = None; ultimo_ts = 0.0
                    try: escribir_estado_campo("cam2_interfaz_lista", False)
                    except Exception: pass
                try:
                    mtime = os.path.getmtime(CAM2_FRAME_TMP)
                    if mtime != ultimo_ts:
                        arr = np.load(CAM2_FRAME_TMP)
                        if _ok(arr):
                            ultimo_ts = mtime
                            with self._lock_cam2:
                                self.frame_cam2 = arr
                except Exception:
                    pass
                owner_prev = "orquestador"; time.sleep(0.033); continue

            if cap is None or not cap.isOpened():
                ahora = time.time()

                if ahora - ultimo_intento < cooldown:
                    time.sleep(0.2); continue
                ultimo_intento = ahora
                if owner_prev == "orquestador":
                    time.sleep(1.2)
                cap = _abrir()
                if cap is None:
                    fallos_abrir += 1
                    cooldown = min(5.0, 1.5 + fallos_abrir * 0.5)
                    owner_prev = "interface"
                    time.sleep(cooldown); continue
                fallos_abrir = 0; cooldown = 1.5; t_abierta = time.time()
                try: escribir_estado_campo("cam2_interfaz_lista", True)
                except Exception: pass

            ret = False; frame = None
            for _ in range(8):
                ret, frame = cap.read()
                if ret and _ok(frame):
                    break
                time.sleep(0.01)
            if not ret or not _ok(frame):
                try: cap.release()
                except Exception: pass
                cap = None; owner_prev = "interface"
                time.sleep(0.2); continue

            if time.time() - t_abierta > 3.0:
                fsum = int(frame[::8, ::8].sum())
                if fsum == ultimo_frame_sum:
                    frames_congelados += 1
                    if frames_congelados >= 90:
                        try: cap.release()
                        except Exception: pass
                        cap = None; owner_prev = "interface"
                        time.sleep(0.2); continue
                else:
                    frames_congelados = 0; ultimo_frame_sum = fsum

            with self._lock_cam2:
                self.frame_cam2 = frame
            owner_prev = "interface"; time.sleep(0.033)

    def _cargar_yolo_en_hilo(self):
        threading.Thread(target=self._cargar_yolo, daemon=True).start()

    def _cargar_yolo(self):
        if not YOLO_OK:
            return
        try:
            modelo = YOLO("yolov8x-worldv2.pt")
            modelo.set_classes(YOLO_CLASSES)
            self.yolo_model = modelo
            print("[YOLO] Modelo listo.", flush=True)
        except Exception as e:
            print(f"[YOLO] Error al cargar: {e}", flush=True)

    def _analizar_modo(self, frame):
        if frame is None or self.yolo_model is None:
            self._escribir_objetivo(None)
            return self.MODO_VACIO
        try:
            h, w = frame.shape[:2]
            res = self.yolo_model.predict(
                frame, conf=YOLO_CONF, iou=0.25, verbose=False
            )[0]

            if len(res.boxes) == 0:
                self._escribir_objetivo(None)
                return self.MODO_VACIO

            mejor = max(res.boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = mejor.xyxy[0].tolist()
            cx_px = (x1 + x2) / 2
            cy_px = (y1 + y2) / 2
            objetivo = {
                "cx_norm": round(cx_px / w, 5),
                "cy_norm": round(cy_px / h, 5),
                "conf":    round(float(mejor.conf[0]), 4),
                "listo":   True,
            }
            self._escribir_objetivo(objetivo)
            return self.MODO_SOLIDO

        except Exception:
            self._escribir_objetivo(None)
            return self.MODO_VACIO

    def _escribir_objetivo(self, objetivo):
        try:
            estado = leer_estado()
            if not estado:
                estado = dict(ESTADO_INICIAL)
            estado["objetivo"] = objetivo
            tmp = ESTADO_JSON + ".tmp.gui"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(estado, f, indent=2, ensure_ascii=False)
            os.replace(tmp, ESTADO_JSON)
        except Exception:
            pass

    def _iniciar_voz(self):
        self.hilo_voz = HiloVoz(
            cb_iniciar=lambda: self.root.after(0, self._voz_iniciar),
            cb_detener=lambda: self.root.after(0, self._voz_detener),
            cb_estado=lambda msg: self.root.after(0, self._set_estado, msg),
            cb_sopa=lambda: self.root.after(0, self._voz_sopa),
        )
        self.hilo_voz.start()

    def _voz_iniciar(self):
        self._set_estado("[VOZ] 'Quiero comer' reconocido.")
        if not self.sistema_activo:
            self._iniciar_sistema()
            self._enviar_comando_cuando_reposo("INICIAR_CICLO")
            return
        print("[VOZ] 'Quiero comer' -> INICIAR_CICLO.", flush=True)
        enviar_comando("INICIAR_CICLO")
        self._set_estado("Modo solido solicitado por voz.")

    def _voz_sopa(self):
        self._set_estado("[VOZ] 'Quiero sopa' reconocido.")
        if not self.sistema_activo:
            self._iniciar_sistema()
            self._enviar_comando_cuando_reposo("INICIAR_CICLO_LIQUIDO")
            return
        print("[VOZ] 'Quiero sopa' -> INICIAR_CICLO_LIQUIDO.", flush=True)
        enviar_comando("INICIAR_CICLO_LIQUIDO")
        self._set_estado("Modo sopa solicitado por voz.")

    def _enviar_comando_cuando_reposo(self, comando, timeout=30.0):
        def _esperar():
            t0 = time.time()
            while time.time() - t0 < timeout:
                if not self.sistema_activo:
                    return
                if leer_estado().get("fase_actual", "") == "REPOSO":
                    enviar_comando(comando)
                    return
                time.sleep(0.2)
        threading.Thread(target=_esperar, daemon=True).start()

    def _btn_quiero_comer(self):
        print("[BTN] QUIERO COMER presionado.", flush=True)
        self._voz_iniciar()

    def _btn_quiero_sopa(self):
        print("[BTN] QUIERO SOPA presionado.", flush=True)
        self._voz_sopa()

    def _actualizar_botones_reposo(self):
        if not self.sistema_activo:
            if hasattr(self, "btn_comer"):
                self.btn_comer.config(state="disabled")
                self.btn_sopa.config(state="disabled")
                self.lbl_reposo.config(fg="#888888")
            return
        try:
            fase = leer_estado().get("fase_actual", "")
        except Exception:
            fase = ""
        en_reposo  = (fase == "REPOSO")
        estado_btn = "normal" if en_reposo else "disabled"
        color_lbl  = "#2060A0" if en_reposo else "#888888"
        if hasattr(self, "btn_comer"):
            self.btn_comer.config(state=estado_btn)
            self.btn_sopa.config(state=estado_btn)
            self.lbl_reposo.config(fg=color_lbl)

    def _voz_detener(self):
        self._set_estado("[VOZ] 'Ya no quiero comer'. Volviendo a HOME...")
        print("[VOZ] PARANDO - usuario dijo 'Ya no quiero comer'.", flush=True)
        hablar("Parando alimentacion")
        self.alarma_actual = self.ALARMA_PARANDO
        self._ultima_voz = "Parando alimentacion"
        enviar_comando("PARAR")

    def _ciclo_actualizacion(self):
        self._mostrar_camaras()
        self._actualizar_modo()
        self._comprobar_voz()
        self._actualizar_alarmas()
        self._actualizar_botones_reposo()
        self._actualizar_estadisticas()
        self.root.after(33, self._ciclo_actualizacion)

    def _comprobar_voz(self):
        estado = leer_estado()
        voz = estado.get("voz", "")
        if voz and voz != self._ultima_voz:
            self._ultima_voz = voz
            print(f"[INTERFAZ] Voz recibida desde orquestador: '{voz}'", flush=True)
            hablar(voz)
            if "agarre" in voz.lower() and "parando" not in voz.lower():
                self.alarma_actual = self.ALARMA_INICIANDO
            elif "llevando" in voz.lower():
                self.alarma_actual = self.ALARMA_LLEVANDO
            elif "listo" in voz.lower():
                self.alarma_actual = self.ALARMA_LISTO
            elif "parando" in voz.lower():
                self.alarma_actual = self.ALARMA_PARANDO

    def _mostrar_camaras(self):
        with self._lock_cam1:
            f1 = self.frame_cam1.copy() if self.frame_cam1 is not None else None
        with self._lock_cam2:
            f2 = self.frame_cam2.copy() if self.frame_cam2 is not None else None

        for frame, lbl_attr, photo_attr in (
            (f1, "lbl_cam1", "_photo1"),
            (f2, "lbl_cam2", "_photo2"),
        ):
            lbl = getattr(self, lbl_attr)
            if frame is not None:
                foto = self._cv2_a_tk(frame)
                if foto:
                    setattr(self, photo_attr, foto)
                    lbl.config(image=foto)
            else:

                lbl.config(image="")

    def _actualizar_modo(self):
        try:
            est = leer_estado()
        except Exception:
            est = {}
        if est.get("modo_liquido_activo", False):
            self.modo_liquido_activo = True
            self._set_ind(self.ind_vacio,   False)
            self._set_ind(self.ind_solido,  False)
            self._set_ind(self.ind_liquido, True)
            return
        self.modo_liquido_activo = False
        _yolo_on = est.get("yolo_activo", True)
        if not _yolo_on:
            self._set_ind(self.ind_vacio,   True)
            self._set_ind(self.ind_solido,  False)
            self._set_ind(self.ind_liquido, False)
            return

        ahora = time.time()
        if ahora - self.ultimo_detect >= INTERVALO_DETECT:
            self.ultimo_detect = ahora
            with self._lock_cam1:
                f1 = self.frame_cam1.copy() if self.frame_cam1 is not None else None

            nuevo = self._analizar_modo(f1)

            if nuevo != self.modo_actual:
                self.modo_actual = nuevo
                if nuevo == self.MODO_VACIO:
                    self._set_estado("Modo VACIO: no se detecta comida en el plato.")

                    fase = leer_estado().get("fase_actual", "INACTIVO")
                    fases_activas = (
                        "AGARRE", "PREDICCION", "VERIFICACION",
                        "ENTREGA", "COMIENDO", "CICLO COMPLETO",
                    )
                    if self.sistema_activo and not any(f in fase for f in fases_activas):
                        hablar("No hay comida")
                elif nuevo == self.MODO_SOLIDO:
                    self._set_estado("Modo SOLIDO: comida detectada en el plato.")

        self._set_ind(self.ind_vacio,   self.modo_actual == self.MODO_VACIO)
        self._set_ind(self.ind_solido,  self.modo_actual == self.MODO_SOLIDO)
        self._set_ind(self.ind_liquido, False)

    def _actualizar_alarmas(self):

        self._set_ind(self.ind_iniciando, self.alarma_actual == self.ALARMA_INICIANDO)
        self._set_ind(self.ind_llevando,  self.alarma_actual == self.ALARMA_LLEVANDO)
        self._set_ind(self.ind_listo,     self.alarma_actual == self.ALARMA_LISTO)
        self._set_ind(self.ind_parando,   self.alarma_actual == self.ALARMA_PARANDO)

    def _actualizar_estadisticas(self):
        estado  = leer_estado()
        ciclos  = estado.get("ciclo", 0)
        agarres = estado.get("agarres_ok", 0)
        self.lbl_ciclos.config(
            text=f"Ciclos: {ciclos}  |  Entregas: {agarres}"
        )

    def _on_modo_liquido_finalizado(self):
        self.modo_liquido_activo = False
        self.modo_actual         = self.MODO_VACIO
        self.root.after(0, self._set_estado,
                        "Servicio de sopa completado. Sistema listo.")

    def _toggle_sistema(self):
        if self.sistema_activo:
            self._detener_sistema()
        else:
            self._iniciar_sistema()

    def _iniciar_sistema(self):
        if self.sistema_activo:
            return

        if not os.path.isfile(ORQUESTADOR):
            self._set_estado(
                f"ERROR: No se encontro '{os.path.basename(ORQUESTADOR)}' "
                "en la carpeta del proyecto."
            )
            return

        self._camaras_activas = True

        for ruta in (CAM2_FRAME_TMP, CAM2_FRAME_TMP_WRITE):
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
            except Exception:
                pass

        args = [
            sys.executable, "-u", ORQUESTADOR,
            "--modo",      "trozo",
            "--cam1",      str(CAM1_INDEX),
            "--cam2",      str(CAM2_INDEX),
            "--puerto",    SERIAL_PORT,
            "--threshold", "20.0",
            "--eating",    "15.0",
            "--far",       "20.0",
            "--modelo",    MODELO_PT,
            "--calib",     CALIB_JSON,
        ]
        self._last_args = args

        import os as _os
        _env = _os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8:replace"
        _env["PYTHONUTF8"]       = "1"

        try:
            self.proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                cwd=DIR_BASE,
                env=_env,
            )
        except Exception as e:
            self._set_estado(f"ERROR al iniciar orquestador: {e}")
            self._camaras_activas = False
            return

        time.sleep(0.4)
        if self.proc.poll() is not None:
            salida = self.proc.stdout.read()
            linea  = salida.strip().splitlines()[-1] if salida.strip() else "sin detalle"
            self._set_estado(f"ERROR: orquestador termino al instante -> {linea}")
            self.proc = None
            self._camaras_activas = False
            return

        self.sistema_activo = True
        enviar_comando("NINGUNO")
        self.btn_sistema.config(
            text="DETENER",
            bg=BTN_ROJO,
            activebackground=BTN_ROJO_H,
        )
        self._set_estado("Sistema ACTIVO. Diga 'Quiero comer' o 'Quiero sopa'.")
        hablar("Sistema iniciado")

        threading.Thread(
            target=self._leer_stdout_orquestador,
            daemon=True,
        ).start()

    def _leer_stdout_orquestador(self):
        MAX_REINICIOS_CRASH = 5
        proc = self.proc
        if proc is None:
            return

        try:
            for linea in proc.stdout:
                if self.proc is not proc:
                    return
                linea = linea.rstrip("\n")
                if linea:
                    print(f"[ORQ] {linea}", flush=True)
                    self.root.after(0, self._set_estado, linea)
        except Exception as e:
            print(f"[ORQ] Excepcion leyendo pipe: {e}", flush=True)

        try:
            codigo_final = proc.wait(timeout=10)
        except Exception:
            codigo_final = proc.poll()

        print(f"[ORQ] Orquestador termino - codigo: {codigo_final} "
              f"({hex((codigo_final or 0) & 0xFFFFFFFF)})", flush=True)

        if self.proc is not proc:
            return
        if not self.sistema_activo:
            return

        ES_CRASH = (codigo_final is not None and codigo_final != 0)

        if ES_CRASH:
            n = getattr(self, "_reinicios_crash", 0) + 1
            self._reinicios_crash = n
            if n <= MAX_REINICIOS_CRASH:
                print(f"[ORQ] CRASH ({hex(codigo_final & 0xFFFFFFFF)}) "
                      f"- reiniciando ({n}/{MAX_REINICIOS_CRASH})...", flush=True)
                self.root.after(0, self._set_estado,
                    f"Reiniciando orquestador ({n}/{MAX_REINICIOS_CRASH})...")

                try:
                    escribir_estado_campo("cam2_owner",          "interface")
                    escribir_estado_campo("cam2_interfaz_lista", True)
                except Exception:
                    pass
                time.sleep(2.0)
                self.root.after(0, self._relanzar_orquestador)
                return
            print("[ORQ] Demasiados crashes - deteniendo sistema.", flush=True)

        self._reinicios_crash = 0
        if self.modo_liquido_activo:
            self.root.after(0, self._on_modo_liquido_finalizado)
            self.root.after(800, self._on_sistema_detenido)
        else:
            self.root.after(0, self._set_estado,
                f"Orquestador termino (codigo {codigo_final}). "
                "Presione INICIAR para reiniciar.")
            self.root.after(0, self._on_sistema_detenido)

    def _relanzar_orquestador(self):
        if not self.sistema_activo:
            return
        print("[ORQ] Relanzando orquestador tras crash...", flush=True)

        args = getattr(self, "_last_args", None)
        if args is None:
            print("[ORQ] _last_args no disponible - no se puede relanzar.", flush=True)
            self.root.after(0, self._on_sistema_detenido)
            return
        import os as _os
        _env = _os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8:replace"
        _env["PYTHONUTF8"]       = "1"
        try:
            self.proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                cwd=DIR_BASE,
                env=_env,
            )
            print("[ORQ] Orquestador relanzado tras crash.", flush=True)
            threading.Thread(
                target=self._leer_stdout_orquestador,
                daemon=True,
                name="StdoutReader-Reinicio",
            ).start()
        except Exception as e:
            print(f"[ORQ] No se pudo relanzar: {e}", flush=True)
            self.root.after(0, self._on_sistema_detenido)

    def _detener_sistema(self):
        if not self.sistema_activo:
            return
        enviar_comando("DETENER")
        self._set_estado("Deteniendo sistema...")

        def _forzar():
            time.sleep(3.5)

            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            self.root.after(0, self._on_sistema_detenido)

        threading.Thread(target=_forzar, daemon=True).start()

    def _on_sistema_detenido(self):
        self.sistema_activo      = False
        self.alarma_actual       = self.ALARMA_NINGUNA
        self.proc                = None
        self.modo_liquido_activo = False
        self.modo_actual         = self.MODO_VACIO
        self._camaras_activas    = False

        try:
            escribir_estado_campo("cam2_owner", "interface")
        except Exception:
            pass

        with self._lock_cam1:
            self.frame_cam1 = None
        with self._lock_cam2:
            self.frame_cam2 = None
        self.btn_sistema.config(
            text="INICIAR",
            bg=BTN_VERDE,
            activebackground=BTN_VERDE_H,
        )
        self._set_estado("Sistema detenido. Listo.")
        hablar("Sistema detenido")

    def _cv2_a_tk(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize(
                (CAM_ANCHO, CAM_ALTO), Image.BILINEAR
            )
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def _set_estado(self, texto):
        self.lbl_estado.config(text=texto)

    def _cerrar(self):
        if self.sistema_activo:
            self._detener_sistema()
            time.sleep(1.2)
        if self.hilo_voz:
            self.hilo_voz.detener()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    NutribotGUI(root)
    root.mainloop()