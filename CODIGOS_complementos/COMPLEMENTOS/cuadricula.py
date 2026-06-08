from ultralytics import YOLOWorld
import cv2
import numpy as np
import json
import os

CAMARA_INDEX = 2

COLOR_PIEZA       = (0, 255, 0)
COLOR_CUADRICULA  = (0, 255, 255)
COLOR_PUNTO       = (0, 0, 255)
COLOR_PUNTO_BORDE = (255, 255, 255)

VENTANA_ANCHO = 1280
VENTANA_ALTO  = 720

GRID_COLS = 12
GRID_ROWS = 9

ARCHIVO_CALIBRACION = "calibracion_cuadricula.json"

# ─── Estado global ────────────────────────────────────────────────────────────
puntos          = []
calib_lista     = False
matriz_perspect = None

NOMBRES_PUNTOS   = [
    "P1 - Esquina SUPERIOR IZQUIERDA",
    "P2 - Esquina SUPERIOR DERECHA",
    "P3 - Esquina INFERIOR DERECHA",
    "P4 - Esquina INFERIOR IZQUIERDA",
]
ETIQUETAS_CORTAS = ["P1 SUP-IZQ", "P2 SUP-DER", "P3 INF-DER", "P4 INF-IZQ"]


# ─── GUARDAR / CARGAR ─────────────────────────────────────────────────────────
def guardar_calibracion():
    datos = {
        "grid_cols": GRID_COLS,
        "grid_rows": GRID_ROWS,
        "puntos"   : [list(p) for p in puntos],
    }
    with open(ARCHIVO_CALIBRACION, "w") as f:
        json.dump(datos, f, indent=4)
    print(f"\n  Calibracion guardada en '{ARCHIVO_CALIBRACION}'")
    print(f"  Puntos: {puntos}\n")


def cargar_calibracion():
    global puntos, calib_lista, matriz_perspect
    if not os.path.exists(ARCHIVO_CALIBRACION):
        return False
    try:
        with open(ARCHIVO_CALIBRACION, "r") as f:
            datos = json.load(f)
        pts = [tuple(p) for p in datos["puntos"]]
        if len(pts) != 4:
            print("  Archivo de calibracion incompleto.")
            return False
        puntos = pts
        _calcular_homografia()
        calib_lista = True
        print(f"  Calibracion cargada desde '{ARCHIVO_CALIBRACION}'")
        print(f"  Grid: {datos.get('grid_cols')}x{datos.get('grid_rows')}")
        print(f"  Puntos: {puntos}\n")
        return True
    except Exception as e:
        print(f"  No se pudo cargar la calibracion: {e}")
        return False


# ─── CALLBACK RATON ───────────────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global puntos, calib_lista, matriz_perspect
    if calib_lista:
        return
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
        puntos.append((x, y))
        print(f"  OK  {NOMBRES_PUNTOS[len(puntos)-1]}  -> ({x}, {y})")
        if len(puntos) == 4:
            _calcular_homografia()
            guardar_calibracion()
            calib_lista = True
            print("  Cuadricula generada. Iniciando deteccion...\n")


def _calcular_homografia():
    global matriz_perspect
    src = np.float32([
        [0,         0        ],
        [GRID_COLS, 0        ],
        [GRID_COLS, GRID_ROWS],
        [0,         GRID_ROWS],
    ])
    dst = np.float32(puntos)
    matriz_perspect = cv2.getPerspectiveTransform(src, dst)


# ─── CUADRICULA ───────────────────────────────────────────────────────────────
def _transformar(pts_norm):
    pts = np.float32(pts_norm).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matriz_perspect).reshape(-1, 2)


def dibujar_cuadricula(frame):
    if matriz_perspect is None:
        return frame
    overlay = frame.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            corn = [[c,r],[c+1,r],[c+1,r+1],[c,r+1]]
            pts_img = _transformar(corn).astype(np.int32)
            cv2.fillPoly(overlay, [pts_img], (0, 60, 60))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for c in range(GRID_COLS + 1):
        p1 = _transformar([[c, 0        ]]).astype(int)[0]
        p2 = _transformar([[c, GRID_ROWS]]).astype(int)[0]
        grosor = 2 if c in (0, GRID_COLS) else 1
        cv2.line(frame, tuple(p1), tuple(p2), COLOR_CUADRICULA, grosor)

    for r in range(GRID_ROWS + 1):
        p1 = _transformar([[0,         r]]).astype(int)[0]
        p2 = _transformar([[GRID_COLS, r]]).astype(int)[0]
        grosor = 2 if r in (0, GRID_ROWS) else 1
        cv2.line(frame, tuple(p1), tuple(p2), COLOR_CUADRICULA, grosor)

    for i, pt in enumerate(puntos):
        cv2.circle(frame, pt, 9, COLOR_PUNTO, -1)
        cv2.circle(frame, pt, 9, COLOR_PUNTO_BORDE, 2)
        cv2.putText(frame, ETIQUETAS_CORTAS[i], (pt[0]+12, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CUADRICULA, 1)
    return frame


# ─── PANTALLA CALIBRACION ─────────────────────────────────────────────────────
def pantalla_calibracion(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame,
                f"CALIBRACION ({GRID_COLS}x{GRID_ROWS}) - los puntos se guardan automaticamente",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    idx = len(puntos)
    if idx < 4:
        cv2.putText(frame, f">> Clic en:  {NOMBRES_PUNTOS[idx]}",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Puntos: {idx}/4   |   'r' reiniciar   |   'q' salir",
                    (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        cv2.putText(frame, "4/4 puntos marcados - generando cuadricula...",
                    (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    for i, pt in enumerate(puntos):
        cv2.circle(frame, pt, 9, COLOR_PUNTO, -1)
        cv2.circle(frame, pt, 9, COLOR_PUNTO_BORDE, 2)
        cv2.putText(frame, ETIQUETAS_CORTAS[i], (pt[0]+12, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
    if len(puntos) > 1:
        for i in range(len(puntos) - 1):
            cv2.line(frame, puntos[i], puntos[i+1], COLOR_CUADRICULA, 1)
        if len(puntos) == 4:
            cv2.line(frame, puntos[3], puntos[0], COLOR_CUADRICULA, 1)
    return frame


def _loop_calibracion(cap, nombre_ventana):
    global puntos, calib_lista, matriz_perspect
    puntos.clear()
    calib_lista     = False
    matriz_perspect = None
    while not calib_lista:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = pantalla_calibracion(frame)
        cv2.imshow(nombre_ventana, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('r'):
            puntos.clear()
            calib_lista     = False
            matriz_perspect = None
            print("  Puntos reiniciados.")
    return True


# ─── DETECTOR ALIMENTOS ───────────────────────────────────────────────────────
class DetectorAlimentos:
    def __init__(self):
        print("Cargando modelo YOLO-World...")
        self.model = YOLOWorld('yolov8m-world.pt')
        self.model.set_classes([
            "apple","pear","peach","plum","apricot","cherry",
            "strawberry","raspberry","blueberry","blackberry",
            "grape","watermelon","melon","cantaloupe","honeydew",
            "banana","mango","papaya","pineapple chunk","kiwi",
            "orange slice","mandarin","lemon slice","lime slice",
            "fig","date","lychee","guava","passion fruit",
            "dragon fruit","star fruit","persimmon","pomegranate seed",
            "tomato","cherry tomato","carrot piece","broccoli floret",
            "cauliflower","lettuce piece","cucumber slice","zucchini",
            "eggplant","bell pepper","corn kernel","pea",
            "green bean","asparagus","artichoke","celery piece",
            "beet","radish","turnip","potato chunk","sweet potato",
            "mushroom","onion piece","leek","spinach","kale",
            "cabbage piece","brussels sprout","bok choy",
            "chicken piece","beef piece","pork piece","lamb piece",
            "turkey piece","sausage slice","meatball","nugget",
            "shrimp","fish piece","salmon chunk","tuna piece",
            "squid piece","octopus piece","crab meat",
            "boiled egg","fried egg piece","omelette piece",
            "tofu cube","tempeh piece",
            "pasta piece","noodle","gnocchi","dumpling",
            "rice ball","bread piece","crouton",
            "cheese cube","mozzarella","ham piece",
            "olive","pickle slice","sun-dried tomato",
            "chickpea","lentil","bean",
            "food piece","fruit piece","vegetable piece","meat piece",
        ])
        print(f"Modelo listo. {len(self.model.names)} clases cargadas.\n")

    def detectar(self, frame):
        results = self.model.predict(frame, conf=0.20, iou=0.25, verbose=False)[0]
        alimentos = []
        for box in results.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            alimentos.append({
                'clase'    : "TROZO DE COMIDA",
                'centro_px': (int((x1+x2)/2), int((y1+y2)/2)),
                'bbox'     : (int(x1),int(y1),int(x2),int(y2)),
                'confianza': float(box.conf[0]),
            })
        alimentos.sort(key=lambda x: x['confianza'], reverse=True)
        return alimentos

    def visualizar(self, frame, alimentos):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (520,40), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"Venus USB Cam  |  Trozos detectados: {len(alimentos)}",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        for a in alimentos:
            x1,y1,x2,y2 = a['bbox']
            cx,cy = a['centro_px']
            cv2.rectangle(frame,(x1,y1),(x2,y2),COLOR_PIEZA,2)
            cv2.circle(frame,(cx,cy),6,(0,0,255),-1)
            cv2.circle(frame,(cx,cy),6,(255,255,255),1)
            label = f"TROZO DE COMIDA {a['confianza']:.2f}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
            cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+4,y1),COLOR_PIEZA,-1)
            cv2.putText(frame,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        if alimentos:
            x1,y1,x2,y2 = alimentos[0]['bbox']
            cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(0,255,255),3)
            cv2.putText(frame,"OBJETIVO",(x1,y2+22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
        return frame


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Abriendo camara (indice {CAMARA_INDEX})...")
    cap = cv2.VideoCapture(CAMARA_INDEX)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir camara {CAMARA_INDEX}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camara abierta.\n")

    NOMBRE_VENTANA = "Brazo Robotico - Deteccion de Alimentos"
    cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NOMBRE_VENTANA, VENTANA_ANCHO, VENTANA_ALTO)
    cv2.setMouseCallback(NOMBRE_VENTANA, mouse_callback)

    # ── Intentar cargar calibracion previa ───────────────────────────────
    calibracion_previa = cargar_calibracion()

    if calibracion_previa:
        print("Se encontro una calibracion guardada.")
        print("  'u'  Usar calibracion guardada")
        print("  'n'  Hacer nueva calibracion\n")

        esperando = True
        while esperando:
            ret, frame = cap.read()
            if not ret:
                break
            frame = dibujar_cuadricula(frame)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (frame.shape[1], 70), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"Calibracion guardada encontrada  ({ARCHIVO_CALIBRACION})",
                        (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
            cv2.putText(frame, "Presiona  'u' = Usar esta cuadricula    'n' = Nueva calibracion",
                        (12,58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(NOMBRE_VENTANA, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('u'):
                print("Usando calibracion guardada.\n")
                esperando = False
            elif key == ord('n'):
                print("Iniciando nueva calibracion...\n")
                if not _loop_calibracion(cap, NOMBRE_VENTANA):
                    cap.release(); cv2.destroyAllWindows(); return
                esperando = False
            elif key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return
    else:
        print("No se encontro calibracion previa. Iniciando calibracion...\n")
        print("Orden de puntos:  P1 SUP-IZQ -> P2 SUP-DER -> P3 INF-DER -> P4 INF-IZQ")
        print("Teclas:  'r' reiniciar  |  'q' salir\n")
        if not _loop_calibracion(cap, NOMBRE_VENTANA):
            cap.release(); cv2.destroyAllWindows(); return

    # ── Deteccion con cuadricula ──────────────────────────────────────────
    detector = DetectorAlimentos()
    print("Deteccion iniciada.")
    print("Teclas:  'r' recalibrar  |  's' guardar de nuevo  |  'q' salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame.")
            break

        frame = dibujar_cuadricula(frame)

        alimentos = detector.detectar(frame)
        if alimentos:
            obj = alimentos[0]
            print(f"OBJETIVO -> {obj['clase']:<25} "
                  f"Centro: {obj['centro_px']}   Conf: {obj['confianza']:.2f}")

        frame = detector.visualizar(frame, alimentos)

        h = frame.shape[0]
        cv2.putText(frame,
                    f"Cuadricula {GRID_COLS}x{GRID_ROWS} activa  |  "
                    f"'r' recalibrar  |  's' guardar  |  'q' salir",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,255), 1)

        cv2.imshow(NOMBRE_VENTANA, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Saliendo...")
            break
        elif key == ord('s'):
            guardar_calibracion()
        elif key == ord('r'):
            print("\nRecalibrando...\n")
            if not _loop_calibracion(cap, NOMBRE_VENTANA):
                break
            print("Nueva cuadricula activa.\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()