"""
test_deteccion_plato.py
=======================
Prueba rapida para ver si YOLO detecta tu plato sopero.
Muestra el video en tiempo real con todas las detecciones visibles.

USO:
  python test_deteccion_plato.py

TECLAS:
  ESC o Q  -> salir
  S        -> guardar captura del frame actual como screenshot.png
"""

import cv2
import numpy as np
from ultralytics import YOLO

CAMARA_INDEX    = 2
MODELO_YOLO_SEG = "yolov8n-seg.pt"

# Clases COCO que nos interesan para el plato
CLASES_PLATO = {"bowl", "cup", "plate", "dish"}

# Confianza minima para mostrar una deteccion
CONFIANZA_MIN = 0.20


def main():
    print("=" * 55)
    print("  TEST DE DETECCION — Plato sopero")
    print("=" * 55)

    print(f"\n[YOLO] Cargando {MODELO_YOLO_SEG}...")
    model = YOLO(MODELO_YOLO_SEG)
    print("[YOLO] Listo.\n")

    print(f"[Cam] Abriendo camara {CAMARA_INDEX}...")
    cap = cv2.VideoCapture(CAMARA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {CAMARA_INDEX}.")
        return

    print("[Cam] Lista. Mostrando detecciones...\n")
    print("Clases que activan deteccion de plato:", CLASES_PLATO)
    print("Todas las demas clases aparecen en gris para referencia.\n")
    print("ESC / Q = salir    S = guardar screenshot\n")

    ventana = "Test deteccion plato — ESC/Q=salir  S=screenshot"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fh, fw = frame.shape[:2]
        display = frame.copy()

        results = model(frame, verbose=False)[0]

        plato_encontrado  = False
        plato_confianza   = 0.0
        plato_clase       = ""

        # Dibujar TODAS las detecciones para ver que ve YOLO
        for i in range(len(results.boxes)):
            cls_id     = int(results.boxes.cls[i])
            confianza  = float(results.boxes.conf[i])
            cls_nombre = model.names[cls_id]

            if confianza < CONFIANZA_MIN:
                continue

            # Bounding box
            x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy().astype(int)

            es_plato = cls_nombre.lower() in CLASES_PLATO

            if es_plato:
                color = (0, 255, 0)      # Verde = clase de plato
                grosor = 3
                plato_encontrado = True
                if confianza > plato_confianza:
                    plato_confianza = confianza
                    plato_clase     = cls_nombre
            else:
                color = (120, 120, 120)  # Gris = otra clase (referencia)
                grosor = 1

            cv2.rectangle(display, (x1, y1), (x2, y2), color, grosor)
            etiqueta = f"{cls_nombre} {confianza:.2f}"
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, etiqueta, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Dibujar mascaras de segmentacion si hay
        if results.masks is not None:
            for i in range(len(results.masks.data)):
                cls_id     = int(results.boxes.cls[i])
                confianza  = float(results.boxes.conf[i])
                cls_nombre = model.names[cls_id]

                if confianza < CONFIANZA_MIN:
                    continue

                mask_raw   = results.masks.data[i].cpu().numpy()
                mask_frame = cv2.resize(mask_raw, (fw, fh),
                                        interpolation=cv2.INTER_LINEAR)
                mask_uint8 = (mask_frame > 0.5).astype(np.uint8)

                es_plato = cls_nombre.lower() in CLASES_PLATO
                capa     = np.zeros_like(frame)

                if es_plato:
                    capa[mask_uint8 > 0] = (0, 200, 0)     # Verde
                    display = cv2.addWeighted(display, 0.7, capa, 0.3, 0)
                else:
                    capa[mask_uint8 > 0] = (80, 80, 80)    # Gris
                    display = cv2.addWeighted(display, 0.85, capa, 0.15, 0)

        # Panel de estado
        if plato_encontrado:
            msg   = f"PLATO DETECTADO: {plato_clase}  conf={plato_confianza:.2f}"
            color_panel = (0, 180, 0)
        else:
            msg   = "Sin deteccion de plato"
            color_panel = (0, 0, 200)

        cv2.rectangle(display, (0, fh - 45), (fw, fh), (20, 20, 20), -1)
        cv2.putText(display, msg,
                    (10, fh - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    color_panel, 2)
        cv2.putText(display, f"Cam {CAMARA_INDEX}  |  conf min {CONFIANZA_MIN}  |  ESC/Q=salir  S=screenshot",
                    (10, fh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (150, 150, 150), 1)

        cv2.imshow(ventana, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("s"), ord("S")):
            cv2.imwrite("screenshot.png", display)
            print(f"[Screenshot] Guardado como screenshot.png")
            print(f"  Plato detectado: {plato_encontrado}")
            if plato_encontrado:
                print(f"  Clase: {plato_clase}  Confianza: {plato_confianza:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nTest finalizado.")


if __name__ == "__main__":
    main()