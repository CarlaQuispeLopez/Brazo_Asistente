"""
detectar_camaras.py — Encuentra el índice de todas las cámaras conectadas
=========================================================================
Prueba los índices 0 al 9 e intenta abrir cada cámara.
Para cada cámara que abre correctamente muestra una ventana en vivo
con su índice marcado en pantalla para que puedas identificar cuál es cuál.

USO:
    python detectar_camaras.py

CONTROLES:
    ESPACIO  → pasar a la siguiente cámara
    ESC / Q  → salir del programa
"""

import cv2
import sys

MAX_INDEX  = 10   # Probar índices del 0 al 9
SHOW_TIME  = 8    # Segundos máximos mostrando cada cámara (luego pasa sola)
RESOLUTION = (640, 480)


def probar_camaras():
    encontradas = []

    print("=" * 55)
    print("  DETECTOR DE CÁMARAS")
    print("  Probando índices 0 al", MAX_INDEX - 1)
    print("=" * 55)

    # ── Paso 1: escanear qué índices abren ────────────────────
    print("\nEscaneando cámaras disponibles...\n")
    for idx in range(MAX_INDEX):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # CAP_DSHOW = más rápido en Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✔  Índice {idx} — OK  ({w}×{h})")
                encontradas.append(idx)
            else:
                print(f"  ~  Índice {idx} — se abre pero no devuelve frames")
            cap.release()
        else:
            print(f"  ✗  Índice {idx} — no disponible")

    print()

    if not encontradas:
        print("  No se encontró ninguna cámara. Verifica las conexiones.")
        return

    print(f"  Cámaras con video real: {encontradas}")
    print()
    print("─" * 55)
    print("  Ahora se mostrará cada cámara en vivo.")
    print("  ESPACIO = siguiente cámara  |  ESC/Q = salir")
    print("─" * 55)

    # ── Paso 2: mostrar cada cámara en vivo ───────────────────
    for idx in encontradas:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

        if not cap.isOpened():
            print(f"  No se pudo reabrir índice {idx}, saltando...")
            continue

        ventana = f"CÁMARA  ÍNDICE {idx}  |  ESPACIO=siguiente  ESC=salir"
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ventana, 700, 540)

        print(f"\n  Mostrando ÍNDICE {idx} — presiona ESPACIO para continuar...")

        t_inicio = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Frame negro de aviso
                frame = __import__("numpy").zeros(
                    (RESOLUTION[1], RESOLUTION[0], 3), dtype=__import__("numpy").uint8)
                cv2.putText(frame, "Sin señal", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)

            h, w = frame.shape[:2]

            # Tiempo transcurrido
            t_actual  = cv2.getTickCount()
            segundos  = (t_actual - t_inicio) / cv2.getTickFrequency()
            restantes = max(0, SHOW_TIME - int(segundos))

            # ── Overlay de información ────────────────────────
            # Fondo semitransparente en la barra superior
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            # Índice grande
            cv2.putText(frame, f"INDICE: {idx}", (12, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)

            # Resolución y timer
            cv2.putText(frame, f"{w}x{h}  |  Auto-pasa en {restantes}s",
                        (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Instrucciones abajo
            cv2.rectangle(frame, (0, h-36), (w, h), (0,0,0), -1)
            cv2.putText(frame, "ESPACIO = siguiente camara   |   ESC / Q = salir",
                        (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

            cv2.imshow(ventana, frame)
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord('q')):          # ESC o Q → salir
                cap.release()
                cv2.destroyAllWindows()
                print("\n  Salida manual. Cámaras encontradas:", encontradas)
                return

            if key == 32:                       # ESPACIO → siguiente
                break

            if segundos >= SHOW_TIME:           # Tiempo agotado → siguiente
                print(f"    (tiempo agotado, pasando a la siguiente)")
                break

        cap.release()
        cv2.destroyWindow(ventana)
        cv2.waitKey(1)

    # ── Resumen final ─────────────────────────────────────────
    cv2.destroyAllWindows()
    print("\n" + "=" * 55)
    print("  RESUMEN — Cámaras con video disponible:")
    for idx in encontradas:
        print(f"    Índice {idx}")
    print()
    print("  Usa el índice correcto en auto_brazo_completo.py:")
    print("    CAMERA_1_INDEX = <índice del brazo>")
    print("    CAMERA_2_INDEX = <índice de la cara>")
    print("=" * 55)


if __name__ == "__main__":
    try:
        probar_camaras()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n  Interrumpido por el usuario.")
        sys.exit(0)
