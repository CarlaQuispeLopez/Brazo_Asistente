# deteccion_alimentos_droidcam.py
# Requiere: pip install ultralytics opencv-python

from ultralytics import YOLOWorld
import cv2

# ─────────────────────────────────────────
# CONFIGURACIÓN DROIDCAM
# Cambia la IP por la que muestra la app DroidCam en tu celular
DROIDCAM_IP = "192.168.1.6"
DROIDCAM_PORT = 4747
DROIDCAM_URL = f"http://192.168.1.6:4747/video"
# ─────────────────────────────────────────

class DetectorAlimentos:
    def __init__(self):
        print("Cargando modelo YOLO-World...")
        self.model = YOLOWorld('yolov8m-world.pt')
        self.model.set_classes([
            "food", "fruit", "vegetable",
            "banana", "apple", "orange", "tomato",
            "lettuce", "chicken", "rice", "carrot",
            "broccoli", "strawberry", "meat", "cheese"
        ])
        print("Modelo listo.")

    def detectar(self, frame):
        results = self.model.predict(frame, conf=0.3, verbose=False)[0]

        alimentos = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            alimentos.append({
                'clase': results.names[int(box.cls[0])],
                'centro_px': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confianza': float(box.conf[0])
            })

        alimentos.sort(key=lambda x: x['confianza'], reverse=True)
        return alimentos

    def visualizar(self, frame, alimentos):
        for a in alimentos:
            x1, y1, x2, y2 = a['bbox']
            cx, cy = a['centro_px']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{a['clase']} {a['confianza']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        return frame


def conectar_droidcam(url, reintentos=5):
    print(f"Conectando a DroidCam: {url}")
    for intento in range(1, reintentos + 1):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"Conexión exitosa en intento {intento}.")
            return cap
        print(f"Intento {intento}/{reintentos} fallido, reintentando...")
    print("No se pudo conectar a DroidCam.")
    print("Verifica que:")
    print("  1. El celular y la PC estén en la misma red WiFi")
    print("  2. La IP en DROIDCAM_IP coincide con la que muestra la app")
    print("  3. La app DroidCam esté abierta y activa")
    return None


def main():
    cap = conectar_droidcam(DROIDCAM_URL)
    if cap is None:
        return

    detector = DetectorAlimentos()

    print("\nDetección iniciada. Presiona 'q' para salir.\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error al leer frame. Reconectando...")
            cap.release()
            cap = conectar_droidcam(DROIDCAM_URL)
            if cap is None:
                break
            continue

        alimentos = detector.detectar(frame)

        if alimentos:
            objetivo = alimentos[0]
            print(f"Objetivo → Clase: {objetivo['clase']:<15} "
                  f"Centro: {objetivo['centro_px']}  "
                  f"Confianza: {objetivo['confianza']:.2f}")
        else:
            print("Sin alimentos detectados...")

        frame_vis = detector.visualizar(frame.copy(), alimentos)

        # Info en pantalla
        cv2.putText(frame_vis, f"DroidCam: {DROIDCAM_IP}:{DROIDCAM_PORT}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame_vis, f"Detecciones: {len(alimentos)}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Brazo Robotico - Deteccion de Alimentos", frame_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saliendo...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()