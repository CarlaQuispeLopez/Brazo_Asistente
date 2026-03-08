from ultralytics import YOLOWorld
import cv2

# ─────────────────────────────────────────
# CONFIGURACIÓN CÁMARA
# 0 = HD Webcam, 1 = Venus USB2.0 Camera
CAMARA_INDEX = 2
# Si no funciona con 1, prueba con 0
# ─────────────────────────────────────────

COLORES = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255),
    (255,255,0),(100,255,100),(255,150,50),(50,200,255),(200,50,255),
    (150,0,255),(0,200,150),(255,50,150),(100,100,255),(50,255,200)
]

class DetectorAlimentos:
    def __init__(self):
        print("Cargando modelo YOLO-World...")
        self.model = YOLOWorld('yolov8m-world.pt')

        self.model.set_classes([
            # ── FRUTAS ──
            "apple", "pear", "peach", "plum", "apricot", "cherry",
            "strawberry", "raspberry", "blueberry", "blackberry",
            "grape", "watermelon", "melon", "cantaloupe", "honeydew",
            "banana", "mango", "papaya", "pineapple chunk", "kiwi",
            "orange slice", "mandarin", "lemon slice", "lime slice",
            "fig", "date", "lychee", "guava", "passion fruit",
            "dragon fruit", "star fruit", "persimmon", "pomegranate seed",

            # ── VERDURAS Y HORTALIZAS ──
            "tomato", "cherry tomato", "carrot piece", "broccoli floret",
            "cauliflower", "lettuce piece", "cucumber slice", "zucchini",
            "eggplant", "bell pepper", "corn kernel", "pea",
            "green bean", "asparagus", "artichoke", "celery piece",
            "beet", "radish", "turnip", "potato chunk", "sweet potato",
            "mushroom", "onion piece", "leek", "spinach", "kale",
            "cabbage piece", "brussels sprout", "bok choy",

            # ── PROTEÍNAS Y CARNES ──
            "chicken piece", "beef piece", "pork piece", "lamb piece",
            "turkey piece", "sausage slice", "meatball", "nugget",
            "shrimp", "fish piece", "salmon chunk", "tuna piece",
            "squid piece", "octopus piece", "crab meat",
            "boiled egg", "fried egg piece", "omelette piece",
            "tofu cube", "tempeh piece",

            # ── CARBOHIDRATOS Y OTROS ──
            "pasta piece", "noodle", "gnocchi", "dumpling",
            "rice ball", "bread piece", "crouton",
            "cheese cube", "mozzarella", "ham piece",
            "olive", "pickle slice", "sun-dried tomato",
            "chickpea", "lentil", "bean",

            # ── GENÉRICOS (fallback) ──
            "food piece", "fruit piece", "vegetable piece", "meat piece"
        ])
        print(f"Modelo listo. {len(self.model.names)} clases cargadas.\n")

    def detectar(self, frame):
        results = self.model.predict(
            frame,
            conf=0.35,
            iou=0.3,
            verbose=False
        )[0]

        alimentos = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            clase = results.names[int(box.cls[0])]
            alimentos.append({
                'clase': clase,
                'centro_px': (int((x1+x2)/2), int((y1+y2)/2)),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confianza': float(box.conf[0]),
                'color': COLORES[int(box.cls[0]) % len(COLORES)]
            })

        alimentos.sort(key=lambda x: x['confianza'], reverse=True)
        return alimentos

    def visualizar(self, frame, alimentos):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (380, 40), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"Venus USB Cam  |  Alimentos: {len(alimentos)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        for a in alimentos:
            x1, y1, x2, y2 = a['bbox']
            cx, cy = a['centro_px']
            color = a['color']

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)
            cv2.circle(frame, (cx, cy), 6, (255,255,255), 1)

            label = f"{a['clase']} {a['confianza']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        # Resaltar objetivo principal
        if alimentos:
            obj = alimentos[0]
            x1,y1,x2,y2 = obj['bbox']
            cv2.rectangle(frame, (x1-3,y1-3), (x2+3,y2+3), (0,255,255), 3)
            cv2.putText(frame, "OBJETIVO", (x1, y2+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

        return frame


def main():
    print(f"Abriendo camara Venus USB (indice {CAMARA_INDEX})...")
    cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW = más estable en Windows

    if not cap.isOpened():
        print(f"Error: no se pudo abrir camara {CAMARA_INDEX}.")
        print("Prueba cambiando CAMARA_INDEX a 0.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camara abierta correctamente.\n")

    detector = DetectorAlimentos()
    print("Deteccion iniciada. Presiona 'q' para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame.")
            break

        alimentos = detector.detectar(frame)

        if alimentos:
            obj = alimentos[0]
            print(f"OBJETIVO → {obj['clase']:<25} "
                  f"Centro: {obj['centro_px']}   "
                  f"Conf: {obj['confianza']:.2f}")

        frame_vis = detector.visualizar(frame.copy(), alimentos)
        cv2.imshow("Brazo Robotico - Deteccion de Alimentos", frame_vis)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Saliendo...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()