"""
vision.py — Pipeline de visión para NutriBot (modo snapshot)
v2: la cámara toma UNA foto desde HOME y de ahí se trabaja.
    No hay tracking en tiempo real durante el movimiento.
    No hay MediaPipe (detección de boca).

Flujo:
    1. Brazo va a HOME (cámara ve el plato bien)
    2. take_snapshot()       → captura el frame
    3. detect_food(frame)    → YOLO + Depth Anything en esa imagen
    4. La posición del alimento (cx, cy, depth_norm) queda fija durante el episodio
    5. El agente RL mueve el brazo usando esa referencia + posiciones articulares
"""

import cv2
import numpy as np
import threading
import time
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image
from ultralytics import YOLOWorld
from transformers import pipeline as hf_pipeline

from config import (
    CAMERA_INDEX, FRAME_W, FRAME_H,
    YOLO_MODEL_PATH, YOLO_CONF_THR, YOLO_IOU_THR, FOOD_CLASSES,
    DEPTH_MODEL_NAME,
    DEPTH_RAW_MIN, DEPTH_RAW_MAX,
    STATE_DIM,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Estructura de datos del alimento detectado
# ============================================================

@dataclass
class FoodDetection:
    label:       str
    confidence:  float
    bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2) en píxeles
    center_px:   Tuple[int, int]             # (cx, cy) en píxeles
    center_norm: Tuple[float, float]         # (cx/W, cy/H) en [0, 1]
    depth_raw:   float                       # valor raw de Depth Anything
    depth_norm:  float                       # normalizado a [0, 1] (mayor = más cerca)


# ============================================================
# Pipeline de visión
# ============================================================

class VisionPipeline:
    """
    Pipeline de visión simplificado para NutriBot.

    Uso:
        with VisionPipeline() as vision:
            frame    = vision.take_snapshot()
            food     = vision.detect_food(frame)
            annotated = vision.draw_detection(frame, food)
            cv2.imshow("Plato", annotated)
    """

    def __init__(self):
        self._cap        = None
        self._yolo       = None
        self._depth_pipe = None

    # ----------------------------------------------------------
    # Ciclo de vida
    # ----------------------------------------------------------

    def __enter__(self):
        self._init_camera()
        self._init_yolo()
        self._init_depth()
        return self

    def __exit__(self, *_):
        self.close()

    def _init_camera(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        print(f"[Vision] Cámara inicializada (índice {CAMERA_INDEX})")

    def _init_yolo(self):
        self._yolo = YOLOWorld(YOLO_MODEL_PATH)
        self._yolo.set_classes(FOOD_CLASSES)
        print(f"[Vision] YOLO cargado: {YOLO_MODEL_PATH}")

    def _init_depth(self):
        self._depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL_NAME,
            device=0 if DEVICE == "cuda" else -1,
        )
        print(f"[Vision] Depth Anything cargado ({DEVICE})")

    def close(self):
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()

    # ----------------------------------------------------------
    # Captura de frame
    # ----------------------------------------------------------

    def read_frame(self) -> Optional[np.ndarray]:
        """Lee un frame de la cámara."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def take_snapshot(self, n_warmup: int = 5) -> Optional[np.ndarray]:
        """
        Captura el frame usado para detección.
        Descarta los primeros n_warmup frames para estabilizar la exposición.
        """
        for _ in range(n_warmup):
            self._cap.read()
        ret, frame = self._cap.read()
        if not ret:
            print("[Vision] ERROR: No se pudo capturar frame.")
            return None
        print("[Vision] Snapshot capturado.")
        return frame

    # ----------------------------------------------------------
    # Depth Anything (síncrono — se corre solo sobre el snapshot)
    # ----------------------------------------------------------

    def _run_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Ejecuta Depth Anything sobre un frame BGR.
        Retorna mapa de profundidad raw (float32, misma resolución que el frame).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = self._depth_pipe(Image.fromarray(rgb))
        d   = np.array(out["depth"], dtype=np.float32)
        # Redimensionar al tamaño del frame si es necesario
        h, w = frame.shape[:2]
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        return d

    def _depth_at_bbox(
        self,
        depth_map: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
    ) -> Optional[float]:
        """Profundidad media en la región del bounding box."""
        h, w = depth_map.shape
        roi = depth_map[
            max(0, int(y1)):min(h, int(y2)),
            max(0, int(x1)):min(w, int(x2)),
        ]
        return float(np.mean(roi)) if roi.size > 0 else None

    def _normalize_depth(
        self,
        depth_map:  np.ndarray,
        raw_value:  float,
    ) -> float:
        """
        Normaliza el valor raw de profundidad al rango [0, 1]
        usando los percentiles del mapa completo del frame (normalización por escena).

        Mayor depth_norm → objeto más cercano (o más en primer plano).
        """
        p_lo = float(np.percentile(depth_map, 2))
        p_hi = float(np.percentile(depth_map, 98))
        norm = (raw_value - p_lo) / (p_hi - p_lo + 1e-6)
        return float(np.clip(norm, 0.0, 1.0))

    # ----------------------------------------------------------
    # Detección de alimentos
    # ----------------------------------------------------------

    def detect_food(self, frame: np.ndarray) -> Optional[FoodDetection]:
        """
        Detecta el alimento objetivo en el frame.

        Proceso:
          1. Depth Anything → mapa de profundidad del frame
          2. YOLO → bounding boxes de alimentos
          3. Para cada candidato: calcular profundidad en su bbox
          4. Retornar el candidato con mayor depth_norm (más cercano al gripper)

        Retorna None si no se detecta ningún alimento.
        """
        h, w = frame.shape[:2]

        # 1. Mapa de profundidad (síncrono, solo una vez por snapshot)
        depth_map = self._run_depth(frame)
        if depth_map is None:
            print("[Vision] ERROR: Depth Anything falló.")
            return None

        # 2. YOLO
        results = self._yolo.predict(
            frame, conf=YOLO_CONF_THR, iou=YOLO_IOU_THR, verbose=False
        )[0]

        if len(results.boxes) == 0:
            print("[Vision] No se detectaron alimentos en el frame.")
            return None

        # 3. Construir candidatos
        candidates = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = self._yolo.names[cls] if cls < len(self._yolo.names) else "food"

            depth_raw = self._depth_at_bbox(depth_map, x1, y1, x2, y2)
            if depth_raw is None:
                depth_raw = float(np.mean(depth_map))

            depth_norm = self._normalize_depth(depth_map, depth_raw)

            candidates.append(FoodDetection(
                label       = label,
                confidence  = conf,
                bbox        = (int(x1), int(y1), int(x2), int(y2)),
                center_px   = (cx, cy),
                center_norm = (cx / w, cy / h),
                depth_raw   = depth_raw,
                depth_norm  = depth_norm,
            ))

        if not candidates:
            return None

        # 4. Elegir el de mayor depth_norm (más cercano / más prominente en escena)
        best = max(candidates, key=lambda c: c.depth_norm)
        print(
            f"[Vision] Alimento detectado: {best.label}  "
            f"conf={best.confidence:.2f}  "
            f"cx={best.center_norm[0]:.3f}  cy={best.center_norm[1]:.3f}  "
            f"depth_norm={best.depth_norm:.3f}"
        )
        return best

    # ----------------------------------------------------------
    # Conversión a vector de estado para el agente RL
    # ----------------------------------------------------------

    def food_to_state_vector(
        self,
        food:            Optional[FoodDetection],
        joint_positions: np.ndarray,   # shape (4,): base, hombro, codo, gripper
        pinza_closed:    bool = False,
    ) -> np.ndarray:
        """
        Construye el vector de estado de 8 dimensiones:
            [cx_norm, cy_norm, depth_norm,
             j_base, j_hombro, j_codo, j_gripper,
             pinza_state]

        Si food es None (no se detectó nada): usa valores por defecto.
        """
        if food is not None:
            cx_norm    = food.center_norm[0]
            cy_norm    = food.center_norm[1]
            depth_norm = food.depth_norm
        else:
            cx_norm    = 0.5
            cy_norm    = 0.5
            depth_norm = 0.5   # profundidad desconocida

        pinza_state = 1.0 if pinza_closed else 0.0

        return np.array([
            cx_norm,
            cy_norm,
            depth_norm,
            joint_positions[0],   # base
            joint_positions[1],   # hombro
            joint_positions[2],   # codo
            joint_positions[3],   # gripper (sube/baja)
            pinza_state,
        ], dtype=np.float32)

    # ----------------------------------------------------------
    # Visualización
    # ----------------------------------------------------------

    def draw_detection(
        self,
        frame: np.ndarray,
        food:  Optional[FoodDetection],
    ) -> np.ndarray:
        """Dibuja la detección sobre el frame y retorna el frame anotado."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if food is None:
            cv2.putText(
                annotated, "SIN DETECCION",
                (w // 2 - 100, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )
            return annotated

        bx1, by1, bx2, by2 = food.bbox
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.circle(annotated, food.center_px, 6, (0, 0, 255), -1)

        tag = (
            f"{food.label} {food.confidence:.2f} | "
            f"cx={food.center_norm[0]:.3f} cy={food.center_norm[1]:.3f} | "
            f"depth={food.depth_norm:.3f}"
        )
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (bx1, by1 - th - 8), (bx1 + tw + 4, by1), (0, 255, 0), -1)
        cv2.putText(annotated, tag, (bx1 + 2, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # HUD inferior con el estado
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        cv2.putText(
            annotated,
            f"SNAPSHOT  alimento: {food.label}  depth_norm={food.depth_norm:.3f}",
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1,
        )

        return annotated

    def show_snapshot_interactive(
        self,
        frame: np.ndarray,
        food:  Optional[FoodDetection],
        window_name: str = "Snapshot - plato",
    ) -> bool:
        """
        Muestra el snapshot con la detección.
        Retorna True si el operador presiona ENTER para confirmar,
        False si presiona ESC para retomar la foto.
        """
        annotated = self.draw_detection(frame, food)
        h, w = annotated.shape[:2]

        msg = "ENTER=confirmar  ESC=repetir foto  Q=salir"
        cv2.putText(annotated, msg, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)

        if key == 13:   # ENTER
            return True
        if key == ord('q'):
            raise KeyboardInterrupt("Operador salió.")
        return False   # ESC → repetir