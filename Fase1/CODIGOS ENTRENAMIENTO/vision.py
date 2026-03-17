import cv2
import numpy as np
import threading
import time
import torch
import mediapipe as mp
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
from PIL import Image
from ultralytics import YOLOWorld
from transformers import pipeline as hf_pipeline

from config import (
    CAMERA_INDEX, FRAME_W, FRAME_H,
    YOLO_MODEL_PATH, YOLO_CONF_THR, YOLO_IOU_THR, FOOD_CLASSES,
    DEPTH_MODEL_NAME, DEPTH_MIN_CM, DEPTH_MAX_CM, DEPTH_EVERY_N_FRAMES,
    DEPTH_CALIB_RAW_POINTS, DEPTH_CALIB_CM_POINTS,
    REACHABLE_ZONE_PX, REACHABLE_DEPTH_MIN_CM, REACHABLE_DEPTH_MAX_CM,
    JOINT_LIMITS,
    DELIVERY_SHOULDER_STEPS, GRIPPER_STABILIZE_SECS,
    GRASP_DEPTH_LOSS_RATIO, MOUTH_PROXIMITY_CM,
    FACE_CONF_THRESHOLD, FACE_TRACK_THRESHOLD,
    MOUTH_UPPER_LIP_IDX, MOUTH_LOWER_LIP_IDX,
    DELIVERY_KP_BASE, DELIVERY_KP_HOMBRO, DELIVERY_KP_CODO,
    DELIVERY_DEAD_ZONE,
    DELIVERY_STEPS_BASE, DELIVERY_STEPS_HOMBRO, DELIVERY_STEPS_CODO,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Enumeración de fases
# ============================================================

class PipelinePhase(Enum):
    SEARCH   = auto()   # RL activo, busca alimento
    GRASP    = auto()   # RL activo, se acerca y agarra
    DELIVERY = auto()   # RL detenido, controlador P lleva a la boca


# ============================================================
# Estructuras de datos
# ============================================================

@dataclass
class FoodDetection:
    label:       str
    confidence:  float
    bbox:        Tuple[int, int, int, int]
    center_px:   Tuple[int, int]
    center_norm: Tuple[float, float]
    depth_cm:    float
    depth_norm:  float
    reachable:   bool = False


@dataclass
class MouthDetection:
    center_px:   Tuple[int, int]
    center_norm: Tuple[float, float]
    depth_cm:    float
    confidence:  float


@dataclass
class GraspEvidence:
    gripper_closed:   bool  = False
    depth_before_cm:  float = 0.0
    depth_after_cm:   float = 0.0
    yolo_lost_target: bool  = False
    close_timestamp:  float = 0.0
    confirmed:        bool  = False

    def evaluate(self) -> bool:
        if not self.gripper_closed:
            return False
        stabilized = (time.time() - self.close_timestamp) >= GRIPPER_STABILIZE_SECS
        if not stabilized:
            return False
        depth_increased = (
            self.depth_after_cm > self.depth_before_cm * (1.0 + GRASP_DEPTH_LOSS_RATIO)
        )
        self.confirmed = depth_increased or self.yolo_lost_target
        return self.confirmed


# ============================================================
# Calibración de profundidad
# ============================================================

class DepthCalibration:

    def __init__(self):
        raw = np.array(DEPTH_CALIB_RAW_POINTS, dtype=np.float32)
        cm  = np.array(DEPTH_CALIB_CM_POINTS,  dtype=np.float32)
        # garantizar orden ascendente para np.interp
        order      = np.argsort(raw)
        self._raw  = raw[order]
        self._cm   = cm[order]

    def raw_to_cm(self, raw_value: float) -> float:
        return float(np.interp(raw_value, self._raw, self._cm))

    def is_reachable_depth(self, depth_cm: float) -> bool:
        return REACHABLE_DEPTH_MIN_CM <= depth_cm <= REACHABLE_DEPTH_MAX_CM

    def in_reachable_zone_px(self, cx: int, cy: int) -> bool:
        x1, y1, x2, y2 = REACHABLE_ZONE_PX
        return x1 <= cx <= x2 and y1 <= cy <= y2


# ============================================================
# Controlador proporcional para la fase DELIVERY
# ============================================================

class DeliveryController:
    """
    Controlador proporcional simple que mueve el brazo hacia la boca.
    El RL no actúa durante esta fase; este controlador toma el mando.

    Lógica:
      - Compara la posición normalizada de la boca (cx, cy) con el centro
        del frame (0.5, 0.5) para calcular el error horizontal y vertical.
      - Compara la profundidad de la boca con MOUTH_PROXIMITY_CM para
        el error de profundidad (codo).
      - Cada error se convierte en una acción discreta de N pasos si supera
        la zona muerta.
    """

    def compute_actions(
        self,
        mouth: MouthDetection,
        robot_interface,
    ) -> list:
        actions = []

        error_x = mouth.center_norm[0] - 0.5
        error_y = mouth.center_norm[1] - 0.5
        error_d = mouth.depth_cm - MOUTH_PROXIMITY_CM

        if abs(error_x) > DELIVERY_DEAD_ZONE:
            steps = int(DELIVERY_KP_BASE * abs(error_x) * DELIVERY_STEPS_BASE)
            steps = max(steps, 10)
            direction = +steps if error_x > 0 else -steps
            robot_interface.move_joint("base", direction)
            actions.append("base+" if direction > 0 else "base-")

        if abs(error_y) > DELIVERY_DEAD_ZONE:
            steps = int(DELIVERY_KP_HOMBRO * abs(error_y) * DELIVERY_STEPS_HOMBRO)
            steps = max(steps, 10)
            direction = -steps if error_y > 0 else +steps
            robot_interface.move_joint("hombro", direction)
            actions.append("hombro+" if direction > 0 else "hombro-")

        if error_d > DELIVERY_DEAD_ZONE * DEPTH_MAX_CM:
            steps = int(DELIVERY_KP_CODO * (error_d / DEPTH_MAX_CM) * DELIVERY_STEPS_CODO)
            steps = max(steps, 10)
            robot_interface.move_joint("codo", +steps)
            actions.append("codo+")

        return actions


# ============================================================
# Pipeline principal
# ============================================================

class VisionPipeline:

    def __init__(self):
        self._phase      = PipelinePhase.SEARCH
        self._phase_lock = threading.Lock()

        self._calib          = DepthCalibration()
        self._grasp_evidence = GraspEvidence()
        self._delivery_ctrl  = DeliveryController()

        self._depth_map:      Optional[np.ndarray] = None
        self._depth_visual:   Optional[np.ndarray] = None
        self._depth_processing = False
        self._depth_lock       = threading.Lock()
        self._frame_n          = 0

        self._last_food:  Optional[FoodDetection]  = None
        self._last_mouth: Optional[MouthDetection] = None

        self._cap       = None
        self._yolo      = None
        self._depth_pipe = None
        self._mp_face   = None

    # ----------------------------------------------------------
    # Ciclo de vida
    # ----------------------------------------------------------

    def __enter__(self):
        self._init_camera()
        self._init_yolo()
        self._init_depth()
        self._init_mediapipe()
        return self

    def __exit__(self, *_):
        self.close()

    def _init_camera(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    def _init_yolo(self):
        self._yolo = YOLOWorld(YOLO_MODEL_PATH)
        self._yolo.set_classes(FOOD_CLASSES)

    def _init_depth(self):
        self._depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL_NAME,
            device=0 if DEVICE == "cuda" else -1,
        )

    def _init_mediapipe(self):
        self._mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=FACE_CONF_THRESHOLD,
            min_tracking_confidence=FACE_TRACK_THRESHOLD,
        )

    def read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    # ----------------------------------------------------------
    # Acceso a la fase actual
    # ----------------------------------------------------------

    @property
    def phase(self) -> PipelinePhase:
        with self._phase_lock:
            return self._phase

    @property
    def rl_should_act(self) -> bool:
        return self.phase in (PipelinePhase.SEARCH, PipelinePhase.GRASP)

    @property
    def delivery_active(self) -> bool:
        return self.phase == PipelinePhase.DELIVERY

    def _set_phase(self, new_phase: PipelinePhase):
        with self._phase_lock:
            if self._phase != new_phase:
                print(f"[Vision] Fase: {self._phase.name} → {new_phase.name}")
                self._phase = new_phase

    # ----------------------------------------------------------
    # Notificaciones externas desde collect_demos / train_rl
    # ----------------------------------------------------------

    def notify_gripper_closed(self, current_depth_cm: float):
        self._grasp_evidence.gripper_closed  = True
        self._grasp_evidence.depth_before_cm = current_depth_cm
        self._grasp_evidence.close_timestamp = time.time()

    def notify_gripper_opened(self):
        self._grasp_evidence = GraspEvidence()
        self._last_food      = None
        self._set_phase(PipelinePhase.SEARCH)

    # ----------------------------------------------------------
    # Método principal — llamado en cada iteración del bucle
    # ----------------------------------------------------------

    def get_state(
        self,
        frame:           np.ndarray,
        joint_positions: np.ndarray,
        robot_interface  = None,
    ) -> Tuple[np.ndarray, PipelinePhase, np.ndarray]:
        """
        Devuelve:
            state      — vector numpy (8,) para el agente RL
            phase      — fase actual del pipeline
            annotated  — frame con anotaciones para mostrar en pantalla
        """
        self._frame_n += 1
        h, w      = frame.shape[:2]
        annotated = frame.copy()
        phase     = self.phase

        # Depth Anything corre asíncrono solo en SEARCH y GRASP
        if (
            self._frame_n % DEPTH_EVERY_N_FRAMES == 0
            and not self._depth_processing
            and phase in (PipelinePhase.SEARCH, PipelinePhase.GRASP)
        ):
            self._launch_depth_async(frame)

        # SEARCH y GRASP: YOLO activo, RL activo
        if phase in (PipelinePhase.SEARCH, PipelinePhase.GRASP):
            food = self._run_yolo(frame, w, h)
            self._last_food = food
            self._update_grasp_depth_evidence(food)
            self._draw_yolo(annotated, food, w, h)
            self._evaluate_phase_transitions(food, robot_interface)

        # DELIVERY: MediaPipe activo, controlador P activo, RL detenido
        mouth = None
        if phase == PipelinePhase.DELIVERY:
            mouth = self._run_mediapipe(frame, w, h)
            self._last_mouth = mouth
            self._draw_mediapipe(annotated, mouth)
            if mouth is not None and robot_interface is not None:
                self._delivery_ctrl.compute_actions(mouth, robot_interface)
            if self.is_delivery_complete():
                if robot_interface is not None:
                    robot_interface.open_gripper()
                self.notify_gripper_opened()

        self._draw_phase_hud(annotated, phase, self._last_food, mouth)

        state = self.food_to_state_vector(self._last_food, joint_positions)
        return state, phase, annotated

    # ----------------------------------------------------------
    # YOLO
    # ----------------------------------------------------------

    def _run_yolo(
        self, frame: np.ndarray, w: int, h: int
    ) -> Optional[FoodDetection]:
        results = self._yolo.predict(
            frame, conf=YOLO_CONF_THR, iou=YOLO_IOU_THR, verbose=False
        )[0]

        candidates = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = self._yolo.names[cls] if cls < len(self._yolo.names) else "food"

            depth_raw = self._depth_at_bbox(x1, y1, x2, y2)
            depth_cm  = self._calib.raw_to_cm(depth_raw) if depth_raw is not None else DEPTH_MAX_CM
            depth_norm = float(np.clip(depth_cm / DEPTH_MAX_CM, 0.0, 1.0))

            reachable = (
                self._calib.is_reachable_depth(depth_cm)
                and self._calib.in_reachable_zone_px(cx, cy)
            )

            candidates.append(FoodDetection(
                label       = label,
                confidence  = conf,
                bbox        = (int(x1), int(y1), int(x2), int(y2)),
                center_px   = (cx, cy),
                center_norm = (cx / w, cy / h),
                depth_cm    = depth_cm,
                depth_norm  = depth_norm,
                reachable   = reachable,
            ))

        if not candidates:
            if self._grasp_evidence.gripper_closed:
                self._grasp_evidence.yolo_lost_target = True
            return None

        reachable_pool = [c for c in candidates if c.reachable]
        pool = reachable_pool if reachable_pool else candidates
        return min(pool, key=lambda c: c.depth_cm)

    # ----------------------------------------------------------
    # MediaPipe
    # ----------------------------------------------------------

    def _run_mediapipe(
        self, frame: np.ndarray, w: int, h: int
    ) -> Optional[MouthDetection]:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_face.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        ul = lm[MOUTH_UPPER_LIP_IDX]
        ll = lm[MOUTH_LOWER_LIP_IDX]

        mx_norm = (ul.x + ll.x) / 2.0
        my_norm = (ul.y + ll.y) / 2.0
        mx_px   = int(mx_norm * w)
        my_px   = int(my_norm * h)

        depth_raw  = self._depth_at_point(mx_px, my_px)
        depth_cm   = self._calib.raw_to_cm(depth_raw) if depth_raw is not None else DEPTH_MAX_CM

        vis = min(ul.visibility, ll.visibility) if hasattr(ul, "visibility") else 1.0

        return MouthDetection(
            center_px   = (mx_px, my_px),
            center_norm = (mx_norm, my_norm),
            depth_cm    = depth_cm,
            confidence  = float(vis),
        )

    # ----------------------------------------------------------
    # Depth Anything (asíncrono)
    # ----------------------------------------------------------

    def _launch_depth_async(self, frame: np.ndarray):
        self._depth_processing = True
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        threading.Thread(
            target=self._depth_worker, args=(rgb,), daemon=True
        ).start()

    def _depth_worker(self, rgb: np.ndarray):
        try:
            out    = self._depth_pipe(Image.fromarray(rgb))
            d      = np.array(out["depth"], dtype=np.float32)
            p_lo   = np.percentile(d, 2)
            p_hi   = np.percentile(d, 98)
            d_clip = np.clip(d, p_lo, p_hi)
            norm   = ((d_clip - p_lo) / (p_hi - p_lo + 1e-6) * 255).astype(np.uint8)
            visual = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            with self._depth_lock:
                self._depth_map    = d
                self._depth_visual = visual
        finally:
            self._depth_processing = False

    def _depth_at_bbox(self, x1, y1, x2, y2) -> Optional[float]:
        with self._depth_lock:
            d = self._depth_map
        if d is None:
            return None
        dh, dw = d.shape
        roi = d[max(0, int(y1)):min(dh, int(y2)), max(0, int(x1)):min(dw, int(x2))]
        return float(np.mean(roi)) if roi.size > 0 else None

    def _depth_at_point(self, cx: int, cy: int, radius: int = 10) -> Optional[float]:
        with self._depth_lock:
            d = self._depth_map
        if d is None:
            return None
        dh, dw = d.shape
        roi = d[
            max(0, cy - radius):min(dh, cy + radius),
            max(0, cx - radius):min(dw, cx + radius),
        ]
        return float(np.mean(roi)) if roi.size > 0 else None

    # ----------------------------------------------------------
    # Transiciones de fase
    # ----------------------------------------------------------

    def _update_grasp_depth_evidence(self, food: Optional[FoodDetection]):
        if food is None:
            return
        if self._grasp_evidence.gripper_closed and self._grasp_evidence.depth_before_cm > 0:
            self._grasp_evidence.depth_after_cm = food.depth_cm

    def _evaluate_phase_transitions(self, food: Optional[FoodDetection], robot_interface):
        phase = self.phase

        # SEARCH → GRASP: hay comida alcanzable
        if phase == PipelinePhase.SEARCH:
            if food is not None and food.reachable:
                self._set_phase(PipelinePhase.GRASP)
            return

        # GRASP → SEARCH: perdimos la comida sin haber cerrado el gripper
        if phase == PipelinePhase.GRASP:
            if food is None and not self._grasp_evidence.gripper_closed:
                self._set_phase(PipelinePhase.SEARCH)
                return

            # GRASP → DELIVERY: las 3 condiciones deben cumplirse
            if not self._grasp_evidence.gripper_closed:
                return

            grasp_confirmed = self._grasp_evidence.evaluate()
            if not grasp_confirmed:
                return

            if robot_interface is None:
                return

            shoulder_steps   = robot_interface.get_raw_positions().get("hombro", 0)
            shoulder_ok      = shoulder_steps >= DELIVERY_SHOULDER_STEPS
            time_ok          = (
                time.time() - self._grasp_evidence.close_timestamp
            ) >= GRIPPER_STABILIZE_SECS

            if shoulder_ok and time_ok:
                self._set_phase(PipelinePhase.DELIVERY)

    # ----------------------------------------------------------
    # Interfaz pública para el agente RL y collect_demos
    # ----------------------------------------------------------

    def food_to_state_vector(
        self,
        food:            Optional[FoodDetection],
        joint_positions: np.ndarray,
    ) -> np.ndarray:
        if food is not None:
            cx_norm    = food.center_norm[0]
            cy_norm    = food.center_norm[1]
            depth_norm = food.depth_norm
        else:
            cx_norm    = 0.5
            cy_norm    = 0.5
            depth_norm = 1.0

        return np.array([
            cx_norm,
            cy_norm,
            depth_norm,
            joint_positions[0],
            joint_positions[1],
            joint_positions[2],
            joint_positions[3],
            joint_positions[4],
        ], dtype=np.float32)

    def get_current_food_depth(self) -> float:
        if self._last_food is not None:
            return self._last_food.depth_cm
        return DEPTH_MAX_CM

    def get_mouth_guidance(self) -> Optional[Tuple[float, float, float]]:
        if self._last_mouth is None:
            return None
        return (
            self._last_mouth.center_norm[0],
            self._last_mouth.center_norm[1],
            self._last_mouth.depth_cm,
        )

    def is_delivery_complete(self) -> bool:
        if self._last_mouth is None:
            return False
        return self._last_mouth.depth_cm <= MOUTH_PROXIMITY_CM

    # ----------------------------------------------------------
    # Visualización
    # ----------------------------------------------------------

    def _draw_yolo(self, frame: np.ndarray, food: Optional[FoodDetection], w: int, h: int):
        with self._depth_lock:
            d_vis = self._depth_visual
        if d_vis is not None:
            frame[:] = cv2.addWeighted(frame, 0.6, cv2.resize(d_vis, (w, h)), 0.4, 0)

        rx1, ry1, rx2, ry2 = REACHABLE_ZONE_PX
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (80, 80, 80), 1)
        cv2.putText(frame, "zona alcanzable", (rx1, ry1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        if food is None:
            return

        bx1, by1, bx2, by2 = food.bbox
        color = (0, 255, 0) if food.reachable else (0, 140, 255)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.circle(frame, food.center_px, 6, (0, 0, 255), -1)

        tag = f"{food.label} {food.confidence:.2f} {food.depth_cm:.1f}cm"
        if food.reachable:
            tag += " [OK]"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (bx1, by1 - th - 8), (bx1 + tw + 4, by1), color, -1)
        cv2.putText(frame, tag, (bx1 + 2, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_mediapipe(self, frame: np.ndarray, mouth: Optional[MouthDetection]):
        if mouth is None:
            cv2.putText(frame, "Buscando rostro...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            return

        cx, cy = mouth.center_px
        cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy),  3, (0, 255, 255), -1)

        tag = f"boca {mouth.depth_cm:.1f}cm"
        cv2.putText(frame, tag, (cx + 14, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        err_x = mouth.center_norm[0] - 0.5
        err_y = mouth.center_norm[1] - 0.5
        h, w  = frame.shape[:2]
        cv2.arrowedLine(frame, (w // 2, h // 2),
                        (w // 2 + int(err_x * 120), h // 2 + int(err_y * 120)),
                        (0, 255, 255), 2, tipLength=0.3)

        if self.is_delivery_complete():
            cv2.putText(frame, "SOLTAR GRIPPER",
                        (cx - 70, cy - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

    def _draw_phase_hud(self, frame, phase, food, mouth):
        h, w = frame.shape[:2]
        phase_color = {
            PipelinePhase.SEARCH:   (200, 200, 0),
            PipelinePhase.GRASP:    (0, 200, 255),
            PipelinePhase.DELIVERY: (0, 255, 0),
        }[phase]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 75), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"FASE: {phase.name}",
                    (8, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_color, 2)

        rl_str = "RL ACTIVO" if self.rl_should_act else "RL DETENIDO"
        rl_color = (0, 255, 0) if self.rl_should_act else (0, 0, 200)
        cv2.putText(frame, rl_str,
                    (w - 160, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rl_color, 2)

        if food and phase in (PipelinePhase.SEARCH, PipelinePhase.GRASP):
            ev = self._grasp_evidence
            grip_str = "CERRADO" if ev.gripper_closed else "ABIERTO"
            conf_str = "CONFIRMADO" if ev.confirmed   else "pendiente"
            cv2.putText(
                frame,
                f"{food.label}  d={food.depth_cm:.1f}cm  gripper={grip_str} ({conf_str})",
                (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
            )

        if mouth and phase == PipelinePhase.DELIVERY:
            cv2.putText(
                frame,
                f"boca d={mouth.depth_cm:.1f}cm  err_x={mouth.center_norm[0]-0.5:+.2f}",
                (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1,
            )

        cv2.putText(frame, f"frame {self._frame_n}",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # ----------------------------------------------------------
    # Cierre
    # ----------------------------------------------------------

    def close(self):
        if self._cap is not None:
            self._cap.release()
        if self._mp_face is not None:
            self._mp_face.close()
        cv2.destroyAllWindows()