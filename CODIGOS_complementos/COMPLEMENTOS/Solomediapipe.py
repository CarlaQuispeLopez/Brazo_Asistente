#!/usr/bin/env python
"""
safety_alarm_mediapipe_mesh.py — Alarma con malla facial completa
- Malla facial de 468 puntos
- Énfasis en barbilla, labios y ojos
- Seguridad con pitido
"""

import cv2
import numpy as np
import argparse
import time
import sys

import mediapipe as mp

# ============================================================
# CONFIGURACIÓN
# ============================================================
THRESHOLD_CM = 12.0
CAMERA_INDEX = 3

# ============================================================
# Inicializar MediaPipe
# ============================================================
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,      # IMPORTANTE: más puntos en labios y ojos
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================
# Colores para la malla (personalizables)
# ============================================================
COLOR_MESH = (80, 200, 80)      # Verde suave para la malla
COLOR_MOUTH = (0, 255, 255)     # Amarillo para boca
COLOR_CHIN = (255, 100, 0)      # Naranja para barbilla
COLOR_EYES = (255, 0, 0)        # Azul para ojos
COLOR_NOSE = (0, 255, 0)        # Verde para nariz
COLOR_DANGER = (0, 0, 255)      # Rojo para peligro
COLOR_SAFE = (0, 255, 0)        # Verde para seguro

# ============================================================
# Función para emitir pitido
# ============================================================
def beep():
    try:
        import winsound
        winsound.Beep(1000, 200)
    except:
        print("\a", end="", flush=True)

# ============================================================
# Función para dibujar la malla facial completa
# ============================================================
def draw_full_face_mesh(frame, landmarks):
    """Dibuja la malla facial completa de MediaPipe"""
    
    # 1. Dibujar todas las conexiones de la malla (líneas)
    mp_drawing.draw_landmarks(
        frame,
        landmarks,
        mp_face.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style()
    )
    
    # 2. Dibujar puntos clave con colores específicos
    h, w = frame.shape[:2]
    
    # Puntos de la boca (labios)
    upper_lip = landmarks.landmark[13]
    lower_lip = landmarks.landmark[14]
    mouth_left = landmarks.landmark[61]
    mouth_right = landmarks.landmark[291]
    
    mouth_center = (int((upper_lip.x + lower_lip.x) / 2 * w),
                    int((upper_lip.y + lower_lip.y) / 2 * h))
    
    # Dibujar contorno de labios más grueso
    lips_points = [13, 14, 61, 78, 308, 291]
    for idx in lips_points:
        px = int(landmarks.landmark[idx].x * w)
        py = int(landmarks.landmark[idx].y * h)
        cv2.circle(frame, (px, py), 3, COLOR_MOUTH, -1)
    
    # Barbilla (puntos 152, 172, 199)
    chin_points = [152, 172, 199, 200, 393]
    for idx in chin_points:
        px = int(landmarks.landmark[idx].x * w)
        py = int(landmarks.landmark[idx].y * h)
        cv2.circle(frame, (px, py), 4, COLOR_CHIN, -1)
    
    # Ojos (puntos alrededor)
    left_eye_points = [33, 133, 157, 158, 159, 160, 161, 173]
    right_eye_points = [362, 263, 387, 386, 385, 384, 398, 466]
    
    for idx in left_eye_points:
        px = int(landmarks.landmark[idx].x * w)
        py = int(landmarks.landmark[idx].y * h)
        cv2.circle(frame, (px, py), 2, COLOR_EYES, -1)
    
    for idx in right_eye_points:
        px = int(landmarks.landmark[idx].x * w)
        py = int(landmarks.landmark[idx].y * h)
        cv2.circle(frame, (px, py), 2, COLOR_EYES, -1)
    
    # Nariz
    nose_points = [1, 2, 4, 5, 6]
    for idx in nose_points:
        px = int(landmarks.landmark[idx].x * w)
        py = int(landmarks.landmark[idx].y * h)
        cv2.circle(frame, (px, py), 3, COLOR_NOSE, -1)
    
    return mouth_center

# ============================================================
# Función para estimar distancia (mejorada)
# ============================================================
def estimate_distance(landmarks, frame_shape):
    """Estima distancia usando ojos, nariz y barbilla"""
    h, w = frame_shape[:2]
    
    # Ojos
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]
    
    left_px = (int(left_eye.x * w), int(left_eye.y * h))
    right_px = (int(right_eye.x * w), int(right_eye.y * h))
    eye_distance = np.sqrt((right_px[0] - left_px[0])**2 +
                           (right_px[1] - left_px[1])**2)
    
    # Nariz a barbilla (para cuando está muy cerca)
    nose = landmarks.landmark[1]
    chin = landmarks.landmark[152]
    nose_px = (int(nose.x * w), int(nose.y * h))
    chin_px = (int(chin.x * w), int(chin.y * h))
    nose_chin_distance = np.sqrt((chin_px[0] - nose_px[0])**2 +
                                  (chin_px[1] - nose_px[1])**2)
    
    # Elegir la mejor métrica
    if eye_distance > 30:  # cara a distancia normal
        distance_cm = 5000 / eye_distance
    else:  # cara muy cerca
        distance_cm = 2500 / nose_chin_distance if nose_chin_distance > 0 else 100
    
    return min(50, max(5, distance_cm))

# ============================================================
# Main
# ============================================================
def main():
    global CAMERA_INDEX, THRESHOLD_CM
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=CAMERA_INDEX)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_CM)
    args = parser.parse_args()
    
    CAMERA_INDEX = args.cam
    THRESHOLD_CM = args.threshold
    
    print("=" * 55)
    print("  🔔 ALARMA CON MALLA FACIAL COMPLETA")
    print(f"  Cámara: {CAMERA_INDEX}  |  Umbral: {THRESHOLD_CM} cm")
    print("  - Malla facial de 468 puntos")
    print("  - Seguimiento de barbilla, boca y ojos")
    print("  ESC para salir")
    print("=" * 55)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir cámara {CAMERA_INDEX}")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    alarm_active = False
    last_beep_time = 0
    lost_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        face_detected = False
        distance_cm = None
        mouth_center = None
        
        if results.multi_face_landmarks:
            lost_frames = 0
            face_detected = True
            landmarks = results.multi_face_landmarks[0]
            
            # Dibujar malla facial completa
            mouth_center = draw_full_face_mesh(frame, landmarks)
            
            # Estimar distancia
            distance_cm = estimate_distance(landmarks, frame.shape)
            
            # Mostrar distancia
            color = COLOR_DANGER if distance_cm < THRESHOLD_CM else COLOR_SAFE
            cv2.putText(frame, f"Distancia: {distance_cm:.1f} cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Mostrar instrucción de seguridad
            cv2.putText(frame, "Mantener distancia > 12cm de la boca", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            lost_frames += 1
            cv2.putText(frame, "Buscando rostro...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            if lost_frames > 30:
                alarm_active = False
        
        # Alarma
        is_danger = face_detected and distance_cm is not None and distance_cm < THRESHOLD_CM
        
        if is_danger:
            if not alarm_active:
                alarm_active = True
                beep()
                print(f"\n🔴 ¡ALERTA! Distancia: {distance_cm:.1f} cm")
            
            if time.time() - last_beep_time > 1.5:
                last_beep_time = time.time()
                beep()
                print(f"⚠️ Distancia: {distance_cm:.1f} cm")
        else:
            if alarm_active:
                alarm_active = False
                print("✅ Zona segura")
        
        # HUD de alarma
        if alarm_active:
            cv2.putText(frame, "⚠️ ALARMA ACTIVA", (w//2 - 100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_DANGER, 2)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), COLOR_DANGER, 5)
        else:
            cv2.putText(frame, "SEGURO", (w//2 - 40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SAFE, 2)
        
        # Información en pantalla
        cv2.putText(frame, f"Umbral: {THRESHOLD_CM}cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "ESC = salir", (w - 90, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        cv2.imshow("Safety Alarm - Malla Facial", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Alarma finalizada.")

if __name__ == "__main__":
    main()