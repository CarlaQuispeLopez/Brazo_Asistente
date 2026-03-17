#!/usr/bin/env python
"""
convert_to_lerobot.py — Convierte demonstrations.pkl a LeRobotDataset

USO:
    python convert_to_lerobot.py
    python convert_to_lerobot.py --demo_file mis_demos.pkl --output mis_datos
"""

import pickle
import numpy as np
import argparse
from pathlib import Path
import torch
from datasets import Dataset, Features, Value, Sequence, Array2D
import cv2
import json

from config import DEMO_FILE, STATE_DIM, ACTION_DIM

# ============================================================
# Configuración
# ============================================================

HUGGINGFACE_REPO = "tu-usuario/nutribot-demos"  # cámbialo
FRAME_H, FRAME_W = 480, 640

# ============================================================
# Función principal
# ============================================================

def convert_to_lerobot(demo_file: str, output_dir: str):
    print(f"[Convert] Cargando demos desde: {demo_file}")
    with open(demo_file, "rb") as f:
        data = pickle.load(f)

    episodes = data.get("episodes", [])
    print(f"[Convert] {len(episodes)} episodios encontrados")

    # Estructura para LeRobot
    frames = []
    actions = []
    episode_ends = []
    task_ids = []
    states = []

    total_steps = 0
    for ep_idx, ep in enumerate(episodes):
        if not ep.get("success", False):
            print(f"[Convert] Episodio {ep_idx} no exitoso, omitiendo")
            continue

        steps = ep["steps"]
        episode_ends.append(total_steps + len(steps))
        task_ids.append(0)  # solo una tarea por ahora

        for step in steps:
            # Estado (8 dims)
            state = step["state"]
            states.append(state)

            # Acción (11 dims, one-hot para discreta)
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[step["action"]] = 1.0
            actions.append(action)

            # Frame dummy (LeRobot espera imágenes, pero usaremos el estado)
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            frames.append(frame)

            total_steps += 1

    print(f"[Convert] Total pasos: {total_steps}")

    # Crear dataset de Hugging Face
    dataset = Dataset.from_dict({
        "frame": frames,
        "observation.state": np.array(states, dtype=np.float32),
        "action": np.array(actions, dtype=np.float32),
        "episode_index": [i for i, end in enumerate(episode_ends) for _ in range(end - (episode_ends[i-1] if i>0 else 0))],
    })

    # Guardar
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    # Guardar metadata
    metadata = {
        "total_episodes": len(episode_ends),
        "total_steps": total_steps,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "fps": 30,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Convert] Dataset guardado en: {output_dir}")
    print(f"[Convert] Para subir a Hugging Face:")
    print(f"    dataset.push_to_hub('{HUGGINGFACE_REPO}')")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", type=str, default=DEMO_FILE)
    parser.add_argument("--output", type=str, default="lerobot_dataset")
    args = parser.parse_args()

    convert_to_lerobot(args.demo_file, args.output)