#!/usr/bin/env python
"""
finetune_lerobot.py — Fine-tuning de un modelo LeRobot con tus datos

USO:
    python finetune_lerobot.py
    python finetune_lerobot.py --dataset ./lerobot_dataset --repo tu-usuario/nutribot-demos
"""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.scripts.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./lerobot_dataset")
    parser.add_argument("--repo_id", type=str, default="tu-usuario/nutribot-demos")
    parser.add_argument("--policy", type=str, default="diffusion")
    parser.add_argument("--batch_size", type=int, default=16)  # para RTX 3070
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save_path", type=str, default="./models/lerobot_finetuned")
    args = parser.parse_args()

    # ========================================================
    # 1. Cargar tu dataset convertido
    # ========================================================
    print(f"[Finetune] Cargando dataset desde: {args.dataset_path}")
    hf_dataset = load_from_disk(args.dataset_path)
    
    # Crear dataset en formato LeRobot
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        split="train",
        fps=30,
        features=hf_dataset.features,
        tolerance_s=0.1,
    )
    
    # Agregar tus datos
    for i in range(len(hf_dataset)):
        item = hf_dataset[i]
        dataset.add_frame(item)
    dataset.finalize()

    print(f"[Finetune] Dataset listo: {len(dataset)} frames")

    # ========================================================
    # 2. Configurar modelo (Diffusion Policy)
    # ========================================================
    config = DiffusionConfig(
        # Dimensiones de entrada/salida
        input_dim=8,           # STATE_DIM
        output_dim=11,         # ACTION_DIM (discreta)
        
        # Arquitectura (chica para RTX 3070)
        n_obs_steps=1,
        n_action_steps=1,
        horizon=4,
        n_diffusion_steps=100,
        n_train_steps=args.steps,
        batch_size=args.batch_size,
        
        # Optimización
        learning_rate=1e-4,
        lr_scheduler="cosine",
        lr_warmup_steps=500,
        adam_weight_decay=1e-6,
        
        # Dataset
        dataset_repo_id=args.repo_id,
        dataset_fps=30,
    )

    policy = DiffusionPolicy(config)

    # ========================================================
    # 3. Entrenamiento
    # ========================================================
    print("\n[Finetune] Iniciando entrenamiento...")
    train(
        policy=policy,
        dataset=dataset,
        output_dir=Path(args.save_path),
        num_train_steps=args.steps,
        gradient_accumulation_steps=1,
        log_freq=100,
        save_freq=1000,
        eval_freq=1000,
        wandb_enabled=False,  # True si quieres logging
    )

    print(f"\n[Finetune] Modelo guardado en: {args.save_path}")
    print("[Finetune] Para evaluar: python evaluate_lerobot.py --model ./models/lerobot_finetuned")

if __name__ == "__main__":
    main()