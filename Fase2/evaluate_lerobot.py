#!/usr/bin/env python
"""
evaluate_lerobot.py — Evaluar modelo fine-tuned en tu brazo real
"""

import time
import argparse
import numpy as np
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from nutribot_robot import NutriBotRobot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/lerobot_finetuned")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    # ========================================================
    # 1. Cargar modelo fine-tuned
    # ========================================================
    print(f"[Eval] Cargando modelo desde: {args.model_path}")
    policy = DiffusionPolicy.from_pretrained(args.model_path)
    policy.eval()
    if torch.cuda.is_available():
        policy.cuda()

    # ========================================================
    # 2. Conectar al robot
    # ========================================================
    robot = NutriBotRobot(simulate=args.sim)
    print("[Eval] Robot conectado")

    # ========================================================
    # 3. Evaluar episodios
    # ========================================================
    successes = 0
    for ep in range(1, args.episodes + 1):
        print(f"\n[Eval] Episodio {ep}/{args.episodes}")
        
        # Reset
        robot.robot.home()
        robot.robot.open_gripper()
        robot._gripper_closed = False
        time.sleep(1)

        # Bucle del episodio
        done = False
        step = 0
        total_reward = 0

        while not done and step < 200:  # max steps
            # Obtener estado
            state = robot.read_state()
            obs_tensor = torch.FloatTensor(state["observation.state"]).cuda().unsqueeze(0)

            # Policy decide acción
            with torch.no_grad():
                action = policy.select_action(obs_tensor)
            
            # Ejecutar acción
            robot.send_action(action.cpu().numpy()[0])
            
            # Esperar movimiento
            time.sleep(0.1)
            
            # Verificar agarre (simplificado)
            if robot._gripper_closed:
                done = True
                successes += 1
                print(f"  ✓ Agarre exitoso en paso {step}")
            
            step += 1
            time.sleep(0.05)

    print(f"\n[Eval] Tasa de éxito: {successes}/{args.episodes} ({successes/args.episodes*100:.1f}%)")
    robot.disconnect()


if __name__ == "__main__":
    main()