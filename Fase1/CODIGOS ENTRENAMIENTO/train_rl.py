"""
train_rl.py — Aprendizaje por Refuerzo Profundo (PPO) — NutriBot
v2: snapshot al inicio del episodio, éxito por lift-test, sin DELIVERY.

USO:
    python train_rl.py
    python train_rl.py --sim
    python train_rl.py --steps 300000
    python train_rl.py --no_bc_init
    python train_rl.py --resume models/ppo_policy/ppo_brazo_50000_steps.zip
    python train_rl.py --run    models/ppo_policy/ppo_final.zip

LÓGICA DEL ÉXITO:
    El agente debe posicionar el gripper sobre el alimento y cerrar la pinza.
    Tras cerrar (acción 8):
        - Se espera GRIPPER_STABILIZE_SECS
        - Se intenta levantar el hombro LIFT_SUCCESS_STEPS pasos
        - Si el lift tiene éxito → episodio termina con reward +10
        - Si el agarre fue "prematuro" (food_depth_norm bajo) → penalización
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    raise ImportError("Instala: pip install stable-baselines3")

from config import (
    RL_MODEL_PATH, RL_TOTAL_STEPS, RL_LEARNING_RATE,
    RL_N_STEPS, RL_BATCH_SIZE, RL_N_EPOCHS,
    RL_GAMMA, RL_ENT_COEF,
    STATE_DIM, ACTION_DIM, ACTION_STEPS,
    MAX_EPISODE_STEPS,
    REWARD_GRASP_SUCCESS, REWARD_STEP_PENALTY,
    REWARD_DIST_SCALE, REWARD_COLLISION,
    BC_MODEL_PATH, BC_HIDDEN_DIM,
    GRIPPER_STABILIZE_SECS, LIFT_SUCCESS_STEPS,
    GRASP_MIN_DEPTH_NORM,
)
from robot_interface import RobotInterface
from vision import VisionPipeline
from train_bc import PolicyNetwork, load_bc_model


# ============================================================
# Entorno Gymnasium
# ============================================================

class RoboticArmEnv(gym.Env):
    """
    Entorno NutriBot para PPO.

    Reset:
        - Brazo va a HOME
        - Se toma snapshot → se detecta el alimento
        - La posición del alimento queda fija durante el episodio

    Step:
        - El agente ejecuta acciones (0-8)
        - La observación cambia solo en la parte articular
        - Acción 8 (cerrar pinza) → lift-test → éxito/fallo

    Observación (STATE_DIM=8):
        [cx_norm, cy_norm, depth_norm, j_base, j_hombro, j_codo, j_gripper, pinza_state]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        robot:       RobotInterface,
        vision:      VisionPipeline,
        render_mode: str = "human",
        max_steps:   int = MAX_EPISODE_STEPS,
    ):
        super().__init__()
        self.robot       = robot
        self.vision      = vision
        self.render_mode = render_mode
        self.max_steps   = max_steps

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        # Estado fijo del alimento para el episodio
        self._food_cx:    float = 0.5
        self._food_cy:    float = 0.5
        self._food_depth: float = 0.5

        self._step_count:     int   = 0
        self._episode_reward: float = 0.0
        self._grasp_attempted: bool = False

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count      = 0
        self._episode_reward  = 0.0
        self._grasp_attempted = False

        # Abrir pinza y volver a HOME
        self.robot.open_gripper()
        self.robot.home()
        time.sleep(0.5)

        # Tomar snapshot y detectar alimento
        frame = self.vision.take_snapshot()
        food  = self.vision.detect_food(frame) if frame is not None else None

        if food is not None:
            self._food_cx    = food.center_norm[0]
            self._food_cy    = food.center_norm[1]
            self._food_depth = food.depth_norm
        else:
            print("[Env] ADVERTENCIA: No se detectó alimento. Usando centro.")
            self._food_cx    = 0.5
            self._food_cy    = 0.5
            self._food_depth = 0.5

        obs = self._get_obs()
        return obs, {}

    # ----------------------------------------------------------
    # Step
    # ----------------------------------------------------------

    def step(self, action: int):
        self._step_count += 1
        reward     = REWARD_STEP_PENALTY
        terminated = False
        truncated  = False
        info       = {}

        # ── Acción 8: cerrar pinza + lift-test ──────────────────
        if action == 8:
            self._grasp_attempted = True

            # Penalizar si el alimento está lejos (agarre prematuro)
            if self._food_depth < GRASP_MIN_DEPTH_NORM:
                reward += REWARD_COLLISION * 0.5
                info["premature_grasp"] = True
                print(f"  [Env] Agarre prematuro (depth_norm={self._food_depth:.3f})")
            else:
                # Cerrar pinza
                self.robot.close_gripper()
                time.sleep(GRIPPER_STABILIZE_SECS)

                # Lift-test: levantar hombro para confirmar agarre
                lift_ok = self.robot.move_joint("hombro", LIFT_SUCCESS_STEPS)

                if lift_ok:
                    reward    += REWARD_GRASP_SUCCESS
                    terminated = True
                    info["success"] = True
                    print(f"  [Env] AGARRE EXITOSO  reward={reward:.2f}")
                else:
                    # Límite de articulación al levantar
                    reward += REWARD_COLLISION * 0.3
                    info["lift_failed"] = True
                    print(f"  [Env] Lift fallido (límite de articulación)")
                    self.robot.open_gripper()

        # ── Acciones 0-7: movimiento de ejes ────────────────────
        else:
            action_ok = self.robot.execute_action(action, steps=ACTION_STEPS)
            time.sleep(0.08)

            if not action_ok:
                reward += REWARD_COLLISION * 0.2
                info["limit_hit"] = True

            # Recompensa por acercarse al centro de la imagen
            # (el agente debe centrar el alimento en la vista de la cámara)
            joints = self.robot.get_joint_positions()
            # Cuanto más centrado esté (joint normalizados cerca de donde debe estar),
            # más recompensa. Proxy simple: recompensa si joint hombro y codo aumentan.
            if action in (2, 4, 6):   # hombro+, codo+, gripper+
                reward += REWARD_DIST_SCALE * 0.01

        # ── Observación actualizada ──────────────────────────────
        obs = self._get_obs()

        # ── Timeout ──────────────────────────────────────────────
        if self._step_count >= self.max_steps:
            truncated       = True
            info["timeout"] = True

        self._episode_reward += reward

        if terminated or truncated:
            info["episode_reward"] = self._episode_reward
            info["episode_steps"]  = self._step_count
            info.setdefault("success", False)

        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------
    # Observación
    # ----------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Construye el vector de estado de 8 dimensiones.
        La parte del alimento (cx, cy, depth) permanece fija del snapshot.
        """
        joints = self.robot.get_joint_positions()   # (4,)
        pinza  = 1.0 if self.robot.is_gripper_closed() else 0.0

        return np.array([
            self._food_cx,
            self._food_cy,
            self._food_depth,
            joints[0],   # base
            joints[1],   # hombro
            joints[2],   # codo
            joints[3],   # gripper (sube/baja)
            pinza,
        ], dtype=np.float32)

    def render(self):
        if self.render_mode != "human":
            return
        joints = self.robot.get_raw_positions()
        print(
            f"  Step={self._step_count:3d} | "
            f"food=({self._food_cx:.2f},{self._food_cy:.2f},d={self._food_depth:.2f}) | "
            f"base={joints.get('base',0):+5d} "
            f"hombro={joints.get('hombro',0):+5d} "
            f"codo={joints.get('codo',0):+5d} "
            f"gripper={joints.get('gripper',0):+5d} | "
            f"pinza={'C' if self.robot.is_gripper_closed() else 'A'}"
        )

    def close(self):
        pass


# ============================================================
# Callbacks
# ============================================================

class BCInitCallback(BaseCallback):
    """Copia pesos del modelo BC a la política PPO en el primer step."""

    def __init__(self, bc_model: PolicyNetwork, verbose: int = 0):
        super().__init__(verbose)
        self.bc_model    = bc_model
        self._initialized = False

    def _on_step(self) -> bool:
        if not self._initialized:
            try:
                pi = self.model.policy.mlp_extractor.policy_net
                bc = self.bc_model
                # Copiar solo los parámetros de forma compatible
                with torch.no_grad():
                    for (n1, p1), (n2, p2) in zip(
                        pi.named_parameters(), bc.named_parameters()
                    ):
                        if p1.shape == p2.shape:
                            p1.data.copy_(p2.data)
                print("[BCInit] Pesos BC transferidos a la política PPO.")
            except Exception as e:
                print(f"[BCInit] No se pudo transferir: {e}")
            self._initialized = True
        return True


class TrainingLogger(BaseCallback):

    def __init__(self, log_freq: int = 500):
        super().__init__()
        self.log_freq      = log_freq
        self._ep_rewards:   list = []
        self._ep_lengths:   list = []
        self._ep_successes: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_reward" in info:
                self._ep_rewards.append(info["episode_reward"])
                self._ep_lengths.append(info["episode_steps"])
                self._ep_successes.append(int(info.get("success", False)))

        if self.num_timesteps % self.log_freq == 0 and self._ep_rewards:
            w       = 50
            recent_r = self._ep_rewards[-w:]
            recent_l = self._ep_lengths[-w:]
            recent_s = self._ep_successes[-w:]
            print(
                f"  [RL] step={self.num_timesteps:8d} | "
                f"reward={np.mean(recent_r):+6.2f} | "
                f"ep_len={np.mean(recent_l):5.0f} | "
                f"éxito={np.mean(recent_s)*100:5.1f}%"
            )
        return True


# ============================================================
# Entrenador RL
# ============================================================

class RLTrainer:

    def __init__(
        self,
        robot:       RobotInterface,
        vision:      VisionPipeline,
        use_bc_init: bool = True,
    ):
        Path(RL_MODEL_PATH).mkdir(parents=True, exist_ok=True)

        def _make_env():
            env = RoboticArmEnv(robot, vision)
            return Monitor(env, os.path.join(RL_MODEL_PATH, "monitor"))

        self.env = DummyVecEnv([_make_env])

        policy_kwargs = dict(
            net_arch=dict(
                pi=[BC_HIDDEN_DIM, BC_HIDDEN_DIM // 2],
                vf=[BC_HIDDEN_DIM, BC_HIDDEN_DIM // 2],
            ),
            activation_fn=nn.ReLU,
        )

        self.ppo = PPO(
            policy          = "MlpPolicy",
            env             = self.env,
            learning_rate   = RL_LEARNING_RATE,
            n_steps         = RL_N_STEPS,
            batch_size      = RL_BATCH_SIZE,
            n_epochs        = RL_N_EPOCHS,
            gamma           = RL_GAMMA,
            ent_coef        = RL_ENT_COEF,
            clip_range      = 0.2,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            policy_kwargs   = policy_kwargs,
            verbose         = 0,
            tensorboard_log = os.path.join(RL_MODEL_PATH, "tb_logs"),
        )

        self.callbacks = [
            CheckpointCallback(save_freq=5000, save_path=RL_MODEL_PATH, name_prefix="ppo_brazo"),
            TrainingLogger(log_freq=500),
        ]

        if use_bc_init and os.path.exists(BC_MODEL_PATH):
            bc_model = load_bc_model(BC_MODEL_PATH)
            self.callbacks.insert(0, BCInitCallback(bc_model))
            print("[RLTrainer] Inicialización BC programada.")
        elif use_bc_init:
            print(f"[RLTrainer] '{BC_MODEL_PATH}' no encontrado. Entrenando desde cero.")

    def train(self, total_steps: int = RL_TOTAL_STEPS):
        print("\n" + "=" * 58)
        print("  NUTRIBOT — ENTRENAMIENTO PPO")
        print(f"  {total_steps:,} pasos totales")
        print(f"  STATE={STATE_DIM}  ACTION={ACTION_DIM}")
        print(f"  Éxito: cerrar pinza + levantar hombro {LIFT_SUCCESS_STEPS} pasos")
        print("=" * 58 + "\n")
        t0 = time.time()
        self.ppo.learn(
            total_timesteps     = total_steps,
            callback            = self.callbacks,
            reset_num_timesteps = True,
            progress_bar        = True,
        )
        elapsed    = time.time() - t0
        final_path = os.path.join(RL_MODEL_PATH, "ppo_final.zip")
        self.ppo.save(final_path)
        print(f"\n[RLTrainer] Listo en {elapsed/60:.1f} min. Modelo: {final_path}")

    def resume(self, checkpoint_path: str, total_steps: int = RL_TOTAL_STEPS):
        self.ppo = PPO.load(
            checkpoint_path,
            env             = self.env,
            tensorboard_log = os.path.join(RL_MODEL_PATH, "tb_logs"),
        )
        print(f"[RLTrainer] Checkpoint: {checkpoint_path}")
        self.train(total_steps)


# ============================================================
# Ejecución de política entrenada
# ============================================================

def run_policy(policy_path: str, robot: RobotInterface, vision: VisionPipeline, n_episodes: int = 10):
    ppo = PPO.load(policy_path)
    env = RoboticArmEnv(robot, vision, render_mode="human")

    print(f"\n[RunPolicy] {n_episodes} episodios con modelo: {policy_path}\n")
    successes = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done   = False
        total  = 0.0
        step   = 0

        while not done:
            action, _ = ppo.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(int(action))
            total += rew
            step  += 1
            done   = terminated or truncated
            env.render()

        s = info.get("success", False)
        successes += int(s)
        print(f"  Ep {ep:2d}: {'ÉXITO' if s else 'FALLO'}  reward={total:+.2f}  pasos={step}")

    print(f"\n[RunPolicy] {successes}/{n_episodes}  ({successes/n_episodes*100:.0f}%)")
    env.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriBot — Entrenamiento PPO")
    parser.add_argument("--sim",        action="store_true")
    parser.add_argument("--steps",      type=int, default=RL_TOTAL_STEPS)
    parser.add_argument("--no_bc_init", action="store_true")
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--run",        type=str, default=None)
    parser.add_argument("--episodes",   type=int, default=10)
    args = parser.parse_args()

    with RobotInterface(simulate=args.sim) as robot:
        with VisionPipeline() as vision:
            robot.home()
            robot.open_gripper()

            if args.run:
                run_policy(args.run, robot, vision, n_episodes=args.episodes)

            elif args.resume:
                RLTrainer(robot, vision, not args.no_bc_init).resume(args.resume, args.steps)

            else:
                if not os.path.exists(BC_MODEL_PATH) and not args.no_bc_init:
                    print(f"[RL] No se encontró '{BC_MODEL_PATH}'.")
                    ans = input("[RL] ¿Continuar sin BC? (s/n): ")
                    if ans.lower() != 's':
                        exit(0)
                RLTrainer(robot, vision, not args.no_bc_init).train(args.steps)