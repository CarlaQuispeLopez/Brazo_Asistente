"""
train_rl.py — Aprendizaje por Refuerzo Profundo (PPO) — NutriBot

USO:
    python train_rl.py
    python train_rl.py --sim
    python train_rl.py --steps 300000
    python train_rl.py --no_bc_init
    python train_rl.py --resume models/ppo_policy/ppo_brazo_50000_steps.zip
    python train_rl.py --run    models/ppo_policy/ppo_final.zip

El agente PPO opera SOLO en fases SEARCH y GRASP.
Cuando vision.rl_should_act == False (fase DELIVERY),
el bucle no llama al agente y deja que el controlador
proporcional de vision.py tome el mando.
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
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, BaseCallback,
    )
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
    DEPTH_MAX_CM,
)
from robot_interface import RobotInterface
from vision import VisionPipeline, PipelinePhase
from train_bc import PolicyNetwork, load_bc_model


# ============================================================
# Entorno Gymnasium — Brazo robótico real
# ============================================================

class RoboticArmEnv(gym.Env):
    """
    Entorno que envuelve el hardware real del brazo NutriBot.

    El agente RL solo controla las fases SEARCH y GRASP.
    Cuando vision.py transiciona a DELIVERY, el entorno
    termina el episodio con terminated=True y reward de éxito,
    sin que el agente necesite saber cómo entregar.

    Espacio de observación : Box(-1, 1, shape=(8,), float32)
    Espacio de acción      : Discrete(11)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

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
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        self._step_count:     int   = 0
        self._episode_reward: float = 0.0
        self._last_depth:     float = DEPTH_MAX_CM
        self._frame               = None
        self._grasp_attempted     = False
        self._initial_pos:  dict  = {}

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count     = 0
        self._episode_reward = 0.0
        self._grasp_attempted = False

        self.robot.open_gripper()
        self.vision.notify_gripper_opened()
        self.robot.home()
        time.sleep(0.5)

        self._initial_pos = self.robot.get_raw_positions()

        obs, depth = self._get_obs()
        self._last_depth = depth
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

        depth_before = self._last_depth

        # El agente solo actúa si el pipeline lo permite
        if self.vision.rl_should_act:
            action_ok = self.robot.execute_action(action, steps=ACTION_STEPS)
            time.sleep(0.08)

            if not action_ok:
                reward += REWARD_COLLISION * 0.2
                info["limit_hit"] = True

            # Notificar cierre de gripper a vision.py
            if action == 10:
                depth_now = self.vision.get_current_food_depth()
                self.vision.notify_gripper_closed(depth_now)
                self._grasp_attempted = True

        obs, depth_now = self._get_obs()
        self._last_depth = depth_now

        phase = self.vision.phase

        # Transición a DELIVERY: el agarre fue confirmado
        # El RL termina el episodio con éxito
        if phase == PipelinePhase.DELIVERY:
            reward    += REWARD_GRASP_SUCCESS
            terminated = True
            info["success"] = True
            info["reason"]  = "grasp_confirmed_delivery_active"
            print(f"  [Env] AGARRE CONFIRMADO → DELIVERY  reward={reward:.2f}")

        elif self.vision.rl_should_act:
            # Recompensa densa durante SEARCH y GRASP
            food = self.vision._last_food

            if food is None and self._grasp_attempted:
                # YOLO perdió el alimento tras intentar agarrar
                # Puede ser agarre real; dejar que vision.py lo confirme
                pass

            elif food is None:
                reward += REWARD_COLLISION * 0.1

            else:
                delta = depth_before - depth_now
                if delta > 0:
                    reward += REWARD_DIST_SCALE * delta / DEPTH_MAX_CM

                cx, cy          = food.center_norm
                center_reward   = (0.5 - abs(cx - 0.5) - abs(cy - 0.5))
                reward         += center_reward * 0.1

                if food.reachable and phase == PipelinePhase.SEARCH:
                    reward += 0.02

                if action == 10 and depth_now > 20.0:
                    reward += REWARD_COLLISION * 0.3
                    info["premature_grasp"] = True

        if self._step_count >= self.max_steps:
            truncated      = True
            info["timeout"] = True

        self._episode_reward += reward

        if terminated or truncated:
            info["episode_reward"] = self._episode_reward
            info["episode_steps"]  = self._step_count
            info["success"]        = info.get("success", False)

        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------
    # Observación
    # ----------------------------------------------------------

    def _get_obs(self):
        frame = self.vision.read_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        joints          = self.robot.get_joint_positions()
        obs, phase, ann = self.vision.get_state(frame, joints, self.robot)
        self._frame     = ann

        depth = self.vision.get_current_food_depth()
        return obs.astype(np.float32), depth

    def render(self):
        import cv2
        if self.render_mode == "human" and self._frame is not None:
            cv2.imshow("NutriBot — RL Training", self._frame)
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            return self._frame

    def close(self):
        import cv2
        cv2.destroyAllWindows()


# ============================================================
# Callback: transferencia de pesos BC → PPO
# ============================================================

class BCInitCallback(BaseCallback):
    """
    Se ejecuta justo antes de que empiece el entrenamiento RL.
    Transfiere los pesos del modelo BC al actor de PPO para que
    el agente no comience desde una política aleatoria.
    """

    def __init__(self, bc_model: PolicyNetwork, verbose: int = 1):
        super().__init__(verbose)
        self.bc_model = bc_model

    def _on_training_start(self):
        try:
            self._transfer_weights()
            if self.verbose:
                print("[BCInit] Pesos BC transferidos al actor de PPO.")
        except Exception as e:
            print(f"[BCInit] No se pudieron transferir pesos: {e}")
            print("[BCInit] El agente entrenará desde cero.")

    def _transfer_weights(self):
        ppo_actor  = self.model.policy
        bc_state   = self.bc_model.state_dict()
        ppo_state  = ppo_actor.state_dict()

        transferred = 0

        # Capa de entrada: fc_in.0 (Linear) → mlp_extractor.policy_net.0
        key_bc  = "fc_in.0.weight"
        key_ppo = "mlp_extractor.policy_net.0.weight"
        if key_bc in bc_state and key_ppo in ppo_state:
            if bc_state[key_bc].shape == ppo_state[key_ppo].shape:
                ppo_state[key_ppo]                             = bc_state[key_bc]
                ppo_state["mlp_extractor.policy_net.0.bias"]  = bc_state["fc_in.0.bias"]
                transferred += 1

        # Cabeza de política: policy_head.3 (último Linear) → action_net
        key_bc  = "policy_head.3.weight"
        key_ppo = "action_net.weight"
        if key_bc in bc_state and key_ppo in ppo_state:
            if bc_state[key_bc].shape == ppo_state[key_ppo].shape:
                ppo_state[key_ppo]           = bc_state[key_bc]
                ppo_state["action_net.bias"] = bc_state["policy_head.3.bias"]
                transferred += 1

        ppo_actor.load_state_dict(ppo_state, strict=False)
        if self.verbose:
            print(f"[BCInit] {transferred} capas transferidas.")

    def _on_step(self) -> bool:
        return True


# ============================================================
# Callback: logger de métricas
# ============================================================

class TrainingLogger(BaseCallback):

    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq      = log_freq
        self._ep_rewards:  list = []
        self._ep_lengths:  list = []
        self._ep_successes: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_reward" in info:
                self._ep_rewards.append(info["episode_reward"])
                self._ep_lengths.append(info["episode_steps"])
                self._ep_successes.append(int(info.get("success", False)))

        if self.num_timesteps % self.log_freq == 0 and self._ep_rewards:
            window       = 50
            recent_r     = self._ep_rewards[-window:]
            recent_l     = self._ep_lengths[-window:]
            recent_s     = self._ep_successes[-window:]
            mean_r       = np.mean(recent_r)
            mean_l       = np.mean(recent_l)
            success_rate = np.mean(recent_s) * 100

            print(
                f"  [RL] step={self.num_timesteps:8d} | "
                f"mean_reward={mean_r:+6.2f} | "
                f"mean_ep_len={mean_l:5.0f} | "
                f"success={success_rate:5.1f}%  "
                f"(últimos {min(len(recent_r), window)} ep)"
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
            env = Monitor(env, os.path.join(RL_MODEL_PATH, "monitor"))
            return env

        self.env = DummyVecEnv([_make_env])
        print(f"[RLTrainer] Entorno creado. obs={STATE_DIM}  act={ACTION_DIM}")

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
        print("[RLTrainer] PPO inicializado.")

        self.callbacks = [
            CheckpointCallback(
                save_freq   = 5000,
                save_path   = RL_MODEL_PATH,
                name_prefix = "ppo_brazo",
            ),
            TrainingLogger(log_freq=500),
        ]

        if use_bc_init and os.path.exists(BC_MODEL_PATH):
            bc_model = load_bc_model(BC_MODEL_PATH)
            self.callbacks.insert(0, BCInitCallback(bc_model, verbose=1))
            print("[RLTrainer] Inicialización BC programada.")
        elif use_bc_init:
            print(f"[RLTrainer] No se encontró '{BC_MODEL_PATH}'. Entrenando desde cero.")

    def train(self, total_steps: int = RL_TOTAL_STEPS):
        print("\n" + "=" * 58)
        print("  NUTRIBOT — ENTRENAMIENTO PPO")
        print(f"  {total_steps:,} pasos totales")
        print("  Agente RL: fases SEARCH y GRASP")
        print("  Fase DELIVERY: controlador proporcional")
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

        print(f"\n[RLTrainer] Entrenamiento terminado en {elapsed/60:.1f} min.")
        print(f"[RLTrainer] Modelo final: {final_path}")

    def resume(self, checkpoint_path: str, total_steps: int = RL_TOTAL_STEPS):
        self.ppo = PPO.load(
            checkpoint_path,
            env             = self.env,
            tensorboard_log = os.path.join(RL_MODEL_PATH, "tb_logs"),
        )
        print(f"[RLTrainer] Checkpoint cargado: {checkpoint_path}")
        self.train(total_steps)


# ============================================================
# Ejecución de la política entrenada
# ============================================================

def run_policy(
    policy_path: str,
    robot:       RobotInterface,
    vision:      VisionPipeline,
    n_episodes:  int = 10,
):
    import cv2

    ppo = PPO.load(policy_path)
    env = RoboticArmEnv(robot, vision, render_mode="human")

    print(f"\n[RunPolicy] Ejecutando {n_episodes} episodios con política entrenada...")
    print(f"[RunPolicy] Modelo: {policy_path}\n")

    successes = 0
    for ep in range(1, n_episodes + 1):
        obs, _  = env.reset()
        done    = False
        total_r = 0.0
        step    = 0

        while not done:
            if vision.rl_should_act:
                action, _ = ppo.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
            else:
                # Fase DELIVERY: vision.py controla; solo avanzar el bucle
                obs, reward, terminated, truncated, info = env.step(0)
                reward = 0.0

            total_r += reward
            step    += 1
            done     = terminated or truncated
            env.render()

        success = info.get("success", False)
        successes += int(success)
        result    = "EXITO" if success else "FALLO"
        print(f"  Ep {ep:2d}: {result}  reward={total_r:+.2f}  pasos={step}")

    rate = successes / n_episodes * 100
    print(f"\n[RunPolicy] Tasa de éxito: {successes}/{n_episodes}  ({rate:.0f}%)")
    env.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento PPO — NutriBot")
    parser.add_argument("--sim",        action="store_true",
                        help="Modo simulado sin hardware")
    parser.add_argument("--steps",      type=int, default=RL_TOTAL_STEPS,
                        help="Pasos totales de entrenamiento")
    parser.add_argument("--no_bc_init", action="store_true",
                        help="No inicializar con BC")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Checkpoint para continuar entrenamiento")
    parser.add_argument("--run",        type=str, default=None,
                        help="Checkpoint para ejecutar la política (no entrenar)")
    parser.add_argument("--episodes",   type=int, default=10,
                        help="Episodios al ejecutar con --run")
    args = parser.parse_args()

    from config import SERIAL_PORT

    with RobotInterface(simulate=args.sim) as robot:
        with VisionPipeline() as vision:

            robot.home()
            robot.open_gripper()

            if args.run:
                run_policy(args.run, robot, vision, n_episodes=args.episodes)

            elif args.resume:
                trainer = RLTrainer(
                    robot, vision,
                    use_bc_init=not args.no_bc_init,
                )
                trainer.resume(args.resume, total_steps=args.steps)

            else:
                if not os.path.exists(BC_MODEL_PATH) and not args.no_bc_init:
                    print(f"[RL] No se encontró '{BC_MODEL_PATH}'.")
                    print("[RL] Ejecuta primero: python train_bc.py")
                    ans = input("¿Continuar sin inicialización BC? (s/n): ")
                    if ans.lower() != 's':
                        exit(0)

                trainer = RLTrainer(
                    robot, vision,
                    use_bc_init=not args.no_bc_init,
                )
                trainer.train(total_steps=args.steps)