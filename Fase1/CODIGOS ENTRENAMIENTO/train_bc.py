"""
train_bc.py — Entrenamiento por Imitación (Behavioral Cloning)

USO:
    python train_bc.py
    python train_bc.py --epochs 200 --only_success
    python train_bc.py --eval
    python train_bc.py --plot
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from collections import Counter

from config import (
    DEMO_FILE, BC_MODEL_PATH,
    BC_EPOCHS, BC_BATCH_SIZE, BC_LR, BC_HIDDEN_DIM,
    STATE_DIM, ACTION_DIM,
)


# ============================================================
# Arquitectura de la red política
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class PolicyNetwork(nn.Module):
    """
    MLP residual que mapea estado (STATE_DIM) → logits (ACTION_DIM).

    Entrada : vector de estado de 8 dimensiones
    Salida  : logits sin normalizar para CrossEntropyLoss
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = BC_HIDDEN_DIM,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(state_dim)

        self.fc_in = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(3)
        ])

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.fc_in(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x)

    def predict(self, state: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return int(self(s).argmax(dim=-1).item())

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return torch.softmax(self(s), dim=-1).numpy()[0]


# ============================================================
# Dataset
# ============================================================

class DemoDataset(Dataset):

    def __init__(self, demo_file: str = DEMO_FILE, only_successful: bool = False):
        self.states:  list = []
        self.actions: list = []
        self._load(demo_file, only_successful)

    def _load(self, demo_file: str, only_successful: bool):
        if not os.path.exists(demo_file):
            raise FileNotFoundError(
                f"No se encontró '{demo_file}'. "
                "Ejecuta primero: python collect_demos.py"
            )

        with open(demo_file, "rb") as f:
            data = pickle.load(f)

        episodes = data.get("episodes", [])

        if only_successful:
            before   = len(episodes)
            episodes = [ep for ep in episodes if ep.get("success", False)]
            print(f"[Dataset] Solo exitosos: {len(episodes)} / {before} episodios")

        grasp_eps = sum(1 for ep in episodes if ep.get("grasp_included", False))
        print(f"[Dataset] Episodios con gripper: {grasp_eps} / {len(episodes)}")

        for ep in episodes:
            for step in ep["steps"]:
                self.states.append(step["state"])
                self.actions.append(step["action"])

        self.states  = np.array(self.states,  dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)

        print(f"[Dataset] {len(self.states)} pasos de {len(episodes)} episodios cargados.")
        self._print_balance()

    def _print_balance(self):
        action_names = {
            0:"base+",  1:"base-",  2:"hombro+", 3:"hombro-",
            4:"codo+",  5:"codo-",  6:"muneca+", 7:"muneca-",
            8:"rot+",   9:"rot-",   10:"gripper",
        }
        cnt = Counter(self.actions.tolist())
        print("[Dataset] Distribución de acciones:")
        for a in range(ACTION_DIM):
            n   = cnt.get(a, 0)
            pct = n / len(self.actions) * 100 if len(self.actions) > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {action_names.get(a,'?'):10s} ({a:2d}): {n:4d} ({pct:4.1f}%) {bar}")

    def get_class_weights(self) -> torch.Tensor:
        cnt    = Counter(self.actions.tolist())
        total  = len(self.actions)
        weights = []
        for i in range(ACTION_DIM):
            n = cnt.get(i, 1)
            weights.append(total / (ACTION_DIM * n))
        return torch.FloatTensor(weights)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
        )


# ============================================================
# Entrenador BC
# ============================================================

class BCTrainer:

    def __init__(
        self,
        model:      PolicyNetwork,
        dataset:    DemoDataset,
        lr:         float = BC_LR,
        val_split:  float = 0.15,
    ):
        self.model  = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        self.train_ds, self.val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_loader = DataLoader(
            self.train_ds, batch_size=BC_BATCH_SIZE,
            shuffle=True, num_workers=0, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=BC_BATCH_SIZE,
            shuffle=False, num_workers=0,
        )

        weights         = dataset.get_class_weights().to(self.device)
        self.criterion  = nn.CrossEntropyLoss(weight=weights)
        self.optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=BC_EPOCHS, eta_min=1e-5
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[BCTrainer] Dispositivo : {self.device}")
        print(f"[BCTrainer] Parámetros  : {n_params:,}")
        print(f"[BCTrainer] Train/Val   : {n_train} / {n_val} pasos")

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = total_acc = n = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for states, actions in loader:
                states  = states.to(self.device)
                actions = actions.to(self.device)

                logits = self.model(states)
                loss   = self.criterion(logits, actions)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                preds       = logits.argmax(dim=-1)
                total_loss += loss.item() * len(states)
                total_acc  += (preds == actions).float().sum().item()
                n          += len(states)

        return total_loss / n, total_acc / n

    def train(
        self,
        epochs:    int = BC_EPOCHS,
        save_path: str = BC_MODEL_PATH,
        patience:  int = 20,
    ) -> dict:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

        best_val_loss  = float("inf")
        patience_count = 0

        header = f"{'Epoca':>6} | {'Tr.Loss':>8} | {'Tr.Acc':>7} | {'Vl.Loss':>8} | {'Vl.Acc':>7} | {'LR':>9}"
        print(f"\n[BCTrainer] Iniciando entrenamiento — {epochs} épocas")
        print(header)
        print("-" * len(header))

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            vl_loss, vl_acc = self._run_epoch(self.val_loader,   train=False)
            self.scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)

            lr = self.optimizer.param_groups[0]["lr"]

            improved = vl_loss < best_val_loss
            if improved:
                best_val_loss  = vl_loss
                patience_count = 0
                torch.save({
                    "epoch":      epoch,
                    "state_dict": self.model.state_dict(),
                    "val_loss":   vl_loss,
                    "val_acc":    vl_acc,
                    "optimizer":  self.optimizer.state_dict(),
                    "config": {
                        "state_dim":  STATE_DIM,
                        "action_dim": ACTION_DIM,
                        "hidden_dim": BC_HIDDEN_DIM,
                    },
                }, save_path)
            else:
                patience_count += 1

            if epoch % 5 == 0 or epoch == 1 or improved:
                mark = " *" if improved else ""
                print(
                    f"{epoch:6d} | {tr_loss:8.4f} | {tr_acc*100:6.2f}% | "
                    f"{vl_loss:8.4f} | {vl_acc*100:6.2f}% | {lr:.3e}{mark}"
                )

            if patience_count >= patience:
                print(f"\n[BCTrainer] Early stopping en época {epoch} (paciencia={patience})")
                break

        print(f"\n[BCTrainer] Mejor val_loss: {best_val_loss:.4f}")
        print(f"[BCTrainer] Modelo guardado: {save_path}")
        self._plot_history(history, save_path)
        return history

    def _plot_history(self, history: dict, save_path: str):
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            axes[0].plot(history["train_loss"], label="Train")
            axes[0].plot(history["val_loss"],   label="Val")
            axes[0].set_title("Pérdida (CrossEntropy ponderada)")
            axes[0].set_xlabel("Época")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot([a * 100 for a in history["train_acc"]], label="Train")
            axes[1].plot([a * 100 for a in history["val_acc"]],   label="Val")
            axes[1].set_title("Precisión (%)")
            axes[1].set_xlabel("Época")
            axes[1].set_ylim(0, 100)
            axes[1].axhline(70, color="red", linestyle="--", linewidth=0.8, label="70% objetivo")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.suptitle("Behavioral Cloning — NutriBot")
            plt.tight_layout()
            out = os.path.join(os.path.dirname(save_path), "bc_training_curves.png")
            plt.savefig(out, dpi=120)
            plt.close()
            print(f"[BCTrainer] Curvas guardadas: {out}")
        except ImportError:
            print("[BCTrainer] matplotlib no disponible, sin gráficas.")


# ============================================================
# Evaluación detallada
# ============================================================

def evaluate_bc_policy(
    model_path: str = BC_MODEL_PATH,
    demo_file:  str = DEMO_FILE,
):
    if not os.path.exists(model_path):
        print(f"No se encontró '{model_path}'. Entrena primero.")
        return None

    checkpoint = torch.load(model_path, map_location="cpu")
    cfg        = checkpoint.get("config", {})
    model      = PolicyNetwork(
        state_dim  = cfg.get("state_dim",  STATE_DIM),
        action_dim = cfg.get("action_dim", ACTION_DIM),
        hidden_dim = cfg.get("hidden_dim", BC_HIDDEN_DIM),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = DemoDataset(demo_file)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False)

    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for states, actions in loader:
            logits = model(states)
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(actions.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = (all_preds == all_labels).mean()

    action_names = {
        0:"base+",  1:"base-",  2:"hombro+", 3:"hombro-",
        4:"codo+",  5:"codo-",  6:"muneca+", 7:"muneca-",
        8:"rot+",   9:"rot-",   10:"gripper",
    }

    print("\n" + "=" * 48)
    print("  EVALUACIÓN BC")
    print("=" * 48)
    print(f"  Precisión global : {accuracy*100:.2f}%")
    print(f"  Época guardada   : {checkpoint.get('epoch','?')}")
    print(f"  Val loss         : {checkpoint.get('val_loss', '?'):.4f}")
    print(f"  Val acc          : {checkpoint.get('val_acc', 0)*100:.2f}%")
    print()
    print("  Precisión por acción:")
    for a in range(ACTION_DIM):
        mask = all_labels == a
        if mask.sum() == 0:
            continue
        acc_a = (all_preds[mask] == a).mean()
        bar   = "#" * int(acc_a * 20)
        print(f"    {action_names.get(a,'?'):10s}: {acc_a*100:5.1f}%  {bar}  (n={mask.sum()})")

    ok_flag = accuracy >= 0.70
    print()
    print(f"  {'LISTO para RL' if ok_flag else 'Necesita mas datos o epocas (objetivo: 70%)'}")
    print("=" * 48)
    return accuracy


# ============================================================
# Carga del modelo BC (usado por train_rl)
# ============================================================

def load_bc_model(model_path: str = BC_MODEL_PATH) -> PolicyNetwork:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró '{model_path}'.")
    checkpoint = torch.load(model_path, map_location="cpu")
    cfg        = checkpoint.get("config", {})
    model      = PolicyNetwork(
        state_dim  = cfg.get("state_dim",  STATE_DIM),
        action_dim = cfg.get("action_dim", ACTION_DIM),
        hidden_dim = cfg.get("hidden_dim", BC_HIDDEN_DIM),
    )
    model.load_state_dict(checkpoint["state_dict"])
    val_acc = checkpoint.get("val_acc", 0) * 100
    print(f"[BC] Modelo cargado desde '{model_path}'  (val_acc={val_acc:.1f}%)")
    return model


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Behavioral Cloning — NutriBot")
    parser.add_argument("--epochs",       type=int,   default=BC_EPOCHS)
    parser.add_argument("--lr",           type=float, default=BC_LR)
    parser.add_argument("--patience",     type=int,   default=20)
    parser.add_argument("--eval",         action="store_true")
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--plot",         action="store_true",
                        help="Solo mostrar curvas si ya existe el modelo")
    args = parser.parse_args()

    if args.eval or args.plot:
        evaluate_bc_policy()
    else:
        dataset = DemoDataset(only_successful=args.only_success)

        if len(dataset) < 20:
            print("ERROR: Muy pocos datos. Graba más demos primero.")
            exit(1)

        model   = PolicyNetwork()
        trainer = BCTrainer(model, dataset, lr=args.lr)
        trainer.train(epochs=args.epochs, patience=args.patience)
        evaluate_bc_policy()