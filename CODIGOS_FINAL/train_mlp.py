import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

CLEAN_PKL  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_complementos\demos\demonstrations_clean.pkl"
MODELO_PT  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_complementos\modelo_bc.pt"

HIDDEN       = [128, 128, 64]
DROPOUT      = 0.05
LR           = 0.001
EPOCHS       = 3000
BATCH_SIZE   = 32
VAL_SPLIT    = 0.15
SEED         = 42
PRINT_EVERY  = 200


class BrazoMLP(nn.Module):
    def __init__(self, hidden=None, dropout=DROPOUT):
        super().__init__()
        if hidden is None:
            hidden = HIDDEN

        capas = []
        in_dim = 2
        for h in hidden:
            capas += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        capas.append(nn.Linear(in_dim, 3))

        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


def cargar_datos(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    eps = data.get("episodes", [])

    X_list, Y_list = [], []
    descartados = 0

    for ep in eps:
        ft    = ep.get("food_target")
        celda = ep.get("celda_objetivo")

        if ft is None or celda is None:
            descartados += 1
            continue
        cx = ft.get("cx_norm")
        cy = ft.get("cy_norm")
        if cx is None or cy is None:
            descartados += 1
            continue

        b = c = h = 0
        for s in ep.get("steps", []):
            cmd   = s.get("cmd", "")
            pasos = s.get("steps_executed", 0)
            if cmd == "B+": b += pasos
            elif cmd == "B-": b -= pasos
            elif cmd == "C+": c += pasos
            elif cmd == "C-": c -= pasos
            elif cmd == "H+": h += pasos
            elif cmd == "H-": h -= pasos

        X_list.append([cx, cy])
        Y_list.append([b, c, h])

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    print(f"  Muestras cargadas: {len(X)}  (descartadas: {descartados})")
    return X, Y


def normalizar(Y_train):
    mean = Y_train.mean(axis=0)
    std  = Y_train.std(axis=0) + 1e-8
    return mean, std


def entrenar(X, Y, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
             val_split=VAL_SPLIT, seed=SEED):

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_val  = max(1, int(len(X) * val_split))
    idx    = np.random.permutation(len(X))
    val_i, tr_i = idx[:n_val], idx[n_val:]

    X_tr, Y_tr = X[tr_i], Y[tr_i]
    X_val, Y_val = X[val_i], Y[val_i]
    print(f"  Train: {len(X_tr)}  |  Val: {len(X_val)}")

    y_mean, y_std = normalizar(Y_tr)
    Y_tr_n  = (Y_tr  - y_mean) / y_std
    Y_val_n = (Y_val - y_mean) / y_std
    print(f"  Y_mean: {y_mean.tolist()}")
    print(f"  Y_std : {y_std.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    Xtr_t   = torch.tensor(X_tr,   device=device)
    Ytr_t   = torch.tensor(Y_tr_n, device=device)
    Xval_t  = torch.tensor(X_val,  device=device)
    Yval_t  = torch.tensor(Y_val_n, device=device)

    loader = DataLoader(TensorDataset(Xtr_t, Ytr_t),
                        batch_size=batch_size, shuffle=True)

    model     = BrazoMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=1.0)

    mejor_val    = float("inf")
    mejor_estado = None
    historial    = {"tr": [], "val": []}

    print(f"\n  Entrenando {epochs} épocas...")
    print(f"  {'Época':>8}  {'Loss_tr':>10}  {'Loss_val':>10}  "
          f"{'RMSE_B':>8}  {'RMSE_C':>8}  {'RMSE_H':>8}")
    print("  " + "─" * 62)

    for ep in range(1, epochs + 1):
        model.train()
        loss_tr = 0.0
        for Xb, Yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_tr += loss.item() * len(Xb)
        loss_tr /= len(X_tr)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val  = model(Xval_t)
            loss_val  = criterion(pred_val, Yval_t).item()
            pred_real = pred_val.cpu().numpy() * y_std + y_mean
            real_real = Y_val
            rmse      = np.sqrt(np.mean((pred_real - real_real)**2, axis=0))

        historial["tr"].append(loss_tr)
        historial["val"].append(loss_val)

        if loss_val < mejor_val:
            mejor_val    = loss_val
            mejor_estado = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % PRINT_EVERY == 0 or ep == 1 or ep == epochs:
            print(f"  {ep:>8}  {loss_tr:>10.5f}  {loss_val:>10.5f}  "
                  f"{rmse[0]:>8.1f}  {rmse[1]:>8.1f}  {rmse[2]:>8.1f}")

    print(f"\n  Mejor val loss: {mejor_val:.5f}")

    model.load_state_dict(mejor_estado)

    model.eval()
    Xall_t = torch.tensor(X, device=device)
    with torch.no_grad():
        Y_pred_n = model(Xall_t).cpu().numpy()
    Y_pred = Y_pred_n * y_std + y_mean
    rmse_all = np.sqrt(np.mean((Y_pred - Y)**2, axis=0))
    mae_all  = np.mean(np.abs(Y_pred - Y), axis=0)

    print(f"\n  ─── Evaluación final (todos los datos) ───")
    print(f"  {'':10} {'RMSE (pasos)':>14}  {'MAE (pasos)':>14}")
    for nombre, r, m in zip(["Base","Codo","Hombro"], rmse_all, mae_all):
        print(f"  {nombre:<10} {r:>14.1f}  {m:>14.1f}")

    return model, y_mean, y_std, device, historial


def guardar_modelo(model, y_mean, y_std, output_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "y_mean":           y_mean.tolist(),
        "y_std":            y_std.tolist(),
        "hidden":           HIDDEN,
        "dropout":          DROPOUT,
        "input_dim":        2,
        "output_dim":       3,
        "output_names":     ["base", "codo", "hombro"],
        "trained_at":       datetime.now().isoformat(),
        "n_params":         sum(p.numel() for p in model.parameters()),
    }
    torch.save(checkpoint, output_path)
    print(f"\n  Modelo guardado en: {output_path}")
    print(f"  Parámetros totales: {checkpoint['n_params']}")


def main():
    parser = argparse.ArgumentParser(description="Entrenar MLP de Behavior Cloning")
    parser.add_argument("--dataset", default=CLEAN_PKL)
    parser.add_argument("--output",  default=MODELO_PT)
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--lr",      type=float, default=LR)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("=" * 60)
    print("  ENTRENAMIENTO MLP — Behavior Cloning Brazo Robótico")
    print("=" * 60)
    print(f"\n  Dataset : {args.dataset}")
    print(f"  Salida  : {args.output}")
    print(f"  Arq.    : 2 → {' → '.join(str(h) for h in HIDDEN)} → 3")
    print(f"  Épocas  : {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch}\n")

    print("[1] Cargando datos...")
    X, Y = cargar_datos(args.dataset)

    print(f"\n  Rango de inputs:")
    print(f"    cx_norm: [{X[:,0].min():.3f}, {X[:,0].max():.3f}]")
    print(f"    cy_norm: [{X[:,1].min():.3f}, {X[:,1].max():.3f}]")
    print(f"  Rango de outputs (pasos):")
    print(f"    Base  : [{Y[:,0].min():.0f}, {Y[:,0].max():.0f}]")
    print(f"    Codo  : [{Y[:,1].min():.0f}, {Y[:,1].max():.0f}]")
    print(f"    Hombro: [{Y[:,2].min():.0f}, {Y[:,2].max():.0f}]")

    print("\n[2] Entrenando MLP...")
    model, y_mean, y_std, device, hist = entrenar(
        X, Y,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
    )

    print("\n[3] Guardando modelo...")
    guardar_modelo(model, y_mean, y_std, args.output)

    print("\n[4] Demo de prediccion (5 ejemplos):")
    model.eval()
    idx_demo = np.random.choice(len(X), 5, replace=False)
    print(f"  {'cx':>6}  {'cy':>6}  "
          f"{'B_real':>8}  {'B_pred':>8}  "
          f"{'C_real':>8}  {'C_pred':>8}  "
          f"{'H_real':>8}  {'H_pred':>8}")
    print("  " + "─" * 76)
    with torch.no_grad():
        for i in idx_demo:
            x_t   = torch.tensor(X[i:i+1], device=device)
            y_pred = (model(x_t).cpu().numpy()[0] * y_std + y_mean).round().astype(int)
            y_real = Y[i].round().astype(int)
            print(f"  {X[i,0]:>6.3f}  {X[i,1]:>6.3f}  "
                  f"{y_real[0]:>8d}  {y_pred[0]:>8d}  "
                  f"{y_real[1]:>8d}  {y_pred[1]:>8d}  "
                  f"{y_real[2]:>8d}  {y_pred[2]:>8d}")

    print("\n  Listo. Ahora ejecuta test_bc_mlp.py para probar en el brazo.")


if __name__ == "__main__":
    main()