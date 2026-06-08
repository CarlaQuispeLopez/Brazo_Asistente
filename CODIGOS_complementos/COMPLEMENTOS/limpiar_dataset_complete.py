"""
limpiar_dataset_complete.py
============================
Limpia demonstrations_complete.pkl (190 episodios: celdas + esquinas).

Reglas:
  1. Eliminar EP7 primera toma (índice 6, 4 pasos, incompleta).
  2. Eliminar episodios con celda_objetivo = None (fuera de la malla).
  3. Eliminar episodios con success=False, EXCEPTO EP15 (celda=16).
  4. Por cada episodio restante:
       a. Eliminar TODOS los pasos GRIP 90.
       b. Sumar pasos con signo por articulación (B, C, H) → valor neto.
       c. Solo conservar Base, Codo, Hombro y GRIP 0.
       d. Orden final: Base → Codo → Hombro → GRIP 0.
       e. Mantener siempre GRIP 0 al final.
  5. Guardar como demonstrations_clean.pkl en la misma carpeta.
"""

import os
import pickle
from collections import Counter
from datetime import datetime

# ── Rutas ─────────────────────────────────────────────────────────────────────
DEMOS_DIR  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\demos"
INPUT_PKL  = os.path.join(DEMOS_DIR, "demonstrations complete.pkl")
OUTPUT_PKL = os.path.join(DEMOS_DIR, "demonstrations_clean.pkl")

# ── Cargar ─────────────────────────────────────────────────────────────────────
with open(INPUT_PKL, "rb") as f:
    data = pickle.load(f)

episodes_raw = data.get("episodes", [])
print(f"Episodios en el dataset completo: {len(episodes_raw)}")

# ── Reglas de filtrado ─────────────────────────────────────────────────────────
BAD_INDEX        = 6    # EP7 (base-0) → toma incompleta de 4 pasos
KEEP_FALSE_CELDA = 16   # EP15: celda=16, success=False → conservar de todas formas

episodes_filtered = []
removidos = []

for i, ep in enumerate(episodes_raw):
    ep_num  = i + 1
    celda   = ep.get("celda_objetivo")
    exito   = ep.get("success")
    n_pasos = len(ep.get("steps", []))

    # 1. EP7 primera toma mala
    if i == BAD_INDEX:
        removidos.append(
            f"EP{ep_num:>4}  celda={celda:<5} exito={exito}  → primera toma incompleta ({n_pasos} pasos)")
        continue

    # 2. Episodios con celda=None (fuera de la malla, no usables para BC)
    if celda is None:
        removidos.append(
            f"EP{ep_num:>4}  celda=None   exito={exito}  → fuera de la malla")
        continue

    # 3. success=False, excepto EP15
    if exito is False and celda != KEEP_FALSE_CELDA:
        removidos.append(
            f"EP{ep_num:>4}  celda={celda:<5} exito=False → eliminado")
        continue

    episodes_filtered.append((ep_num, ep))

print(f"\nEliminados ({len(removidos)}):")
for r in removidos:
    print(f"  - {r}")

# ── Función de limpieza por episodio ──────────────────────────────────────────
def limpiar_episodio(ep):
    """
    Devuelve (clean_steps, total_base, total_codo, total_hombro).
    clean_steps: lista de máximo 4 entradas → [Base?, Codo?, Hombro?, GRIP_0].
    """
    total_base   = 0
    total_codo   = 0
    total_hombro = 0
    tiene_grip0  = False

    for s in ep.get("steps", []):
        cmd   = s.get("cmd", "")
        pasos = s.get("steps_executed", 0)

        if cmd == "GRIP 90":
            continue                  # siempre se borra
        elif cmd == "GRIP 0":
            tiene_grip0 = True        # se añade al final
        elif cmd == "B+":
            total_base   += pasos
        elif cmd == "B-":
            total_base   -= pasos
        elif cmd == "C+":
            total_codo   += pasos
        elif cmd == "C-":
            total_codo   -= pasos
        elif cmd == "H+":
            total_hombro += pasos
        elif cmd == "H-":
            total_hombro -= pasos
        # M/G (muneca, rotacion) → ignorar

    clean_steps = []

    if total_base != 0:
        clean_steps.append({
            "cmd":            "B+" if total_base > 0 else "B-",
            "steps_executed": abs(total_base),
            "action":         0 if total_base > 0 else 1,
            "angulo_pinza":   None,
        })

    if total_codo != 0:
        clean_steps.append({
            "cmd":            "C+" if total_codo > 0 else "C-",
            "steps_executed": abs(total_codo),
            "action":         4 if total_codo > 0 else 5,
            "angulo_pinza":   None,
        })

    if total_hombro != 0:
        clean_steps.append({
            "cmd":            "H+" if total_hombro > 0 else "H-",
            "steps_executed": abs(total_hombro),
            "action":         2 if total_hombro > 0 else 3,
            "angulo_pinza":   None,
        })

    # GRIP 0 siempre al final
    clean_steps.append({
        "cmd":            "GRIP 0",
        "steps_executed": 0,
        "action":         10,
        "angulo_pinza":   0,
    })

    return clean_steps, total_base, total_codo, total_hombro


# ── Construir dataset limpio ───────────────────────────────────────────────────
clean_episodes = []

print(f"\n{'─'*82}")
print(f"{'EP_orig':>8}  {'Celda':>6}  {'Esq':>5}  {'Exito':>6}  "
      f"{'Base':>7}  {'Codo':>7}  {'Hombro':>7}  Cmds")
print(f"{'─'*82}")

for ep_num, ep in episodes_filtered:
    clean_steps, b, c, h = limpiar_episodio(ep)

    ep_limpio = {
        # Metadata del objetivo
        "food_target":      ep.get("food_target"),
        "celda_objetivo":   ep.get("celda_objetivo"),
        "fila_objetivo":    ep.get("fila_objetivo"),
        "columna_objetivo": ep.get("columna_objetivo"),
        # Info de esquina (demos de corners)
        "esquina_objetivo": ep.get("esquina_objetivo"),
        "esquina_col":      ep.get("esquina_col"),
        "esquina_row":      ep.get("esquina_row"),
        "esquina_px":       ep.get("esquina_px"),
        "esquina_py":       ep.get("esquina_py"),
        # Trayectoria comprimida
        "steps":            clean_steps,
        "n_steps":          len(clean_steps),
        # Metadata
        "success":          ep.get("success"),
        "timestamp":        ep.get("timestamp"),
        "home_used":        ep.get("home_used"),
        "ep_original":      ep_num,
    }
    clean_episodes.append(ep_limpio)

    esq      = ep.get("esquina_objetivo") or "-"
    exito_s  = "True " if ep.get("success") else "False"
    cmds_s   = " ".join(s["cmd"] for s in clean_steps)
    print(f"  EP{ep_num:>4}    {ep.get('celda_objetivo'):>5}   {esq:>4}   {exito_s}  "
          f"{b:>+7}  {c:>+7}  {h:>+7}  {cmds_s}")

print(f"{'─'*82}")
print(f"\nEpisodios limpios: {len(clean_episodes)}")

# ── Guardar ────────────────────────────────────────────────────────────────────
clean_data = {
    "episodes":   clean_episodes,
    "created":    data.get("created"),
    "cleaned_at": datetime.now().isoformat(),
    "n_original": len(episodes_raw),
    "n_clean":    len(clean_episodes),
}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(clean_data, f)

print(f"\nGuardado en: {OUTPUT_PKL}")

# ── Análisis de cobertura de la malla ─────────────────────────────────────────
GRID_COLS    = 12
GRID_ROWS    = 9
TOTAL_CELDAS = GRID_COLS * GRID_ROWS   # 108

celdas_con_demo = set(
    ep["celda_objetivo"] for ep in clean_episodes
    if ep["celda_objetivo"] is not None
)
celdas_sin_demo = sorted(set(range(1, TOTAL_CELDAS + 1)) - celdas_con_demo)
cnt_demo = Counter(
    ep["celda_objetivo"] for ep in clean_episodes
    if ep["celda_objetivo"] is not None
)

# Separar episodios de celda vs esquina para estadísticas
eps_celda   = [ep for ep in clean_episodes if ep.get("esquina_objetivo") is None]
eps_esquina = [ep for ep in clean_episodes if ep.get("esquina_objetivo") is not None]

print(f"\n{'='*62}")
print(f"  COBERTURA DE LA MALLA ({GRID_COLS}x{GRID_ROWS} = {TOTAL_CELDAS} celdas)")
print(f"{'='*62}")
print(f"  Demos de CENTRO de celda : {len(eps_celda)}")
print(f"  Demos de ESQUINA          : {len(eps_esquina)}")
print(f"  Total demos limpias       : {len(clean_episodes)}")
print(f"  Celdas CON al menos 1 demo: {len(celdas_con_demo)}")
print(f"  Celdas SIN ninguna demo   : {len(celdas_sin_demo)}")

# Mapa visual
print("\n  Mapa de la malla  (o=con demo  X=sin demo):")
for r in range(GRID_ROWS):
    fila = ""
    for c in range(GRID_COLS):
        num = r * GRID_COLS + c + 1
        fila += "o " if num in celdas_con_demo else "X "
    ini = r * GRID_COLS + 1
    fin = (r + 1) * GRID_COLS
    print(f"  Fila {r+1:>2} ({ini:>3}-{fin:>3}):  {fila}")

print(f"\n  Celdas SIN demo ({len(celdas_sin_demo)} total):")
for r in range(GRID_ROWS):
    sin = [c for c in celdas_sin_demo if r * GRID_COLS + 1 <= c <= (r+1) * GRID_COLS]
    if sin:
        print(f"    Fila {r+1}: {sin}")

print(f"\n  Celdas con MÁS de 1 demo:")
for celda, cnt in sorted(cnt_demo.items()):
    if cnt > 1:
        r = (celda - 1) // GRID_COLS
        c = (celda - 1) % GRID_COLS
        print(f"    Celda {celda:>3} (Fila {r+1}, Col {c+1}): {cnt} demos")
