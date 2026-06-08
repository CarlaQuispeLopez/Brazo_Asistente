"""
Limpieza del dataset de demostraciones.

Reglas aplicadas:
  1. Eliminar EP7 primera toma (4 pasos, incompleta).
  2. Eliminar todos los episodios con success=False, EXCEPTO EP15.
  3. Por cada episodio restante:
       - Eliminar todos los pasos GRIP 90.
       - Sumar pasos con signo por articulacion (B, C, H).
       - Mantener GRIP 0 al final.
       - Solo conservar Base, Codo, Hombro y GRIP 0.
       - Ordenar: Base → Codo → Hombro → GRIP 0.
  4. Guardar como demonstrations_clean.pkl en la misma carpeta.
"""

import os
import pickle
from collections import Counter
from datetime import datetime

# ── Rutas ────────────────────────────────────────────────────────────────────
DEMOS_DIR  = r"C:\Users\MSI LAPTOP\Documents\BRAZO2\CODIGOS_NUEVO\demos"
INPUT_PKL  = os.path.join(DEMOS_DIR, "demonstrations.pkl")
OUTPUT_PKL = os.path.join(DEMOS_DIR, "demonstrations_clean.pkl")

# ── Cargar ────────────────────────────────────────────────────────────────────
with open(INPUT_PKL, "rb") as f:
    data = pickle.load(f)

episodes_raw = data.get("episodes", [])
print(f"Episodios originales: {len(episodes_raw)}")

# ── 1. Identificar EP7 primera toma ──────────────────────────────────────────
# EP7 es el índice 6 (base-0). Tiene solo 4 pasos y celda=7.
# La segunda toma de celda=7 es EP8 (índice 7) y es válida.
BAD_INDEX = 6   # EP7 en base-1 = índice 6 en base-0

# ── 2. Filtrar episodios ──────────────────────────────────────────────────────
KEEP_FALSE_CELDA = 16   # EP15: celda=16, exito=False → conservar

episodes_filtered = []
removidos = []

for i, ep in enumerate(episodes_raw):
    ep_num = i + 1   # número 1-based para mensajes

    # Eliminar EP7 (primera toma mala)
    if i == BAD_INDEX:
        removidos.append(f"EP{ep_num} (primera toma incompleta, celda={ep.get('celda_objetivo')})")
        continue

    # Eliminar exito=False, EXCEPTO la celda especial
    if ep.get("success") is False and ep.get("celda_objetivo") != KEEP_FALSE_CELDA:
        removidos.append(f"EP{ep_num} (success=False, celda={ep.get('celda_objetivo')})")
        continue

    episodes_filtered.append((ep_num, ep))

print(f"\nEpisodios eliminados ({len(removidos)}):")
for r in removidos:
    print(f"  - {r}")

# ── 3. Limpiar y comprimir cada episodio ─────────────────────────────────────
def limpiar_episodio(ep):
    """
    Devuelve lista de 4 pasos limpios: [Base, Codo, Hombro, GRIP_0]
    """
    steps = ep.get("steps", [])

    total_base  = 0
    total_codo  = 0
    total_hombro = 0
    grip_0_step  = None

    for s in steps:
        cmd            = s.get("cmd", "")
        pasos          = s.get("steps_executed", 0)
        angulo_pinza   = s.get("angulo_pinza")

        if cmd == "GRIP 90":
            continue                      # ← eliminar apertura de garra

        elif cmd == "GRIP 0":
            grip_0_step = s              # guardar el paso GRIP 0 original

        elif cmd == "B+":
            total_base  += pasos
        elif cmd == "B-":
            total_base  -= pasos

        elif cmd == "C+":
            total_codo  += pasos
        elif cmd == "C-":
            total_codo  -= pasos

        elif cmd == "H+":
            total_hombro += pasos
        elif cmd == "H-":
            total_hombro -= pasos

        # M (muneca) y G/rotacion: ignorar

    # Construir los 4 pasos limpios (solo si el valor != 0)
    clean_steps = []

    # Base
    if total_base != 0:
        clean_steps.append({
            "cmd":            "B+" if total_base > 0 else "B-",
            "steps_executed": abs(total_base),
            "action":         0 if total_base > 0 else 1,
            "angulo_pinza":   None,
        })

    # Codo
    if total_codo != 0:
        clean_steps.append({
            "cmd":            "C+" if total_codo > 0 else "C-",
            "steps_executed": abs(total_codo),
            "action":         4 if total_codo > 0 else 5,
            "angulo_pinza":   None,
        })

    # Hombro
    if total_hombro != 0:
        clean_steps.append({
            "cmd":            "H+" if total_hombro > 0 else "H-",
            "steps_executed": abs(total_hombro),
            "action":         2 if total_hombro > 0 else 3,
            "angulo_pinza":   None,
        })

    # GRIP 0 al final (siempre)
    if grip_0_step is not None:
        clean_steps.append({
            "cmd":            "GRIP 0",
            "steps_executed": 0,
            "action":         10,
            "angulo_pinza":   0,
        })
    else:
        # Si no había GRIP 0 en la demo, igualmente lo añadimos
        clean_steps.append({
            "cmd":            "GRIP 0",
            "steps_executed": 0,
            "action":         10,
            "angulo_pinza":   0,
        })

    return clean_steps, total_base, total_codo, total_hombro


# ── 4. Construir dataset limpio ───────────────────────────────────────────────
clean_episodes = []

print(f"\n{'─'*70}")
print(f"{'EP_orig':>8}  {'Celda':>6}  {'Exito':>6}  {'Base':>7}  {'Codo':>7}  {'Hombro':>7}  Pasos")
print(f"{'─'*70}")

for ep_num, ep in episodes_filtered:
    clean_steps, b, c, h = limpiar_episodio(ep)

    ep_limpio = {
        # Metadata del objetivo
        "food_target":      ep.get("food_target"),
        "celda_objetivo":   ep.get("celda_objetivo"),
        "fila_objetivo":    ep.get("fila_objetivo"),
        "columna_objetivo": ep.get("columna_objetivo"),
        # Trayectoria comprimida
        "steps":            clean_steps,
        "n_steps":          len(clean_steps),
        # Info de la demo original
        "success":          ep.get("success"),
        "timestamp":        ep.get("timestamp"),
        "home_used":        ep.get("home_used"),
        "ep_original":      ep_num,   # referencia al número original
    }
    clean_episodes.append(ep_limpio)

    exito_str = "True " if ep.get("success") else "False"
    cmds_str  = " ".join(s["cmd"] for s in clean_steps)
    print(f"  EP{ep_num:>4}    {ep.get('celda_objetivo'):>5}   {exito_str}  "
          f"{b:>+7}  {c:>+7}  {h:>+7}  {cmds_str}")

print(f"{'─'*70}")
print(f"\nEpisodios limpios: {len(clean_episodes)}")

# ── 5. Guardar ────────────────────────────────────────────────────────────────
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

# ── 6. Análisis de cobertura de la malla ─────────────────────────────────────
GRID_COLS = 12
GRID_ROWS = 9
TOTAL_CELDAS = GRID_COLS * GRID_ROWS   # 108

celdas_con_demo = set(
    ep["celda_objetivo"] for ep in clean_episodes
    if ep["celda_objetivo"] is not None
)

celdas_sin_demo = sorted(set(range(1, TOTAL_CELDAS + 1)) - celdas_con_demo)

print(f"\n{'='*60}")
print(f"  COBERTURA DE LA MALLA ({GRID_COLS}x{GRID_ROWS} = {TOTAL_CELDAS} celdas)")
print(f"{'='*60}")
print(f"  Celdas CON demo  : {len(celdas_con_demo)}")
print(f"  Celdas SIN demo  : {len(celdas_sin_demo)}")
print()

# Mostrar en formato de malla
print("  Malla (o=con demo, X=sin demo):")
for r in range(GRID_ROWS):
    fila = ""
    for c in range(GRID_COLS):
        num = r * GRID_COLS + c + 1
        fila += "o " if num in celdas_con_demo else "X "
    row_nums = [r * GRID_COLS + c + 1 for c in range(GRID_COLS)]
    print(f"  Fila {r+1:>2} ({row_nums[0]:>3}-{row_nums[-1]:>3}):  {fila}")

print()
print(f"  Celdas SIN demo ({len(celdas_sin_demo)} total):")

# Agrupar por filas para mejor lectura
cnt_demo = Counter(ep["celda_objetivo"] for ep in clean_episodes if ep["celda_objetivo"])
for r in range(GRID_ROWS):
    sin_en_fila = [c for c in celdas_sin_demo
                   if r * GRID_COLS + 1 <= c <= (r + 1) * GRID_COLS]
    if sin_en_fila:
        print(f"    Fila {r+1}: {sin_en_fila}")

print(f"\n  Celdas con MAS de 1 demo:")
for celda, cnt in sorted(cnt_demo.items()):
    if cnt > 1:
        r = (celda - 1) // GRID_COLS
        c = (celda - 1) % GRID_COLS
        print(f"    Celda {celda:>3} (Fila {r+1}, Col {c+1}): {cnt} demos")
