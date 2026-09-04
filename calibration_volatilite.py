"""
Calibration de la volatilite historique de l'action BNP Paribas.

Source : annexe A du rapport (M2 Actuariat, ISFA), reconstruit a l'identique
depuis le code LaTeX du document (pas d'extraction PDF -> pas d'artefact).

Ce script correspond a l'enchainement de plusieurs cellules Colab :
  1. Chargement des donnees + 4 estimateurs de volatilite (close-to-close,
     EWMA, Parkinson, Garman-Klass)
  2. Volatilite glissante (fenetres 21/42/63 jours)
  3. Tests de stabilite de la variance (Fisher, Levene, balayage supF)

Necessite : numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from google.colab import files

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

# =====================================================================
# 1) CHARGEMENT DES DONNEES ET ESTIMATEURS DE VOLATILITE
# =====================================================================

# --- Upload ---
print(">>> Upload bnp_data.xlsx :")
uploaded = files.upload()
fname = list(uploaded.keys())[0]

# --- Nettoyage ---
def parse_date(x):
    if isinstance(x, str):
        return pd.to_datetime(x, format="%d/%m/%Y", errors="coerce")
    return pd.to_datetime(x)

def to_float(x):
    if isinstance(x, str):
        return float(x.replace(",", ".").replace(" ", "")
                      .replace("BP", "").replace("%", ""))
    return float(x)

df = pd.read_excel(fname)
df["Date"]  = df["Date"].apply(parse_date)
df["close"] = df["Dernier"].apply(to_float)
df["high"]  = df["+ haut"].apply(to_float)
df["low"]   = df["+ bas"].apply(to_float)
df["open"]  = df["Ouverture"].apply(to_float)
df = df.sort_values("Date").reset_index(drop=True)

print(f"\nPeriode  : {df['Date'].min().date()} -> {df['Date'].max().date()}")
print(f"Seances  : {len(df)}")
print(f"Cours min/max : {df['close'].min():.2f} / {df['close'].max():.2f} EUR")
print(f"Cours au dernier jour : {df['close'].iloc[-1]:.2f} EUR")

# --- Log-rendements ---
df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
ret = df["log_ret"].dropna().values
n   = len(ret)

# ---- Estimateur 1 : close-to-close (equiponderé) ----
sigma_c2c = ret.std(ddof=1) * np.sqrt(252)
se_c2c    = sigma_c2c / np.sqrt(2*(n-1))
ci95      = (sigma_c2c - 1.96*se_c2c, sigma_c2c + 1.96*se_c2c)

# ---- Estimateur 2 : EWMA (RiskMetrics lambda=0.94) ----
lam = 0.94
w   = (1-lam) * lam**np.arange(n-1, -1, -1); w /= w.sum()
sigma_ewma = np.sqrt(np.sum(w * (ret - np.sum(w*ret))**2) * 252)

# ---- Estimateur 3 : Parkinson (1980) — exploite High/Low ----
ln_hl      = np.log(df["high"] / df["low"]).dropna().values
sigma_park = np.sqrt(np.mean(ln_hl**2) / (4*np.log(2))) * np.sqrt(252)

# ---- Estimateur 4 : Garman-Klass (1980) — exploite OHLC ----
ln_co    = np.log(df["close"] / df["open"]).dropna().values
ln_hl_gk = ln_hl[:len(ln_co)]
gk       = 0.5*ln_hl_gk**2 - (2*np.log(2)-1)*ln_co**2
sigma_gk = np.sqrt(np.mean(gk) * 252)

# ---- Synthese ----
print("\n" + "="*55)
print(f"Close-to-close  : {sigma_c2c*100:.2f}%   IC95 [{ci95[0]*100:.1f}%, {ci95[1]*100:.1f}%]")
print(f"EWMA (lambda=0.94)   : {sigma_ewma*100:.2f}%")
print(f"Parkinson (HL)  : {sigma_park*100:.2f}%")
print(f"Garman-Klass    : {sigma_gk*100:.2f}%")

SIGMA_RETENUE  = float(np.median([sigma_c2c, sigma_ewma, sigma_park, sigma_gk]))
SIGMA_ARRONDIE = round(SIGMA_RETENUE, 2)
print(f"\nMediane des 4   : {SIGMA_RETENUE*100:.2f}%")
print(f">>> sigma retenue pour le pricer = {SIGMA_ARRONDIE*100:.0f}%")
print("="*55)

# ---- Figure 1 : cours + log-rendements ----
fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
axes[0].plot(df["Date"], df["close"], color="#1f4e79", lw=1.3)
axes[0].fill_between(df["Date"], df["low"], df["high"],
                     color="#1f4e79", alpha=0.12)
axes[0].set_ylabel("Cours (EUR)")
axes[0].set_title(f"BNP Paribas — cours action, "
                  f"{df['Date'].min().date()} au {df['Date'].max().date()}")
axes[1].bar(df["Date"], df["log_ret"]*100,
            color="#c0392b", alpha=0.8, width=0.8)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_ylabel("Log-rendement (%)")
axes[1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("fig_42_cours_returns.pdf", bbox_inches="tight")
plt.show()

# ---- Figure 2 : comparaison des 4 estimateurs ----
fig, ax = plt.subplots(figsize=(8, 4.2))
labels = ["Close-to-close\n(equiponderé)", "EWMA\n(lambda=0.94)",
          "Parkinson\n(HL)", "Garman–Klass\n(OHLC)"]
vals   = [sigma_c2c*100, sigma_ewma*100, sigma_park*100, sigma_gk*100]
colors = ["#2c3e50", "#34495e", "#2980b9", "#27ae60"]
bars   = ax.bar(labels, vals, color=colors, alpha=0.85)
ax.axhline(SIGMA_ARRONDIE*100, color="#c0392b", linestyle="--",
           lw=1.4, label=f"sigma retenue = {SIGMA_ARRONDIE*100:.0f}%")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.3,
            f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("Volatilite annualisee (%)")
ax.set_title("Comparaison des estimateurs — action BNP Paribas")
ax.set_ylim(0, max(vals)*1.18)
ax.legend()
plt.tight_layout()
plt.savefig("fig_42_estimateurs.pdf", bbox_inches="tight")
plt.show()


# =====================================================================
# 2) VOLATILITE GLISSANTE (fenetres 21 / 42 / 63 jours)
# =====================================================================

# --- Parametre : taille de la fenetre glissante (en jours) ---
fenetres = {"21j (1 mois)": 21, "42j (2 mois)": 42, "63j (3 mois)": 63}

fig, ax = plt.subplots(figsize=(11, 5))

colors = ["#2980b9", "#27ae60", "#8e44ad"]
for (label, w), col in zip(fenetres.items(), colors):
    # rolling std sur w jours, annualisee
    vol_roll = (
        df["log_ret"]
        .rolling(window=w, min_periods=w)
        .std(ddof=1)
        * np.sqrt(252)
        * 100   # en %
    )
    ax.plot(df["Date"], vol_roll, lw=1.4, color=col, label=f"Fenetre {label}")

# Valeur retenue (issue de la partie 1 ci-dessus)
ax.axhline(SIGMA_RETENUE * 100, color="#c0392b", lw=1.5,
           linestyle="--", label=f"sigma retenue = {SIGMA_RETENUE*100:.0f}%")

# Vol full-sample (reference)
vol_full = df["log_ret"].std(ddof=1) * np.sqrt(252) * 100
ax.axhline(vol_full, color="black", lw=1, linestyle=":",
           label=f"sigma full-sample = {vol_full:.1f}%")

ax.set_ylabel("Volatilite annualisee (%)")
ax.set_xlabel("Date")
ax.set_title("Volatilite glissante des log-rendements de l'action BNP Paribas")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("fig_vol_glissante.pdf", bbox_inches="tight")
plt.show()

# --- Statistiques descriptives par sous-periode (trim.) ---
print("\n=== Volatilite par trimestre ===")
df_tmp = df.dropna(subset=["log_ret"]).copy()
df_tmp["trimestre"] = df_tmp["Date"].dt.to_period("Q")
for q, grp in df_tmp.groupby("trimestre"):
    r = grp["log_ret"].values
    sig = r.std(ddof=1) * np.sqrt(252) * 100
    print(f"  {q}  n={len(r):3d}  sigma = {sig:.2f}%")


# =====================================================================
# 3) TESTS DE STABILITE DE LA VARIANCE
# =====================================================================

ret_clean = df["log_ret"].dropna().values
n = len(ret_clean)
dates_clean = df["Date"].dropna().iloc[1:].reset_index(drop=True)  # dates des log-returns

# ---- 1. TEST F (comparaison 2 sous-periodes : premiere vs seconde moitie) ----
print("=" * 60)
print("1. TEST F DE FISHER — COUPURE A LA MEDIANE")
print("=" * 60)

mid = n // 2
r1, r2 = ret_clean[:mid], ret_clean[mid:]
s1, s2 = r1.std(ddof=1), r2.std(ddof=1)
n1, n2 = len(r1), len(r2)

F_stat = (s1**2) / (s2**2)   # si s1 > s2, sinon inverser
df1, df2 = n1 - 1, n2 - 1
# p-value bilaterale
p_val_F = 2 * min(stats.f.cdf(F_stat, df1, df2),
                  1 - stats.f.cdf(F_stat, df1, df2))

print(f"  Sous-periode 1 : n={n1}, sigma = {s1*np.sqrt(252)*100:.2f}%")
print(f"  Sous-periode 2 : n={n2}, sigma = {s2*np.sqrt(252)*100:.2f}%")
print(f"  Statistique F  = {F_stat:.4f}")
print(f"  p-value        = {p_val_F:.4f}")
if p_val_F < 0.05:
    print("  -> H0 rejetee a 5% : les variances sont significativement differentes")
    print("    (la volatilite n'est PAS stable sur toute la periode)")
else:
    print("  -> H0 non rejetee a 5% : pas de preuve de rupture de variance")
    print("    (la volatilite est stable, la fenetre annuelle est coherente)")


# ---- 2. TEST DE LEVENE — k = 4 TRIMESTRES ----
print("\n" + "=" * 60)
print("2. TEST DE LEVENE — 4 SOUS-PERIODES (trimestres)")
print("=" * 60)

df_tmp = df.dropna(subset=["log_ret"]).copy()
df_tmp["trimestre"] = df_tmp["Date"].dt.to_period("Q")
groupes = [grp["log_ret"].values for _, grp in df_tmp.groupby("trimestre")]

# Afficher les sigma par trimestre
for (q, _), g in zip(df_tmp.groupby("trimestre"), groupes):
    print(f"  {q}  n={len(g):3d}  sigma = {g.std(ddof=1)*np.sqrt(252)*100:.2f}%")

lev_stat, lev_p = stats.levene(*groupes, center="median")  # center=median = Brown-Forsythe, robuste
print(f"\n  Statistique de Levene (Brown-Forsythe) = {lev_stat:.4f}")
print(f"  p-value                                 = {lev_p:.4f}")
if lev_p < 0.05:
    print("  -> H0 rejetee a 5% : variance heterogene entre trimestres")
else:
    print("  -> H0 non rejetee a 5% : variance homogene entre trimestres")


# ---- 3. BALAYAGE DU POINT DE RUPTURE (supF) ----
print("\n" + "=" * 60)
print("3. BALAYAGE DU POINT DE RUPTURE (supF test)")
print("=" * 60)
print("   Teste toutes les coupures possibles (excl. 15% extremes)")

trim = int(0.15 * n)   # exclure les 15% de chaque cote
F_vals, break_idx = [], []

for tau in range(trim, n - trim):
    r_a = ret_clean[:tau]
    r_b = ret_clean[tau:]
    s_a, s_b = r_a.std(ddof=1), r_b.std(ddof=1)
    if s_b == 0: continue
    F_t = (s_a**2) / (s_b**2)
    F_t = max(F_t, 1/F_t)   # toujours >= 1
    F_vals.append(F_t)
    break_idx.append(tau)

best_tau = break_idx[np.argmax(F_vals)]
best_F   = max(F_vals)
best_date = dates_clean.iloc[best_tau].date()

# p-value approximative au point de rupture optimal
p_approx = 2 * (1 - stats.f.cdf(best_F, best_tau-1, n-best_tau-1))
print(f"\n  Point de rupture optimal : index {best_tau} -> {best_date}")
print(f"  supF = {best_F:.4f}  (p-value approx. = {p_approx:.4f})")
print(f"  sigma avant : {ret_clean[:best_tau].std(ddof=1)*np.sqrt(252)*100:.2f}%")
print(f"  sigma apres : {ret_clean[best_tau:].std(ddof=1)*np.sqrt(252)*100:.2f}%")

# ---- Graphe du balayage ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Courbe supF
axes[0].plot(dates_clean.iloc[break_idx], F_vals, color="#2c3e50", lw=1.3)
axes[0].axvline(dates_clean.iloc[best_tau], color="#c0392b",
                linestyle="--", lw=1.3, label=f"Rupture optimale\n{best_date}")
axes[0].axhline(stats.f.ppf(0.975, n//2, n//2), color="#e67e22",
                linestyle=":", lw=1.2, label="Seuil 5% (approx.)")
axes[0].set_title("Statistique F par point de coupure (supF)")
axes[0].set_ylabel("Statistique F")
axes[0].set_xlabel("Date de coupure")
axes[0].legend(fontsize=9)
axes[0].tick_params(axis='x', labelsize=8, rotation=45)
# Volatilite par trimestre (barres)
vols_trim = [g.std(ddof=1)*np.sqrt(252)*100 for g in groupes]
labels_trim = [str(q) for q, _ in df_tmp.groupby("trimestre")]
axes[1].bar(labels_trim, vols_trim, color="#2980b9", alpha=0.8)
axes[1].axhline(SIGMA_RETENUE * 100, color="#c0392b", linestyle="--",
                lw=1.3, label=f"sigma retenue = {SIGMA_RETENUE*100:.0f}%")
axes[1].set_title("Volatilite annualisee par trimestre")
axes[1].set_ylabel("sigma annualisee (%)")
axes[1].set_xlabel("Trimestre")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig("fig_stabilite_variance.pdf", bbox_inches="tight")
plt.show()

# ---- Conclusion synthetique ----
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print(f"  Test F (mediane)     : p = {p_val_F:.4f}  ->  "
      + ("instable" if p_val_F < 0.05 else "stable"))
print(f"  Test Levene (4 trim) : p = {lev_p:.4f}  ->  "
      + ("instable" if lev_p < 0.05 else "stable"))
print(f"  supF (rupture opt.)  : p ~ {p_approx:.4f}  ->  "
      + ("instable" if p_approx < 0.05 else "stable"))
print()
if max(p_val_F, lev_p) > 0.05:
    print("  La variance est homogene sur la periode -> la fenetre annuelle")
    print("  est justifiee : aucun regime distinct n'est detecte.")
else:
    print("  La variance est heterogene -> la fenetre annuelle integre")
    print("  plusieurs regimes. Envisager une fenetre plus courte")
    print(f"  (apres {best_date}) ou un estimateur EWMA.")
