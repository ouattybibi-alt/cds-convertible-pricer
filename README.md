# CDS Convertible Pricer

Hybrid credit–equity pricing of a Credit Default Swap (CDS) on a convertible bond, combining a reduced-form default intensity model with Monte-Carlo simulation of the underlying equity.

Academic project (M2 Actuariat, ISFA), Co-authored with C. A. D. Kouamé and S. Ouattara.

## What this does

- Calibrates the default intensity from BNP Paribas' 5-year CDS spread (reduced-form model, Beta-distributed recovery) and historical volatility from 254 daily equity observations.
- Simulates the equity price path via an Euler–Maruyama discretisation of a geometric Brownian motion.
- Determines the optimal conversion decision using Longstaff–Schwartz least-squares Monte-Carlo (quadratic regression for the continuation value).
- Estimates the CDS-equivalent spread on the convertible and the probability of default before conversion.
- Computes a full sensitivity surface: Delta, Vega, Rho, Theta, Lambda, Kappa.

## Files

| File | Content |
|---|---|
| `calibration_volatilite.py` | Data cleaning, historical volatility estimation (close-to-close, Parkinson, Garman–Klass), rolling-window analysis, variance stability tests |
| `pricer_cds.py` | Core pricer: GBM simulation, Longstaff–Schwartz conversion logic, CDS-equivalent spread calculation |

## Status

Reconstructed directly from the LaTeX source of the report appendix (not from PDF text extraction, so no OCR/formatting artefacts). Syntax-checked; `pricer_cds.py` runs end-to-end standalone (see `__main__` block for a worked example). `calibration_volatilite.py` requires the original `bnp_data.xlsx` market data file and a Colab/Jupyter environment (`google.colab.files.upload`) to run as-is — swap that cell for a local file read (`pd.read_excel("bnp_data.xlsx")`) to run outside Colab.

## Requirements

```
numpy
pandas
matplotlib
scipy
```

## Full write-up

The complete methodology, market data sources, and results are detailed in the project report (not included here — contact for the full PDF).
