"""
Pricer hybride credit-action d'un CDS sur obligation convertible.

Source : annexe B du rapport (M2 Actuariat, ISFA), reconstruit a l'identique
depuis le code LaTeX du document (pas d'extraction PDF -> pas d'artefact).

Enchainement :
  1. simulate_GBM_euler       : simulation du sous-jacent (Euler-Maruyama)
  2. quadratic_regression_predict : regression quadratique (coeur de LSM)
  3. LSM_conversion_and_default   : decision de conversion optimale +
                                     probabilite de defaut avant conversion
  4. cds_convertible_spread       : spread CDS-equivalent de la convertible

Necessite : numpy
"""

import numpy as np


# I) FONCTION POUR SIMULER S_t SELON EULER-MARUYAMA

def simulate_GBM_euler(S0, r, sigma, T, N, n_sim, return_paths=False, seed=123):
    np.random.seed(seed)
    dt = T / N

    if return_paths:
        paths = np.zeros((n_sim, N + 1))
        paths[:, 0] = S0
        S = paths[:, 0].copy()
    else:
        S = np.full(n_sim, S0, dtype=float)

    Z = np.random.normal(0, 1, size=(n_sim, N))

    for i in range(N):
        S += r * S * dt + sigma * S * np.sqrt(dt) * Z[:, i]
        if return_paths:
            paths[:, i + 1] = S

    return paths if return_paths else S


# II) FONCTION POUR FAIRE LA REGRESSION SUR (1,X,X^2) DE LA METHODE LSM

def quadratic_regression_predict(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    A = np.column_stack((np.ones_like(X), X, X**2))
    coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    Y_pred = A @ coeffs

    return coeffs, Y_pred


# III) FONCTION POUR CALCULER P*

def LSM_conversion_and_default(lambda_default, Cr, N_nominal, r, sigma, T, D, n_sim, S0, seed=123):
    dt = T / (D - 1)

    # 1) simulation sous-jacent
    S_paths = simulate_GBM_euler(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        N=D - 1,
        n_sim=n_sim,
        return_paths=True,
        seed=seed
    )

    X = Cr * S_paths   # valeur de conversion immediate

    # 2) initialisation a maturite
    # payoff terminal = max(conversion, nominal)
    cashflow = np.maximum(X[:, -1], N_nominal)
    exercise_idx = np.full(n_sim, D - 1, dtype=int)

    # 3) backward induction
    for t in range(D - 2, -1, -1):
        # valeur de conversion immediate
        immediate_ex = X[:, t]

        # valeur de continuation = payoff futur actualise jusqu'a t
        continuation = cashflow * np.exp(-r * dt * (exercise_idx - t))

        # regression sur toutes les trajectoires encore vivantes
        _, continuation_pred = quadratic_regression_predict(immediate_ex, continuation)

        # regle d'exercice
        exercise_now = immediate_ex >= continuation_pred

        # si on exerce a t, on remplace la decision future
        cashflow[exercise_now] = immediate_ex[exercise_now]
        exercise_idx[exercise_now] = t

    # 4) reconstruction de V
    V = np.zeros((n_sim, D))
    for i in range(n_sim):
        V[i, exercise_idx[i]] = cashflow[i]

    # 5) tau_c
    tau_c = exercise_idx * dt

    # 6) defaut
    rng = np.random.default_rng(seed + 1)
    tau_default = rng.exponential(scale=1.0 / lambda_default, size=n_sim)

    indicators = (tau_default < tau_c).astype(int)
    p_star_hat = indicators.mean()

    return V, tau_c, tau_default, p_star_hat, indicators, X


# IV) FONCTION POUR CALCULER LE SPREAD DU CDS SUR MESURE
#     (p_star s'obtient par appel de la fonction precedente)

def cds_convertible_spread(R, lambda_default, p_star):
    """
    Calcule le spread du CDS sur obligation convertible.

    Formule :
        spread = lambda_default * (1 - R) * p_star

    Parametres
    ----------
    R : float
        Taux de recovery, entre 0 et 1
    lambda_default : float
        Intensite de defaut
    p_star : float
        Probabilite de defaut avant conversion

    Retour
    ------
    spread : float
        Spread en valeur decimale
        (ex : 0.012 = 120 bps)
    """
    if not (0 <= R <= 1):
        raise ValueError("R doit etre compris entre 0 et 1.")
    if lambda_default < 0:
        raise ValueError("lambda_default doit etre positif.")
    if not (0 <= p_star <= 1):
        raise ValueError("p_star doit etre compris entre 0 et 1.")

    spread = lambda_default * (1 - R) * p_star
    return spread


if __name__ == "__main__":
    # Exemple d'utilisation avec des parametres illustratifs
    # (remplacer par les valeurs calibrees : cf. calibration_volatilite.py
    #  et le rapport pour le spread CDS 5 ans de reference)
    V, tau_c, tau_default, p_star_hat, indicators, X = LSM_conversion_and_default(
        lambda_default=0.0069,   # intensite de defaut calibree (~0.69%/an)
        Cr=1.0,                  # ratio de conversion
        N_nominal=100.0,         # nominal de l'obligation
        r=0.03,                  # taux sans risque
        sigma=0.25,              # volatilite (cf. calibration_volatilite.py)
        T=5.0,                   # maturite (5 ans, coherent avec le CDS de reference)
        D=60,                    # nombre de dates de conversion (mensuel sur 5 ans)
        n_sim=100_000,
        S0=89.23,                # spot de reference (cf. rapport)
        seed=123
    )

    spread = cds_convertible_spread(R=0.40, lambda_default=0.0069, p_star=p_star_hat)

    print(f"p* (probabilite de defaut avant conversion) = {p_star_hat:.4f}")
    print(f"Spread CDS-equivalent = {spread*10000:.2f} bps")
