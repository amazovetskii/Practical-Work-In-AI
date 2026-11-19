import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_STYLES = {
    "CVEM b_ref": {"color": "tab:blue", "marker": "o"},
    "CVEM b_hat": {"color": "tab:orange", "marker": "s"},
}


def _mc_reference_mse(df_B: pd.DataFrame, normalize_by: str) -> float:
    """MSE of plain MC for one budget B, used as the denominator of the ratio."""
    mc_row = df_B[df_B["method"] == "MC"].iloc[0]
    if normalize_by == "theory":
        return float(mc_row["MSE_theory"])
    if normalize_by == "empirical":
        return float(mc_row["MSE"])
    raise ValueError("normalize_by must be 'theory' or 'empirical'")


def plot_mse_ratio_vs_kappa(results: pd.DataFrame,
                            methods: tuple[str, ...] = ("CVEM b_ref", "CVEM b_hat"),
                            normalize_by: str = "theory",
                            n_se: float = 2.0,
                            show_cv_level: bool = True,
                            figsize_per_panel: tuple[float, float] = (4.2, 3.6),
                            title: str | None = "Relative MSE of CVEM vs. plain MC at fixed budget B",
                            kappa_star: float | None = None):
    """
    Parameters
    ----------
    results : DataFrame with columns B, kappa, method, MSE, SE, MSE_theory
        (one row per (B, method, kappa); MC and CV rows have kappa = NaN)
    methods : CVEM methods to plot against kappa
    normalize_by : "theory" divides everything by the theoretical MSE of plain MC
        (deterministic denominator, so the band is exactly n_se * SE / MSE_MC);
        "empirical" divides by the empirical MSE of plain MC (the band then
        ignores the uncertainty of the denominator)
    n_se : half-width of the band in standard errors
    show_cv_level : draw the level of classical CV with known mu_C (b_ref),
        (same budget B, no auxiliary experiment needed)
    title : figure title (None for no title)
    kappa_star : theoretically optimal ratio kappa* = N'/N, drawn as a vertical
        gray dashed line in every panel (None for no line)

    Returns
    -------
    fig, axes : one panel per budget B with MSE / MSE_MC against kappa (log-log),
        empirical points with +- n_se * SE bands and the theoretical curve
    """
    B_values = np.sort(results["B"].unique())
    fig, axes = plt.subplots(1, len(B_values),
                             figsize=(figsize_per_panel[0] * len(B_values), figsize_per_panel[1]),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, B in zip(axes, B_values):
        df_B = results[results["B"] == B]
        mse_mc = _mc_reference_mse(df_B, normalize_by)

        for method in methods:
            df_m = df_B[df_B["method"] == method].sort_values("kappa")
            kappa = df_m["kappa"].to_numpy()
            ratio = df_m["MSE"].to_numpy() / mse_mc
            half_band = n_se * df_m["SE"].to_numpy() / mse_mc
            ratio_theory = df_m["MSE_theory"].to_numpy() / mse_mc
            style = METHOD_STYLES.get(method, {})

            lower = ratio - half_band
            # on a log axis a non-positive lower bound cannot be drawn: clip it
            lower = np.where(lower > 0, lower, ratio * 1e-3)
            ax.fill_between(kappa, lower, ratio + half_band,
                            color=style.get("color"), alpha=0.2, linewidth=0)
            ax.plot(kappa, ratio, linestyle="none", markersize=4,
                    label=f"{method} (empirical)", **style)
            if method.endswith("b_ref"):
                ax.plot(kappa, ratio_theory, color="black", linewidth=1.2,
                        label="theory")

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="plain MC")
        if show_cv_level:
            cv_row = df_B[df_B["method"] == "CV b_ref"]
            if not cv_row.empty:
                ax.axhline(float(cv_row["MSE"].iloc[0]) / mse_mc, color="tab:green",
                           linestyle=":", linewidth=1.2, label=r"CV, known $\mu_C$")

        if kappa_star is not None:
            ax.axvline(kappa_star, color="gray", linestyle="--", linewidth=1,
                       label=rf"$\kappa^* = {kappa_star:.3g}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa = N'/N$")
        ax.set_title(f"$B = {int(B)}$")
        ax.grid(True, which="major", alpha=0.3)

    denom = r"\mathrm{MSE}_{\mathrm{MC}}^{\mathrm{theory}}" if normalize_by == "theory" \
        else r"\widehat{\mathrm{MSE}}_{\mathrm{MC}}"
    axes[0].set_ylabel(rf"$\mathrm{{MSE}} / {denom}$  (lower is better)")
    axes[-1].legend(fontsize=8, loc="best")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
