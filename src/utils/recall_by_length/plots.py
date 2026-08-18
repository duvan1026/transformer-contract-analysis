"""
src/utils/recall_by_length/plots.py

Grafica de linea: recall real (calculado sobre el ground truth actual)
por rango de longitud, con el tamaño de muestra N anotado en cada punto
(mismo criterio que el ranking de riesgo por clase: un recall bajo sin
soporte de muestra visible puede ser ruido, no señal).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def graficar_recall_por_longitud(df_recall, output_path, title_suffix="", zona_riesgo=None):
    sns.set_theme(style="white")

    if df_recall.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No hay datos suficientes para calcular recall por longitud.",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df_recall))

    colores_punto = [
        "#C44E52" if (zona_riesgo and row.bin_min == zona_riesgo["bin_min"]) else "#4C72B0"
        for row in df_recall.itertuples()
    ]

    ax.plot(x, df_recall["recall"], color="#4C72B0", linewidth=2, zorder=1)
    ax.scatter(x, df_recall["recall"], color=colores_punto, s=90, zorder=2)

    for xi, row in zip(x, df_recall.itertuples()):
        ax.annotate(f"n={row.n}", (xi, row.recall), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df_recall["bin_label"], rotation=0, fontsize=10)
    ax.set_xlabel("Rango de tokens", fontsize=11, weight="bold")
    ax.set_ylabel("Recall real (%)", fontsize=11, weight="bold")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    if zona_riesgo:
        ax.annotate(
            "Zona de menor acierto\n(calculada con datos actuales)",
            xy=(x[[row.bin_min for row in df_recall.itertuples()].index(zona_riesgo["bin_min"])],
                zona_riesgo["recall"]),
            xytext=(10, -35), textcoords="offset points",
            fontsize=9, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52"),
        )

    ax.set_title(f"Recall real por rango de longitud (verificacion de zona de riesgo) {title_suffix}",
                 fontsize=13, pad=15, weight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_impacto_estimado(tabla, resumen_global, output_path, title_suffix=""):
    """
    Barras agrupadas por clase: metrica actual vs. objetivo (recall para
    positivas, especificidad para NONE), con el resumen de macro-recall y
    accuracy globales como texto en la figura.
    """
    sns.set_theme(style="white")

    tabla_valida = tabla.dropna(subset=["metrica_actual_pct"]).copy()
    if tabla_valida.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No hay datos suficientes para la simulacion de impacto.",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    clases = tabla_valida["clase"].tolist()
    x = range(len(clases))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(clases) * 1.8), 6.5))
    ax.bar([i - ancho / 2 for i in x], tabla_valida["metrica_actual_pct"], width=ancho,
           color="#8172B2", label="Actual (longitud real de inferencia)")

    tiene_objetivo = tabla_valida["metrica_objetivo_pct"].notna()
    ax.bar([i + ancho / 2 for i in x], tabla_valida["metrica_objetivo_pct"].fillna(0), width=ancho,
           color="#55A868", label="Objetivo (si la longitud igualara a train)")

    for i, row in zip(x, tabla_valida.itertuples()):
        ax.text(i - ancho / 2, row.metrica_actual_pct, f"{row.metrica_actual_pct:.0f}%",
                ha="center", va="bottom", fontsize=8)
        if pd.notna(row.metrica_objetivo_pct):
            ax.text(i + ancho / 2, row.metrica_objetivo_pct, f"{row.metrica_objetivo_pct:.0f}%",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(clases, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Recall / Especificidad (%)", fontsize=11, weight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Impacto estimado de igualar la longitud de inferencia a la de train {title_suffix}",
                 fontsize=13, pad=15, weight="bold")

    resumen_txt = (
        f"Macro-Recall (positivas): {resumen_global['macro_recall_actual']}% -> "
        f"{resumen_global['macro_recall_objetivo']}%\n"
        f"Accuracy global estimada: {resumen_global['accuracy_actual']}% -> "
        f"{resumen_global['accuracy_objetivo']}%\n"
        f"(Macro-F1 no estimable con estos datos: requiere precision por clase)"
    )
    ax.text(0.01, -0.30, resumen_txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", family="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()