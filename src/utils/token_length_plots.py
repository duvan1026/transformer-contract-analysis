"""
src/utils/token_length_plots.py

Generacion de graficos de longitud (boxplot, ECDF).
Estilo alineado a visualization_utils.py: seaborn sobre matplotlib,
guardado con dpi=300 y bbox_inches='tight', firma (df, output_path, title_suffix="").

Las lineas de referencia de "zona de riesgo" usan una etiqueta descriptiva,
sin IDs de hallazgo (ej. "H04"), porque el rango de tokens que delimitan
es un valor configurable (TOKEN_LENGTH_RISK_ZONE_TOKENS en text_standards.py)
que puede cambiar si se rehace el analisis de errores.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def graficar_boxplot(df_todas_fuentes: pd.DataFrame, output_path, title_suffix: str = ""):
    """
    Boxplot de tokens por clase, con una caja por fuente
    (train/val/test/inferencia) agrupadas lado a lado dentro de cada clase.
    """
    sns.set_theme(style="white")

    clases = sorted(df_todas_fuentes["clase"].unique())
    fuentes = ["train", "val", "test", "inferencia"]
    fuentes_presentes = [f for f in fuentes if f in df_todas_fuentes["fuente"].unique()]

    fig, ax = plt.subplots(figsize=(max(10, len(clases) * 2.2), 6))

    posiciones, datos, etiquetas_x = [], [], []
    ancho = 0.8 / max(len(fuentes_presentes), 1)

    for i, clase in enumerate(clases):
        for j, fuente in enumerate(fuentes_presentes):
            subset = df_todas_fuentes[
                (df_todas_fuentes["clase"] == clase) & (df_todas_fuentes["fuente"] == fuente)
            ]["n_tokens"]
            if len(subset) == 0:
                continue
            pos = i + (j - len(fuentes_presentes) / 2) * ancho
            posiciones.append(pos)
            datos.append(subset.values)
            etiquetas_x.append(fuente)

    bp = ax.boxplot(datos, positions=posiciones, widths=ancho * 0.9, patch_artist=True)

    colores = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52", "inferencia": "#8172B2"}
    for patch, etiqueta in zip(bp["boxes"], etiquetas_x):
        patch.set_facecolor(colores.get(etiqueta, "#999999"))
        patch.set_alpha(0.7)

    ax.set_xticks(range(len(clases)))
    ax.set_xticklabels(clases, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Tokens (BERT)", fontsize=12, labelpad=15, weight="bold")
    ax.set_title(f"Distribucion de longitud en tokens por clase y fuente {title_suffix}",
                 fontsize=14, pad=20, weight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colores[f], alpha=0.7) for f in fuentes_presentes]
    ax.legend(handles, fuentes_presentes, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_ecdf(df_todas_fuentes: pd.DataFrame, output_path, title_suffix: str = "",
                   umbral_palabras: int = 200, zona_riesgo_tokens=(400, 600)):
    """
    ECDF de tokens, train vs. inferencia, en PEQUEÑOS MULTIPLOS: un panel
    por clase (las 5 positivas + NONE), en vez de una sola curva agregada
    para "clases positivas".

    Se descarto la version agregada: promediar las 5 clases en una sola
    curva esconde variacion real entre ellas (en los datos del proyecto,
    el % de fragmentos en la zona de riesgo va de ~18% a ~27% segun la
    clase) y ademas queda dominada por las clases con mas muestras,
    subrepresentando visualmente a las que tienen menos. Un panel por
    clase permite comparar la brecha train-vs-inferencia de cada una
    sin que ninguna eclipse a las demas.
    """
    sns.set_theme(style="white")

    clases_positivas = sorted(c for c in df_todas_fuentes["clase"].unique() if c != "none")
    tiene_none = "none" in df_todas_fuentes["clase"].unique()
    clases_ordenadas = clases_positivas + (["none"] if tiene_none else [])

    n_paneles = len(clases_ordenadas)
    n_cols = 3
    n_filas = (n_paneles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_filas, n_cols, figsize=(15, 5.2 * n_filas))
    axes = axes.flatten() if n_paneles > 1 else [axes]

    etiqueta_zona_riesgo = f"Rango de menor acierto observado ({zona_riesgo_tokens[0]}-{zona_riesgo_tokens[1]} tokens)"

    # Limite de eje X compartido entre los paneles de clases positivas,
    # para que sean comparables entre si. NONE mantiene su propio rango
    # (su escala es varias veces mayor, ver panel derecho de balance_clases).
    datos_positivas = df_todas_fuentes[
        (df_todas_fuentes["clase"] != "none") & (df_todas_fuentes["fuente"].isin(["train", "inferencia"]))
    ]
    xlim_positivas = datos_positivas["n_tokens"].max() * 1.05 if not datos_positivas.empty else None

    for idx, clase in enumerate(clases_ordenadas):
        ax = axes[idx]
        for fuente, color in [("train", "#4C72B0"), ("inferencia", "#8172B2")]:
            subset = df_todas_fuentes[
                (df_todas_fuentes["fuente"] == fuente) & (df_todas_fuentes["clase"] == clase)
            ]["n_tokens"]
            if len(subset) == 0:
                continue
            valores_ordenados = subset.sort_values().values
            y = (pd.Series(range(1, len(valores_ordenados) + 1)) / len(valores_ordenados)).values
            ax.plot(valores_ordenados, y, label=fuente, color=color, linewidth=2)

        ax.axvline(zona_riesgo_tokens[0], color="grey", linestyle="--", linewidth=1)
        ax.axvline(zona_riesgo_tokens[1], color="grey", linestyle="--", linewidth=1)

        if clase != "none" and xlim_positivas:
            ax.set_xlim(0, xlim_positivas)

        ax.set_title(clase, fontsize=11, weight="bold")
        ax.set_xlabel("Tokens (BERT)", fontsize=9)
        ax.grid(alpha=0.3)
        if idx % n_cols == 0:
            ax.set_ylabel("Proporcion acumulada", fontsize=9)

    # Apagar paneles sobrantes de la grilla (si n_paneles no llena filas x cols)
    for idx in range(n_paneles, len(axes)):
        axes[idx].axis("off")

    # Leyenda unica para toda la figura, evita repetirla en cada panel
    handles = [
        plt.Line2D([0], [0], color="#4C72B0", linewidth=2, label="train"),
        plt.Line2D([0], [0], color="#8172B2", linewidth=2, label="inferencia"),
        plt.Line2D([0], [0], color="grey", linestyle="--", linewidth=1, label=etiqueta_zona_riesgo),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=10, frameon=False)

    fig.suptitle(f"ECDF de longitud en tokens por clase: train vs. inferencia {title_suffix}",
                 fontsize=14, weight="bold", y=1.06)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


_COLORES_FUENTE = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52", "inferencia": "#8172B2"}

# Paleta fija por clase, para que el color de cada clase sea el mismo en
# las gráficas de composición apiladas (con y sin NONE) y no cambie según
# qué clases estén presentes en cada llamada.
_COLORES_CLASE = {
    "audit rights": "#4C72B0",
    "governing law": "#55A868",
    "limitation of liability": "#C44E52",
    "non-compete": "#8172B2",
    "termination for convenience": "#CCB974",
    "none": "#BFBFBF",
}


def graficar_balance_clases(df_conteos: pd.DataFrame, output_path, title_suffix: str = ""):
    """
    Volumen bruto de spans/fragmentos por clase y fuente, en DOS PANELES
    con escala lineal independiente (en vez de un unico eje logaritmico).

    Se descarto la escala logaritmica: aunque resuelve el problema de que
    NONE (miles) y las clases positivas (decenas) quepan en la misma
    imagen, la altura de una barra en escala log ya no se lee como
    "el doble de alta = el doble de grande" -- rompe la intuicion visual
    por defecto de un grafico de barras. Separar en dos paneles lineales
    evita ese problema: cada panel se lee de forma directa, sin que el
    lector necesite interpretar el eje.
    """
    sns.set_theme(style="white")

    clases_positivas = sorted(c for c in df_conteos["clase"].unique() if c != "none")
    fuentes_presentes = [f for f in ["train", "val", "test", "inferencia"]
                         if f in df_conteos["fuente"].unique()]

    fig, (ax_izq, ax_der) = plt.subplots(
        1, 2, figsize=(max(12, len(clases_positivas) * 2.2 + 3), 6),
        gridspec_kw={"width_ratios": [len(clases_positivas), 1]}
    )

    # --- Panel izquierdo: clases positivas, escala lineal ---
    ancho = 0.8 / max(len(fuentes_presentes), 1)
    for j, fuente in enumerate(fuentes_presentes):
        valores = []
        for clase in clases_positivas:
            fila = df_conteos[(df_conteos["clase"] == clase) & (df_conteos["fuente"] == fuente)]
            valores.append(fila["cantidad"].iloc[0] if not fila.empty else 0)
        posiciones = [i + (j - len(fuentes_presentes) / 2) * ancho for i in range(len(clases_positivas))]
        barras = ax_izq.bar(posiciones, valores, width=ancho * 0.9,
                             color=_COLORES_FUENTE.get(fuente, "#999999"), label=fuente)
        for barra, valor in zip(barras, valores):
            if valor > 0:
                ax_izq.text(barra.get_x() + barra.get_width() / 2, valor, str(int(valor)),
                            ha="center", va="bottom", fontsize=8)

    ax_izq.set_xticks(range(len(clases_positivas)))
    ax_izq.set_xticklabels(clases_positivas, rotation=30, ha="right", fontsize=10)
    ax_izq.set_ylabel("Cantidad de fragmentos", fontsize=12, labelpad=15, weight="bold")
    ax_izq.set_title("Clases positivas", fontsize=12, weight="bold")
    ax_izq.legend(loc="upper right", fontsize=9)

    # --- Panel derecho: NONE, escala lineal propia (no comparte eje con el izquierdo) ---
    valores_none = []
    for fuente in fuentes_presentes:
        fila = df_conteos[(df_conteos["clase"] == "none") & (df_conteos["fuente"] == fuente)]
        valores_none.append(fila["cantidad"].iloc[0] if not fila.empty else 0)

    barras_none = ax_der.bar(fuentes_presentes, valores_none,
                              color=[_COLORES_FUENTE.get(f, "#999999") for f in fuentes_presentes])
    for barra, valor in zip(barras_none, valores_none):
        if valor > 0:
            ax_der.text(barra.get_x() + barra.get_width() / 2, valor, str(int(valor)),
                        ha="center", va="bottom", fontsize=9)

    ax_der.set_xticks(range(len(fuentes_presentes)))
    ax_der.set_xticklabels(fuentes_presentes, rotation=30, ha="right", fontsize=10)
    ax_der.set_title("Clase NONE (eje propio)", fontsize=12, weight="bold")

    fig.suptitle(f"Volumen de fragmentos por clase y fuente {title_suffix}",
                 fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_composicion_relativa(df_composicion: pd.DataFrame, output_path,
                                   title_suffix: str = "", incluir_none: bool = True):
    """
    Barras agrupadas por clase (una barra por fuente), con el porcentaje
    como altura directa desde 0% y la etiqueta de "%" siempre visible
    encima de cada barra.

    Se descarto la barra 100% apilada: en un apilado, cada segmento
    (salvo el primero y el ultimo) empieza en una altura distinta segun
    cuanto ocupan los segmentos debajo, asi que comparar la misma clase
    entre dos fuentes exige restar alturas mentalmente. Con barras
    agrupadas, todas comparten la base en 0%, que es la codificacion
    visual mas facil de comparar con precision (Cleveland & McGill, 1984).
    Ademas, al no compartir columna con otras 5 clases, cada barra tiene
    espacio propio para mostrar su etiqueta de porcentaje sin solaparse.
    """
    sns.set_theme(style="white")

    fuentes_presentes = [f for f in ["train", "val", "test", "inferencia"]
                         if f in df_composicion["fuente"].unique()]
    clases_presentes = sorted(df_composicion["clase"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(clases_presentes) * 2.2), 6))
    ancho = 0.8 / max(len(fuentes_presentes), 1)

    for j, fuente in enumerate(fuentes_presentes):
        valores = []
        for clase in clases_presentes:
            fila = df_composicion[(df_composicion["clase"] == clase) & (df_composicion["fuente"] == fuente)]
            valores.append(fila["porcentaje"].iloc[0] if not fila.empty else 0)
        posiciones = [i + (j - len(fuentes_presentes) / 2) * ancho for i in range(len(clases_presentes))]
        barras = ax.bar(posiciones, valores, width=ancho * 0.9,
                         color=_COLORES_FUENTE.get(fuente, "#999999"), label=fuente)
        for barra, valor in zip(barras, valores):
            ax.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.1f}%",
                    ha="center", va="bottom", fontsize=7, rotation=90 if valor < 8 else 0)

    ax.set_xticks(range(len(clases_presentes)))
    ax.set_xticklabels(clases_presentes, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Porcentaje dentro de la fuente (%)", fontsize=12, labelpad=15, weight="bold")
    subtitulo = "incluye NONE" if incluir_none else "solo clases positivas"
    ax.set_title(f"Composicion relativa por fuente ({subtitulo}) {title_suffix}",
                 fontsize=14, pad=20, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(df_composicion["porcentaje"].max() * 1.15, 10))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_riesgo_truncamiento(df_riesgo: pd.DataFrame, output_path, title_suffix: str = ""):
    """
    Barras agrupadas (clase x fuente), solo train/val/test, con la cantidad
    de spans que superan el limite fisico de tokens del modelo (MAX_TOKENS)
    y por tanto se truncan/descartan silenciosamente al entrenar.
    """
    sns.set_theme(style="white")

    if df_riesgo.empty or df_riesgo["cantidad_excede"].sum() == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Ningun fragmento de train/val/test supera el limite de tokens.",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    clases = sorted(df_riesgo["clase"].unique())
    fuentes_presentes = [f for f in ["train", "val", "test"] if f in df_riesgo["fuente"].unique()]

    fig, ax = plt.subplots(figsize=(max(9, len(clases) * 2), 5.5))
    ancho = 0.8 / max(len(fuentes_presentes), 1)

    for j, fuente in enumerate(fuentes_presentes):
        valores = []
        for clase in clases:
            fila = df_riesgo[(df_riesgo["clase"] == clase) & (df_riesgo["fuente"] == fuente)]
            valores.append(fila["cantidad_excede"].iloc[0] if not fila.empty else 0)
        posiciones = [i + (j - len(fuentes_presentes) / 2) * ancho for i in range(len(clases))]
        barras = ax.bar(posiciones, valores, width=ancho * 0.9,
                         color=_COLORES_FUENTE.get(fuente, "#999999"), label=fuente)
        for barra, valor in zip(barras, valores):
            if valor > 0:
                ax.text(barra.get_x() + barra.get_width() / 2, valor, str(int(valor)),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(clases)))
    ax.set_xticklabels(clases, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Cantidad de spans que exceden el limite", fontsize=11, labelpad=15, weight="bold")
    ax.set_title(f"Spans en riesgo de truncamiento al entrenar (>MAX_TOKENS) {title_suffix}",
                 fontsize=13, pad=20, weight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_riesgo_por_clase(df_riesgo_inferencia: pd.DataFrame, output_path, title_suffix: str = ""):
    """
    Barras horizontales, una por clase, ordenadas de mayor a menor
    porcentaje de fragmentos reales de inferencia que caen en la zona de
    riesgo (rango de tokens de menor acierto observado). Es el resumen
    mas directamente accionable de todo el analisis: responde "que clase
    se beneficiaria mas de una refragmentacion (F02/F03)".
    """
    sns.set_theme(style="white")

    if df_riesgo_inferencia.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No hay datos de inferencia disponibles.", ha="center", va="center", fontsize=11)
        ax.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    df = df_riesgo_inferencia.sort_values("porcentaje_riesgo", ascending=True)  # ascendente para barh (arriba = mayor)

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(df) * 0.9)))
    colores = [_COLORES_CLASE.get(c, "#999999") for c in df["clase"]]
    barras = ax.barh(df["clase"], df["porcentaje_riesgo"], color=colores)

    for barra, (_, fila) in zip(barras, df.iterrows()):
        etiqueta = f"{fila['porcentaje_riesgo']:.1f}%  ({int(fila['cantidad_riesgo'])} de {int(fila['cantidad_fragmentos'])} fragmentos)"
        ax.text(barra.get_width() + max(df["porcentaje_riesgo"]) * 0.015,
                barra.get_y() + barra.get_height() / 2, etiqueta,
                va="center", ha="left", fontsize=9)

    ax.set_xlabel("Porcentaje de fragmentos reales en zona de riesgo (%)", fontsize=11, weight="bold")
    ax.set_title(f"Ranking de exposicion a la zona de riesgo, por clase (inferencia) {title_suffix}",
                 fontsize=13, pad=15, weight="bold")
    ax.set_xlim(0, max(df["porcentaje_riesgo"]) * 1.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()