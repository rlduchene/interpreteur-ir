import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks, peak_widths, savgol_filter

GRID_MIN, GRID_MAX = 400, 4000
MODES = ["Auto", "Transmittance (%)", "Transmittance (0–1)", "Absorbance"]

# Zones de référence : INFORMATIVES seulement (colonne « attributions possibles »).
# Le diagnostic, lui, repose sur la logique de `analyse()` (contexte + intensité + largeur).
REFERENCE_ZONES = [
    (3580, 3700, "O–H libre"),
    (3200, 3550, "O–H lié (large)"),
    (3250, 3500, "N–H"),
    (3260, 3330, "≡C–H"),
    (3010, 3110, "=C–H (sp²)"),
    (2840, 3000, "C–H sp³"),
    (2690, 2770, "C–H aldéhyde"),
    (2500, 2700, "O–H acide (épaulement très large)"),
    (2210, 2265, "C≡N"),
    (2100, 2210, "C≡C"),
    (1790, 1850, "C=O anhydride / chlorure d'acyle"),
    (1735, 1790, "C=O ester / lactone / cétone cyclique"),
    (1700, 1735, "C=O cétone / aldéhyde / acide / ester conjugué"),
    (1630, 1700, "C=O amide / cétone conjuguée ; C=C"),
    (1575, 1615, "C=C aromatique"),
    (1440, 1510, "C=C aromatique ; CH₂/CH₃"),
    (1500, 1570, "NO₂ (asym.)"),
    (1300, 1390, "NO₂ (sym.) ; CH₃"),
    (1150, 1320, "C–O ester / acide / phénol"),
    (1000, 1150, "C–O alcool / éther"),
    (960, 975, "alcène trans (hors plan)"),
    (880, 920, "alcène terminal =CH₂ (hors plan)"),
    (800, 860, "aromatique para (hors plan)"),
    (730, 780, "aromatique ortho / mono (hors plan)"),
    (680, 725, "aromatique mono / méta ; alcène cis"),
    (500, 800, "C–Cl / C–Br"),
]


# ----------------------------------------------------------------------------
# Lecture du fichier
# ----------------------------------------------------------------------------
def parse_spectrum(raw: bytes):
    """Lit un fichier texte à 2 colonnes (séparateur ; , tabulation ou espace,
    virgule ou point décimal, avec ou sans en-tête). Retourne x croissant, y."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    semicolon = ";" in text
    has_dot = bool(re.search(r"\d\.\d", text))
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if semicolon:
            parts = [p.strip().replace(",", ".") for p in line.split(";")]
        elif has_dot:
            parts = [p for p in re.split(r"[,\t ]+", line) if p]
        elif re.search(r"[\t ]", line):
            parts = [p.replace(",", ".") for p in re.split(r"[\t ]+", line) if p]
        else:
            parts = [p for p in line.split(",") if p]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue  # ligne d'en-tête ou de commentaire
        if len(vals) >= 2:
            rows.append(vals[:2])

    if len(rows) < 20:
        raise ValueError("Moins de 20 lignes numériques à 2 colonnes trouvées.")

    arr = np.array(rows)
    arr = arr[np.argsort(arr[:, 0])]
    xu, inv = np.unique(arr[:, 0], return_inverse=True)
    yu = np.bincount(inv, weights=arr[:, 1]) / np.bincount(inv)
    return xu, yu


# ----------------------------------------------------------------------------
# Intensité : conversion en absorbance normalisée (0 → 1)
# ----------------------------------------------------------------------------
def detect_mode(y):
    if np.nanmax(y) > 5:
        return "Transmittance (%)"
    if np.nanmedian(y) > 0.5:
        return "Transmittance (0–1)"
    return "Absorbance"


def to_absorbance(y, mode):
    """Ramène n'importe quelle échelle à une absorbance normalisée, corrigée
    de la ligne de base. Les seuils d'intensité deviennent ainsi indépendants
    de l'unité et du niveau de base du spectre."""
    if mode == "Absorbance":
        A = y.astype(float)
    else:
        T = y / 100.0 if mode == "Transmittance (%)" else y.astype(float)
        T0 = np.percentile(T, 98)  # ligne de base ≈ 100 % de transmission
        T = np.clip(T / T0, 1e-3, None)
        A = -np.log10(T)
    A = np.clip(A - np.percentile(A, 2), 0, None)
    return A / A.max() if A.max() > 0 else A


def force(a):
    return "forte" if a >= 0.5 else ("moyenne" if a >= 0.2 else "faible")


def prepare(raw, mode, smooth, prom_min):
    x, y = parse_spectrum(raw)
    used_mode = detect_mode(y) if mode == "Auto" else mode
    keep = (x >= GRID_MIN) & (x <= GRID_MAX)
    x, y = x[keep], y[keep]
    if len(x) < 20:
        raise ValueError("Aucune donnée entre 400 et 4000 cm⁻¹ : vérifiez que la 1re colonne est le nombre d'onde.")

    A = to_absorbance(y, used_mode)

    # Grille régulière à 1 cm⁻¹ : les paramètres (lissage, distance, largeur)
    # sont ainsi en cm⁻¹ quelle que soit la résolution du fichier.
    xg = np.arange(np.ceil(x[0]), np.floor(x[-1]) + 1.0, 1.0)
    Ag = np.interp(xg, x, A)
    yg = np.interp(xg, x, y)
    As = savgol_filter(Ag, smooth, 3) if len(Ag) > smooth else Ag
    As = np.clip(As, 0, None)

    idx, props = find_peaks(As, prominence=prom_min, distance=4)
    peaks = []
    if len(idx):
        widths = peak_widths(As, idx, rel_height=0.5)[0]
        for i, pr, w in zip(idx, props["prominences"], widths):
            if As[i] >= 0.03 and not (2280 <= xg[i] <= 2400):  # 2280–2400 : CO₂ atmosphérique
                peaks.append(dict(i=int(i), x=float(xg[i]), a=float(As[i]),
                                  prom=float(pr), w=float(w)))
    return xg, yg, As, peaks, used_mode


# ----------------------------------------------------------------------------
# Interprétation
# ----------------------------------------------------------------------------
def analyse(x, A, peaks):
    """Retourne (fonctions, squelette, remarques).

    fonctions : liste de dict(label, ev, conf). Chaque détecteur est INDÉPENDANT :
    plusieurs entrées = composé polyfonctionnel."""

    def mean(lo, hi):
        m = (x >= lo) & (x <= hi)
        return float(A[m].mean()) if m.any() else 0.0

    def best(lo, hi, min_a=0.0, max_w=None, min_w=None):
        c = [p for p in peaks
             if lo <= p["x"] <= hi and p["a"] >= min_a
             and (max_w is None or p["w"] <= max_w)
             and (min_w is None or p["w"] >= min_w)]
        return max(c, key=lambda p: p["a"]) if c else None

    def fmt(p):
        return f"{p['x']:.0f} cm⁻¹, {force(p['a'])}"

    fonctions, squelette, remarques = [], [], []

    def add(label, ev):
        conf = "élevée" if len(ev) >= 3 else ("moyenne" if len(ev) == 2 else "faible")
        fonctions.append(dict(label=label, ev=ev, conf=conf))

    # ---- Indices de base --------------------------------------------------
    oh_broad = best(3150, 3600, 0.12, min_w=70)         # O–H lié : bande LARGE
    oh_free = best(3580, 3700, 0.05, max_w=60)          # O–H libre : bande FINE
    # O–H d'acide : absorbe là où les C–H n'absorbent pas (2500–2700), relativement
    # à une zone « vide » (1900–2100) pour ne pas être trompé par une ligne de base inclinée.
    acid_oh = (mean(2500, 2700) - mean(1900, 2100)) >= 0.08
    alkyne_h = best(3250, 3340, 0.25, max_w=45)
    alkyne_cc_term = best(2100, 2150, 0.02, max_w=40)
    terminal_alkyne = bool(alkyne_h and alkyne_cc_term)
    nh = [p for p in peaks
          if 3250 <= p["x"] <= 3500 and p["a"] >= 0.06 and p["w"] <= 70
          and not (terminal_alkyne and p is alkyne_h)]
    ald_ch = best(2690, 2770, 0.04, max_w=60)           # C–H aldéhyde (le pic à ~2720)
    ch_sp3 = best(2840, 3000, 0.1, max_w=60)
    ch_sp2 = best(3010, 3110, 0.03, max_w=60)
    nitrile = best(2210, 2265, 0.1, max_w=40)
    alkyne_int = None if nitrile else best(2190, 2260, 0.03, max_w=40)
    arom1 = best(1575, 1615, 0.03, max_w=40)
    arom2 = best(1480, 1515, 0.05)                      # bande ~1500 : propre au cycle aromatique
    aromatic = bool(arom1 and arom2 and ch_sp2)
    alkene_cc = best(1625, 1685, 0.04, max_w=40)
    nitro = best(1500, 1570, 0.3) and best(1300, 1390, 0.3)

    # ---- Carbonyles : on en cherche PLUSIEURS --------------------------------
    co = best(1630, 1850, 0.15)
    c_o = best(1150, 1320, 0.15)
    cos = []
    if co:
        cos.append(co)
        for p in peaks:
            amide_low = bool(nh and 1630 <= p["x"] < 1700 and best(1510, 1580, 0.1))  # C=O d'amide + bande amide II
            if (p is not co and (1680 <= p["x"] <= 1850 or amide_low)
                    and abs(p["x"] - co["x"]) >= 15            # évite de compter deux fois la même bande
                    and p["a"] >= max(0.15, 0.3 * co["a"])):   # et les épaulements négligeables
                cos.append(p)
        cos.sort(key=lambda p: -p["x"])

    def ratio(p):  # intensité du C–O rapportée à celle de ce C=O
        return c_o["a"] / p["a"] if c_o else 0.0

    # anhydride / chlorure d'acyle (C=O ≥ 1790 cm⁻¹)
    high = [p for p in cos if p["x"] >= 1790]
    if high:
        h = high[0]
        partner = best(1730, 1790, 0.15)
        if partner:
            add("Anhydride d'acide", [f"C=O à {fmt(h)}", f"2ᵉ bande C=O à {fmt(partner)}"])
            cos = [p for p in cos if p is not h and p is not partner]
        else:
            add("Chlorure d'acyle (ou anhydride)", [f"C=O à {fmt(h)}"])
            cos = [p for p in cos if p is not h]

    used = set()
    for p in cos:
        v = p["x"]
        r = ratio(p)
        if v >= 1735:
            if ald_ch and "aldehyde" not in used and v <= 1745 and r < 0.5:
                role = "aldehyde"
            elif r >= 0.5 and "ester" not in used:
                role = "ester"
            else:
                role = "cetone_cyc"
        elif v >= 1700:
            if acid_oh and "acide" not in used:
                role = "acide"
            elif ald_ch and "aldehyde" not in used:
                role = "aldehyde"
            elif r >= 0.5 and "ester" not in used:
                role = "ester"
            else:
                role = "cetone"
        elif acid_oh and "acide" not in used:
            role = "acide"
        elif ald_ch and "aldehyde" not in used and v >= 1660:
            role = "aldehyde"
        elif nh and "amide" not in used:
            role = "amide"
        elif v >= 1660:
            role = "cetone_conj"
        else:
            role = "amide_tert"
        if role in used:
            continue  # même fonction déjà décrite (ex. deux bandes C=O de cétone)
        used.add(role)

        ev = [f"C=O à {fmt(p)}"]
        if role == "acide":
            ev.append("absorption large entre 2500 et 2700 cm⁻¹ (O–H de dimère d'acide)")
            if v < 1700:
                ev.append("C=O abaissé (< 1700 cm⁻¹) : acide conjugué à une double liaison")
            if c_o:
                ev.append(f"C–O à {fmt(c_o)}")
            add("Acide carboxylique", ev)
        elif role == "aldehyde":
            ev.append(f"C–H aldéhydique à {ald_ch['x']:.0f} cm⁻¹")
            if v < 1710:
                ev.append("C=O abaissé : aldéhyde conjugué / aromatique")
            add("Aldéhyde", ev)
        elif role == "amide":
            ev.append(f"N–H ({len(nh)} bande(s) vers {nh[0]['x']:.0f} cm⁻¹)")
            amide2 = best(1510, 1580, 0.1)
            if amide2:
                ev.append(f"bande amide II à {amide2['x']:.0f} cm⁻¹")
            add("Amide primaire" if len(nh) >= 2 else "Amide secondaire", ev)
        elif role == "ester":
            ev.append(f"C–O intense à {fmt(c_o)} (≥ 50 % de ce C=O)")
            add("Ester", ev)
        elif role == "cetone_cyc":
            add("Cétone probable (C=O haut : cycle à 5 chaînons ?) ou lactone", ev)
        elif role == "cetone":
            if not ald_ch:
                ev.append("pas de C–H aldéhydique")
            if not acid_oh:
                ev.append("pas d'O–H d'acide")
            add("Cétone (aliphatique)", ev)
        elif role == "cetone_conj":
            add("Cétone conjuguée / aromatique", ev + ["C=O abaissé par conjugaison"])
        else:
            add("Amide tertiaire ou C=O très conjugué", ev)

    # ---- Fonctions sans C=O (ou en plus d'un C=O) ----------------------------
    oh = oh_broad if (oh_broad and oh_broad["a"] >= 0.2) else None
    if oh and (not acid_oh or oh["x"] >= 3300):   # sinon, c'est l'enveloppe de l'acide
        c_alc = best(1000, 1210, 0.2)
        c_phen = best(1200, 1260, 0.2) if (aromatic and not cos) else None
        if c_phen:
            add("Phénol", [f"O–H lié large à {fmt(oh)}", f"C–O à {fmt(c_phen)}"])
        elif c_alc:
            nu = c_alc["x"]
            lab = ("Alcool (primaire probable)" if nu < 1085 else
                   "Alcool (secondaire probable)" if nu < 1125 else "Alcool (tertiaire probable)")
            add(lab, [f"O–H lié large à {fmt(oh)}", f"C–O à {fmt(c_alc)}"])
            if cos:
                remarques.append("Bande O–H large avec un carbonyle : alcool (composé polyfonctionnel) ou eau résiduelle ; vérifiez la bande de l'eau vers 1640 cm⁻¹.")
        else:
            remarques.append("Bande O–H large sans C–O net : possible eau résiduelle (vérifiez aussi la bande vers 1640 cm⁻¹).")

    if nh and "amide" not in used:
        add("Amine primaire" if len(nh) >= 2 else "Amine secondaire (ou autre N–H)",
            [f"N–H : {len(nh)} bande(s) fine(s) vers {nh[0]['x']:.0f} cm⁻¹"])

    if nitrile:
        add("Nitrile", [f"C≡N à {fmt(nitrile)}"])
    if nitro:
        add("Groupe nitro NO₂", ["deux bandes intenses vers 1500–1570 et 1300–1390 cm⁻¹"])
    if terminal_alkyne:
        add("Alcyne terminal", [f"≡C–H à {alkyne_h['x']:.0f} cm⁻¹", f"C≡C à {alkyne_cc_term['x']:.0f} cm⁻¹"])
    elif alkyne_int:
        add("Alcyne interne possible", [f"C≡C à {alkyne_int['x']:.0f} cm⁻¹ (bande faible normale)"])

    # éther : seulement si le C–O intense n'est pas déjà expliqué par un acide, un ester ou un alcool
    expliquee = bool(used & {"acide", "ester"}) or any(
        f["label"].startswith(("Alcool", "Phénol")) for f in fonctions)
    if not expliquee:
        c_eth = best(1050, 1150, 0.4)
        c_aryl = best(1230, 1270, 0.4) if aromatic else None
        pe = c_eth or c_aryl
        if pe:
            ev = [f"C–O intense à {fmt(pe)}"]
            if not cos:
                ev.append("pas de C=O, pas d'O–H")
            add("Éther probable", ev)

    # ---- Squelette -----------------------------------------------------------
    if oh_free:
        squelette.append(f"O–H libre ({fmt(oh_free)})")

    if aromatic:
        txt = "Cycle aromatique (C=C 1450–1610 + =C–H > 3000 cm⁻¹)"
        p690, p730 = best(680, 725, 0.1), best(730, 780, 0.1)
        p770, p800 = best(770, 810, 0.1), best(800, 860, 0.1)
        p860 = best(860, 900, 0.08)
        if p730 and p690:
            txt += " — motif monosubstitué probable (bandes ~750 et ~700 cm⁻¹)"
        elif p860 and p770 and p690:
            txt += " — motif méta probable"
        elif p730 and not p690:
            txt += " — motif ortho probable (~750 cm⁻¹)"
        elif p800 and not p690 and not p730:
            txt += " — motif para probable (~820 cm⁻¹)"
        squelette.append(txt)
    elif ch_sp2 and alkene_cc:
        txt = f"Alcène (=C–H {ch_sp2['x']:.0f} + C=C {alkene_cc['x']:.0f} cm⁻¹)"
        if acid_oh:
            pass  # la bande O–H d'acide (~920 cm⁻¹) brouille les déformations hors plan
        elif best(960, 975, 0.1):
            txt += " — trans (~965 cm⁻¹)"
        elif best(985, 1000, 0.1) and best(905, 920, 0.1):
            txt += " — vinyle terminal (~990 et ~910 cm⁻¹)"
        elif best(880, 895, 0.1):
            txt += " — 1,1-disubstitué (~890 cm⁻¹)"
        squelette.append(txt)
    elif ch_sp2 and cos and cos[-1]["x"] < 1710:
        squelette.append(f"Alcène conjugué au carbonyle probable (=C–H {ch_sp2['x']:.0f} cm⁻¹ + C=O abaissé ; C=C masquée par la bande C=O)")
    elif ch_sp2:
        squelette.append(f"=C–H sp² faible ({ch_sp2['x']:.0f} cm⁻¹) : insaturation à confirmer")

    if ch_sp3:
        if acid_oh:
            squelette.append(f"Bande vers {ch_sp3['x']:.0f} cm⁻¹ : C–H sp³ à confirmer (l'enveloppe O–H d'acide recouvre cette zone)")
        else:
            squelette.append(f"C–H aliphatique sp³ ({fmt(ch_sp3)})")

    if not aromatic and ch_sp2 and (alkene_cc or (cos and cos[-1]["x"] < 1710)):
        remarques.append("Isomérie cis/trans : les critères classiques (≈ 965 cm⁻¹ pour trans) ne sont pas fiables pour un alcène conjugué à des C=O ; comparez à un spectre de référence.")

    hal = best(500, 800, 0.5)
    if hal and not aromatic and not cos:
        squelette.append(f"Bande intense à {hal['x']:.0f} cm⁻¹ : halogénoalcane (C–Cl / C–Br) possible, à confirmer")

    # ---- Repli : hydrocarbures ------------------------------------------------
    if not fonctions:
        if aromatic:
            add("Hydrocarbure aromatique", ["pas d'hétéroatome détecté"])
        elif ch_sp2 and alkene_cc:
            add("Alcène / hydrocarbure insaturé", ["pas d'hétéroatome détecté"])
        elif ch_sp3:
            add("Hydrocarbure saturé (alcane)", ["C–H sp³ seuls, aucune autre bande significative"])

    return fonctions, squelette, remarques


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
def zones_for(x):
    return " / ".join(lbl for lo, hi, lbl in REFERENCE_ZONES if lo <= x <= hi) or "—"


def main():
    st.set_page_config(page_title="Interpréteur de Spectre IR", page_icon="🧪", layout="wide")
    st.title("🧪 Interpréteur de spectre infrarouge (IR)")
    st.markdown("Analyse par **position, intensité relative et largeur** des bandes (400–4000 cm⁻¹).")

    with st.sidebar:
        st.header("Paramètres")
        mode = st.selectbox("Unité de l'axe Y", MODES, help="« Auto » devine l'unité ; corrigez-la si le diagnostic est incohérent.")
        prom = st.slider("Sensibilité (prominence min.)", 0.01, 0.15, 0.02, 0.01,
                         help="Plus petit = plus de pics détectés (et plus de bruit).")
        smooth = st.slider("Lissage (cm⁻¹)", 3, 21, 7, 2)

    up = st.file_uploader("Fichier de spectre (2 colonnes : nombre d'onde, intensité)",
                          type=["csv", "txt", "tsv", "dat", "dpt", "asc"])
    if up is None:
        st.info("Chargez un fichier pour lancer l'analyse.")
        return

    try:
        x, y, A, peaks, used_mode = prepare(up.getvalue(), mode, smooth, prom)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.info("Le fichier doit contenir deux colonnes numériques (en-tête facultatif).")
        return

    st.caption(f"Unité utilisée : **{used_mode}** — plage analysée : {x[0]:.0f}–{x[-1]:.0f} cm⁻¹")
    if x[0] > 700:
        st.warning(f"Le spectre commence à {x[0]:.0f} cm⁻¹ : la zone d'empreinte est incomplète.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Spectre")
        is_t = used_mode != "Absorbance"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Spectre"))
        if peaks:
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in peaks], y=[y[p["i"]] for p in peaks],
                mode="markers+text", name="Pics",
                text=[f"{p['x']:.0f}" for p in peaks],
                textposition="bottom center" if is_t else "top center",
                marker=dict(color="red", size=6)))
        fig.update_layout(
            xaxis=dict(autorange="reversed", title="Nombre d'onde (cm⁻¹)"),
            yaxis=dict(title=used_mode), showlegend=False, height=480)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Diagnostic")
        fonctions, squelette, remarques = analyse(x, A, peaks)
        if fonctions:
            noms = " + ".join(f["label"] for f in fonctions)
            if len(fonctions) > 1:
                st.success(f"🧩 Composé polyfonctionnel probable ({len(fonctions)} fonctions) : **{noms}**")
            else:
                st.success(f"**{noms}**")
            for f in fonctions:
                st.markdown(f"**{f['label']}** — confiance {f['conf']}")
                for e in f["ev"]:
                    st.markdown(f"- {e}")
        else:
            st.warning("Aucune fonction identifiable avec ces réglages.")
        if squelette:
            st.markdown("**Squelette et liaisons**")
            for g in squelette:
                st.markdown(f"- {g}")
        for r in remarques:
            st.info(r)
        if len(fonctions) > 1:
            st.caption("Polyfonctionnel : les bandes de fonctions voisines se recouvrent (C=O, C–O, O–H), et l'IR ne compte pas les groupes identiques (mono- ou diacide ?).")
        st.caption("L'IR seul ne prouve pas une structure : confrontez ce résultat à la formule brute, à la RMN ou à la masse.")

    with st.expander("Détail des bandes détectées"):
        if peaks:
            df = pd.DataFrame([{
                "Nombre d'onde (cm⁻¹)": round(p["x"]),
                "Intensité relative": round(p["a"], 2),
                "Force": force(p["a"]),
                "Largeur (cm⁻¹)": round(p["w"]),
                "Attributions possibles": zones_for(p["x"]),
            } for p in sorted(peaks, key=lambda p: -p["x"])])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.write("Aucun pic détecté : diminuez la prominence minimale.")


if __name__ == "__main__":
    main()
