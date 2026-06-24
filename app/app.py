"""
app.py — PlantCare AI  ·  Version 2.0
UX/UI premium avec images embarquées, mode démo intelligent
"""

import os, sys, base64, numpy as np, streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.database import (
    init_db, insert_prediction, get_all_predictions,
    get_total_predictions, get_top_diseases,
    get_average_confidence, get_predictions_per_day, clear_history,
)
from scripts.utils import load_model, predict_image, DISEASES, format_disease_name

# ── helpers ──────────────────────────────────────────────────────────────────
def img_to_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

LOGO_B64 = img_to_b64(os.path.join(ROOT, "static", "images", "logo.png"))
BG_B64   = img_to_b64(os.path.join(ROOT, "static", "images", "bg.png"))

LOGO_SRC = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""
BG_SRC   = f"data:image/png;base64,{BG_B64}"   if BG_B64  else ""

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700&display=swap');

/* ── reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, [data-testid="stAppViewContainer"] {{
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif;
}}
[data-testid="stHeader"] {{ display: none; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(160deg, #0a1628 0%, #0d1f0f 100%) !important;
    border-right: 1px solid #1e3a2f !important;
}}
[data-testid="stSidebar"] * {{ color: #c9d1d9 !important; }}

/* ── sidebar brand ── */
.sidebar-brand {{
    display: flex; align-items: center; gap: 12px;
    padding: 1.2rem 0 1rem;
    border-bottom: 1px solid #1e3a2f;
    margin-bottom: 1.2rem;
}}
.sidebar-logo {{
    width: 44px; height: 44px; border-radius: 12px;
    object-fit: cover;
    border: 2px solid #2ea043;
    box-shadow: 0 0 12px #2ea04355;
}}
.sidebar-name {{
    font-size: 1.15rem; font-weight: 800;
    background: linear-gradient(90deg, #3fb950, #79c0ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.sidebar-tag {{
    font-size: 0.7rem; color: #8b949e !important; margin-top: 1px;
}}

/* ── nav pills ── */
[data-testid="stRadio"] > div {{
    display: flex; flex-direction: column; gap: 4px;
}}
[data-testid="stRadio"] label {{
    border-radius: 10px !important;
    padding: 0.55rem 1rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: background 0.18s, color 0.18s !important;
    cursor: pointer !important;
}}
[data-testid="stRadio"] label:hover {{
    background: #1e3a2f !important;
    color: #3fb950 !important;
}}

/* ── hero ── */
.hero {{
    position: relative; border-radius: 20px; overflow: hidden;
    min-height: 260px; margin-bottom: 2rem;
    display: flex; align-items: flex-end;
    background: #0d1f0f;
}}
.hero-bg {{
    position: absolute; inset: 0;
    background-image: url('{BG_SRC}');
    background-size: cover; background-position: center;
    filter: brightness(0.35) saturate(0.8);
}}
.hero-overlay {{
    position: absolute; inset: 0;
    background: linear-gradient(to top, #0d1117ee 30%, transparent 100%);
}}
.hero-content {{
    position: relative; z-index: 2;
    padding: 2.5rem;
}}
.hero-eyebrow {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3fb950; margin-bottom: 0.5rem;
}}
.hero-title {{
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 700; line-height: 1.15;
    color: #f0f6fc; margin-bottom: 0.6rem;
}}
.hero-title span {{ color: #3fb950; }}
.hero-desc {{
    font-size: 0.95rem; color: #8b949e;
    max-width: 520px; line-height: 1.6;
}}

/* ── stat cards ── */
.stats-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin-bottom: 2rem;
}}
.stat-card {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px; padding: 1.2rem 1.4rem;
    transition: border-color 0.2s, transform 0.2s;
}}
.stat-card:hover {{
    border-color: #2ea043;
    transform: translateY(-2px);
}}
.stat-value {{
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #3fb950, #79c0ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
}}
.stat-label {{
    font-size: 0.78rem; color: #8b949e; margin-top: 5px; font-weight: 500;
}}

/* ── feature cards ── */
.features-row {{
    display: grid; grid-template-columns: repeat(3,1fr); gap: 14px;
}}
.feature-card {{
    background: #161b22; border: 1px solid #21262d;
    border-radius: 14px; padding: 1.4rem;
}}
.feature-icon {{ font-size: 1.8rem; margin-bottom: 0.7rem; }}
.feature-title {{
    font-size: 0.95rem; font-weight: 700; color: #f0f6fc;
    margin-bottom: 0.4rem;
}}
.feature-desc {{ font-size: 0.82rem; color: #8b949e; line-height: 1.55; }}

/* ── section title ── */
.section-title {{
    font-size: 1.1rem; font-weight: 700; color: #f0f6fc;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 8px;
}}
.section-title::after {{
    content: ''; flex: 1; height: 1px; background: #21262d;
}}

/* ── upload panel ── */
.upload-hint {{
    background: #161b22; border: 1px dashed #2ea04355;
    border-radius: 14px; padding: 2.5rem; text-align: center;
    color: #8b949e; font-size: 0.88rem; line-height: 1.7;
}}
.upload-hint .icon {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}

/* ── result card ── */
.result-card {{
    background: linear-gradient(135deg, #0d1f0f 0%, #161b22 100%);
    border: 1px solid #2ea043;
    border-radius: 16px; padding: 1.8rem;
    box-shadow: 0 0 30px #2ea04322;
}}
.result-disease {{
    font-size: 1.4rem; font-weight: 800; color: #f0f6fc;
    margin-bottom: 0.3rem; line-height: 1.2;
}}
.result-plant {{
    font-size: 0.82rem; color: #8b949e; margin-bottom: 1rem;
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500;
}}
.badge {{
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 999px; padding: 0.3rem 0.9rem;
    font-size: 0.8rem; font-weight: 700;
    margin-bottom: 1.2rem;
}}
.badge-healthy {{ background: #1a3a1f; color: #3fb950; border: 1px solid #2ea043; }}
.badge-disease {{ background: #3a1a1a; color: #f85149; border: 1px solid #da3633; }}
.badge-demo    {{ background: #1a2a3a; color: #79c0ff; border: 1px solid #388bfd; }}

/* ── confidence bar ── */
.conf-label {{
    font-size: 0.78rem; color: #8b949e; margin-bottom: 6px;
    display: flex; justify-content: space-between;
}}
.conf-bar {{
    height: 8px; background: #21262d; border-radius: 999px;
    overflow: hidden; margin-bottom: 1.4rem;
}}
.conf-fill {{
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #2ea043, #3fb950);
    transition: width 0.6s ease;
}}

/* ── top5 ── */
.prob-row {{
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 8px; font-size: 0.82rem;
}}
.prob-name {{ flex: 1; color: #c9d1d9; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }}
.prob-bar-bg {{
    width: 100px; height: 6px; background: #21262d;
    border-radius: 999px; overflow: hidden; flex-shrink: 0;
}}
.prob-bar-fill {{
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #388bfd, #79c0ff);
}}
.prob-pct {{ color: #8b949e; width: 42px; text-align: right; flex-shrink: 0; }}

/* ── demo warning ── */
.demo-banner {{
    background: #1a2a3a; border: 1px solid #388bfd33;
    border-radius: 10px; padding: 0.8rem 1.1rem;
    font-size: 0.82rem; color: #79c0ff;
    margin-bottom: 1.2rem; display: flex; gap: 8px; align-items: flex-start;
}}

/* ── table ── */
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* ── metric boxes ── */
.dash-metric {{
    background: #161b22; border: 1px solid #21262d;
    border-radius: 14px; padding: 1.2rem 1.4rem;
    text-align: center;
}}
.dash-metric-val {{
    font-size: 2.2rem; font-weight: 800; color: #3fb950; line-height: 1;
}}
.dash-metric-lbl {{
    font-size: 0.78rem; color: #8b949e; margin-top: 4px;
}}

/* ── file uploader restyle ── */
[data-testid="stFileUploader"] {{
    background: #161b22 !important;
    border: 1.5px dashed #2ea043 !important;
    border-radius: 14px !important;
}}
[data-testid="stFileUploader"] * {{ color: #8b949e !important; }}

/* ── streamlit elements ── */
.stButton > button {{
    background: #2ea043 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: background 0.18s !important;
}}
.stButton > button:hover {{ background: #3fb950 !important; }}
.stDownloadButton > button {{
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 8px !important;
    font-size: 0.85rem !important;
}}
div[data-testid="stMarkdownContainer"] p {{ color: #c9d1d9; }}
</style>
""", unsafe_allow_html=True)

# ── init ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌿 Chargement…")
def startup():
    init_db()
    return load_model()

model = startup()
is_demo = model is None

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_html = f'<img src="{LOGO_SRC}" class="sidebar-logo">' if LOGO_SRC else '🌿'
    st.markdown(f"""
    <div class="sidebar-brand">
        {logo_html}
        <div>
            <div class="sidebar-name">PlantCare AI</div>
            <div class="sidebar-tag">Deep Learning · OFPPT 2026</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Accueil",
        "🔬  Analyser",
        "📊  Historique",
        "📈  Dashboard",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    if is_demo:
        st.markdown("""
        <div style="background:#1a2a3a;border:1px solid #388bfd44;border-radius:10px;
                    padding:0.8rem 1rem;font-size:0.78rem;color:#79c0ff;">
            <b>⚡ Mode démonstration</b><br>
            Placez un fichier <code>.h5</code> ou <code>.keras</code>
            dans <code>models/</code> pour activer l'IA.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0d1f0f;border:1px solid #2ea04355;border-radius:10px;
                    padding:0.8rem 1rem;font-size:0.78rem;color:#3fb950;">
            ✅ <b>Modèle chargé</b><br>
            Prêt pour l'analyse en temps réel.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:#484f58;line-height:1.7;">
        38 maladies · 14 espèces<br>
        PlantVillage Dataset<br>
        MobileNetV2 Fine-Tuning
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE ACCUEIL
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Accueil":

    st.markdown(f"""
    <div class="hero">
        <div class="hero-bg"></div>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-eyebrow">Deep Learning · Computer Vision</div>
            <div class="hero-title">
                Détectez les maladies<br>de vos <span>plantes</span> en secondes
            </div>
            <div class="hero-desc">
                Téléchargez une photo de feuille — notre modèle identifie
                parmi 38 maladies sur 14 espèces végétales avec une précision
                supérieure à 95 %.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">38</div>
            <div class="stat-label">Maladies détectables</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">14</div>
            <div class="stat-label">Espèces végétales</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">&gt;95%</div>
            <div class="stat-label">Précision MobileNetV2</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">54K+</div>
            <div class="stat-label">Images d'entraînement</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="features-row">
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <div class="feature-title">Simple à utiliser</div>
            <div class="feature-desc">
                Glissez une photo JPG, PNG ou WEBP.
                Le diagnostic s'affiche en moins d'une seconde.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">3 modèles comparés</div>
            <div class="feature-desc">
                CNN from scratch, MobileNetV2 et ResNet50 —
                benchmarkés sur le même dataset.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Historique complet</div>
            <div class="feature-desc">
                Toutes les analyses sont sauvegardées en SQLite
                avec statistiques et export CSV.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE ANALYSER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Analyser":

    st.markdown('<div class="section-title">🔬 Analyser une image</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Déposez une photo de feuille",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div class="upload-hint">
            <div class="icon">🌿</div>
            <b>Glissez une photo ici</b><br>
            JPG · JPEG · PNG · WEBP · 200 MB max<br>
            <span style="color:#484f58;font-size:0.78rem;">
                Idéalement une feuille unique, bien éclairée, fond neutre
            </span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Les 38 maladies détectables"):
            cols = st.columns(3)
            for i, d in enumerate(DISEASES):
                cols[i % 3].markdown(
                    f"<span style='font-size:0.78rem;color:#8b949e;'>• {format_disease_name(d)}</span>",
                    unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown('<div class="section-title">🖼️ Image</div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown(
                f"<span style='font-size:0.75rem;color:#484f58;'>"
                f"{uploaded.name} · {image.size[0]}×{image.size[1]} px</span>",
                unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="section-title">🔍 Résultat</div>', unsafe_allow_html=True)

            with st.spinner("Analyse…"):
                pred_class, confidence, all_probs = predict_image(model, image)

            disease_raw  = DISEASES[pred_class] if pred_class < len(DISEASES) else "Inconnu"
            disease_name = format_disease_name(disease_raw)
            is_healthy   = "healthy" in disease_raw.lower()

            # split plant / disease
            if "___" in disease_raw:
                plant_part, dis_part = disease_raw.split("___", 1)
            else:
                plant_part, dis_part = disease_raw, ""
            plant_label = plant_part.replace("_"," ").replace("(","").replace(")","").strip().title()
            dis_label   = dis_part.replace("_"," ").strip().title()

            badge_cls  = "badge-demo" if is_demo else ("badge-healthy" if is_healthy else "badge-disease")
            badge_icon = "⚡" if is_demo else ("✅" if is_healthy else "⚠️")
            badge_text = "Démo — résultat aléatoire" if is_demo else ("Plante saine" if is_healthy else "Maladie détectée")

            demo_warning = ""
            if is_demo:
                demo_warning = """
                <div class="demo-banner">
                    ⚡ <span><b>Mode démonstration :</b> aucun modèle chargé.
                    Placez un fichier <code>.h5</code> dans <code>models/</code>
                    pour obtenir de vraies prédictions.</span>
                </div>"""

            conf_color = "#3fb950" if is_healthy else "#f85149"
            bar_style  = f"width:{min(confidence,100):.1f}%;background:{'linear-gradient(90deg,#2ea043,#3fb950)' if is_healthy else 'linear-gradient(90deg,#da3633,#f85149)'};"

            st.markdown(f"""
            {demo_warning}
            <div class="result-card">
                <div class="result-plant">{plant_label}</div>
                <div class="result-disease">{dis_label or 'Saine'}</div>
                <div>
                    <span class="badge {badge_cls}">{badge_icon} {badge_text}</span>
                </div>
                <div class="conf-label">
                    <span>Confiance</span>
                    <span style="color:{conf_color};font-weight:700;">{confidence:.1f}%</span>
                </div>
                <div class="conf-bar">
                    <div class="conf-fill" style="{bar_style}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # top-5
            st.markdown("<br><div class='section-title' style='font-size:0.85rem;'>Top 5 probabilités</div>",
                        unsafe_allow_html=True)
            top5 = np.argsort(all_probs)[::-1][:5]
            rows_html = ""
            for i in top5:
                name = format_disease_name(DISEASES[i] if i < len(DISEASES) else f"Classe {i}")
                pct  = float(all_probs[i]) * 100
                w    = min(pct / max(float(all_probs[top5[0]])*100, 1) * 100, 100)
                rows_html += f"""
                <div class="prob-row">
                    <div class="prob-name">{name}</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{w:.1f}%"></div>
                    </div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)

            if not is_demo:
                insert_prediction(uploaded.name, disease_raw, round(confidence, 2))

# ════════════════════════════════════════════════════════════════════════════
# PAGE HISTORIQUE
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Historique":
    import pandas as pd

    st.markdown('<div class="section-title">📊 Historique des analyses</div>',
                unsafe_allow_html=True)

    records = get_all_predictions(limit=200)

    if not records:
        st.markdown("""
        <div class="upload-hint">
            <div class="icon">📂</div>
            <b>Aucune analyse enregistrée</b><br>
            Allez dans <b>Analyser</b> pour tester votre première image.
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(records)
        df["Maladie"]       = df["prediction"].apply(format_disease_name)
        df["Confiance (%)"] = df["confidence"].round(2)
        df = df.rename(columns={
            "id": "ID", "image_name": "Fichier", "date_prediction": "Date"
        })[["ID","Fichier","Maladie","Confiance (%)","Date"]]

        st.dataframe(df, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns([1,1,4])
        with c1:
            if st.button("🗑️ Vider"):
                clear_history(); st.rerun()
        with c2:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ CSV", csv, "historique.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
# PAGE DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈  Dashboard":
    import pandas as pd
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    st.markdown('<div class="section-title">📈 Dashboard</div>', unsafe_allow_html=True)

    total    = get_total_predictions()
    avg_conf = get_average_confidence()
    top_dis  = get_top_diseases(limit=6)
    per_day  = get_predictions_per_day(days=14)

    c1,c2,c3 = st.columns(3)
    for col, val, lbl in [
        (c1, str(total),         "Analyses totales"),
        (c2, f"{avg_conf:.1f}%", "Confiance moyenne"),
        (c3, str(len(top_dis)),  "Maladies distinctes"),
    ]:
        col.markdown(f"""
        <div class="dash-metric">
            <div class="dash-metric-val">{val}</div>
            <div class="dash-metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if total == 0:
        st.markdown("""
        <div class="upload-hint">
            <div class="icon">📈</div>
            <b>Pas encore de données</b><br>
            Effectuez des analyses pour voir les statistiques ici.
        </div>""", unsafe_allow_html=True)
    elif HAS_PLOTLY:
        ca, cb = st.columns(2)
        with ca:
            if top_dis:
                df_top = pd.DataFrame(top_dis)
                df_top["label"] = df_top["prediction"].apply(format_disease_name)
                fig = px.bar(df_top, x="occurrences", y="label", orientation="h",
                    color="occurrences", color_continuous_scale=[[0,"#1a3a2f"],[1,"#3fb950"]],
                    labels={"occurrences":"Analyses","label":""})
                fig.update_layout(
                    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    font_color="#c9d1d9", coloraxis_showscale=False,
                    margin=dict(l=0,r=0,t=10,b=0), height=280)
                fig.update_xaxes(gridcolor="#21262d")
                fig.update_yaxes(gridcolor="#21262d")
                st.plotly_chart(fig, use_container_width=True)
        with cb:
            if per_day:
                df_day = pd.DataFrame(per_day)
                df_day["day"] = pd.to_datetime(df_day["day"])
                fig2 = px.area(df_day.sort_values("day"), x="day", y="total",
                    labels={"day":"Date","total":"Analyses"},
                    color_discrete_sequence=["#3fb950"])
                fig2.update_traces(fill="tozeroy", fillcolor="#2ea04322", line_width=2)
                fig2.update_layout(
                    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    font_color="#c9d1d9",
                    margin=dict(l=0,r=0,t=10,b=0), height=280)
                fig2.update_xaxes(gridcolor="#21262d")
                fig2.update_yaxes(gridcolor="#21262d")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Installez plotly : `pip install plotly`")
