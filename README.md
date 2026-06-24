# 🌿 PlantCare AI — Détection des maladies des plantes

Application web Streamlit basée sur Deep Learning pour détecter **38 maladies** sur **14 espèces végétales** à partir d'une photo de feuille.

---

## 🚀 Lancement rapide (sans Docker)

```bash
# 1. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. (Optionnel) Placer votre modèle entraîné dans models/
#    Nom attendu : best_model.keras  ou  mobilenetv2.h5  ou  cnn_scratch.h5

# 4. Lancer l'application
streamlit run app/app.py
```

Ouvrir dans le navigateur : http://localhost:8501

---

## 🐳 Lancement avec Docker

```bash
docker-compose up --build
```

---

## 📁 Structure du projet

```
plant-disease-detection/
├── app/
│   └── app.py              # Application Streamlit principale
├── notebooks/              # Notebooks Jupyter (phases 1-5)
├── scripts/
│   ├── database.py         # Couche SQLite (historique)
│   ├── utils.py            # Chargement modèle & inférence
│   └── schema.sql          # Schéma de la base de données
├── models/                 # Modèles entraînés (.h5 / .keras)
├── data/PlantVillage/      # Dataset (non inclus dans Git)
├── static/                 # CSS & images statiques
├── logs/                   # Logs applicatifs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🧠 Modèles

| Modèle        | Accuracy cible | Fichier attendu          |
|---------------|---------------|--------------------------|
| CNN Scratch   | ~85%          | `models/cnn_scratch.h5`  |
| MobileNetV2   | >95%          | `models/mobilenetv2.h5`  |
| ResNet50      | >94%          | `models/resnet50.h5`     |

> Sans modèle, l'application fonctionne en **mode démonstration**.

---

## 👥 Équipe — OFPPT 2026

- Stagiaire 1 : Data Engineering & Prétraitement
- Stagiaire 2 : CNN From Scratch
- Stagiaire 3 : Fine-Tuning (MobileNetV2 & ResNet50)
- Stagiaire 4 : Déploiement (Streamlit, Docker, Cloud)
