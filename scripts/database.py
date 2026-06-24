"""
database.py
-------------------------------------------------------------------
Couche d'accès aux données (SQLite) pour l'application de détection
des maladies des plantes.

Responsabilités :
    - Création de la base et de la table 'predictions' au démarrage
    - Insertion d'une nouvelle prédiction (historique automatique)
    - Lecture de l'historique
    - Calcul des statistiques pour le Dashboard

Auteur : Équipe Projet (4 stagiaires) - OFPPT
-------------------------------------------------------------------
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any

# Chemin de la base de données (surchargeable via variable d'environnement)
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "predictions.db"))

# Chemin du script SQL de création de schéma
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


@contextmanager
def get_connection():
    """
    Context manager qui ouvre une connexion SQLite et la ferme proprement,
    même en cas d'erreur. Évite les fuites de connexions.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Initialise la base de données en exécutant le schéma SQL si la table
    n'existe pas encore. Doit être appelée une fois au démarrage de l'app.
    """
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
    else:
        # Schéma de secours si le fichier schema.sql est introuvable
        schema_sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            date_prediction TEXT NOT NULL
        );
        """

    with get_connection() as conn:
        conn.executescript(schema_sql)


def insert_prediction(image_name: str, prediction: str, confidence: float) -> int:
    """
    Enregistre une nouvelle prédiction dans l'historique.

    Args:
        image_name: nom du fichier image analysé
        prediction: classe prédite (ex: 'Tomato___Early_blight')
        confidence: score de confiance entre 0 et 100

    Returns:
        L'identifiant (id) de la ligne insérée.
    """
    date_prediction = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (image_name, prediction, confidence, date_prediction)
            VALUES (?, ?, ?, ?)
            """,
            (image_name, prediction, confidence, date_prediction),
        )
        return cursor.lastrowid


def get_all_predictions(limit: int = 200) -> List[Dict[str, Any]]:
    """Retourne l'historique des prédictions, les plus récentes en premier."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, image_name, prediction, confidence, date_prediction
            FROM predictions
            ORDER BY date_prediction DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_total_predictions() -> int:
    """Nombre total d'analyses effectuées."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM predictions").fetchone()
        return row["total"] if row else 0


def get_top_diseases(limit: int = 5) -> List[Dict[str, Any]]:
    """Retourne les maladies les plus fréquemment détectées."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT prediction, COUNT(*) AS occurrences
            FROM predictions
            GROUP BY prediction
            ORDER BY occurrences DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_average_confidence() -> float:
    """Confiance moyenne de toutes les prédictions enregistrées."""
    with get_connection() as conn:
        row = conn.execute("SELECT AVG(confidence) AS avg_conf FROM predictions").fetchone()
        return round(row["avg_conf"], 2) if row and row["avg_conf"] is not None else 0.0


def get_predictions_per_day(days: int = 14) -> List[Dict[str, Any]]:
    """Nombre de prédictions par jour, utile pour un graphique d'évolution."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DATE(date_prediction) AS day, COUNT(*) AS total
            FROM predictions
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
            """,
            (days,),
        ).fetchall()
        return [dict(r) for r in rows]


def clear_history() -> None:
    """Vide complètement l'historique (utilisé pour les tests/démo)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions")
