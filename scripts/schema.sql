-- =====================================================================
-- schema.sql
-- Schéma de la base de données SQLite pour l'historique des prédictions
-- Projet : Détection des maladies des plantes avec Deep Learning
-- =====================================================================

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name      TEXT    NOT NULL,
    prediction      TEXT    NOT NULL,
    confidence      REAL    NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
    date_prediction TEXT    NOT NULL
);

-- Index pour accélérer les requêtes du Dashboard (tri par date, par maladie)
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions (date_prediction);
CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions (prediction);
