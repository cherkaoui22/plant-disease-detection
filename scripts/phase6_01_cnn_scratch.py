# =============================================================================
# PHASE 6 - MODÈLE 1 : CNN FROM SCRATCH
# Projet : Détection des maladies des plantes - PlantVillage (38 classes)
# =============================================================================

# ─────────────────────────────────────────────
# ÉTAPE 0 : INSTALLATION DES DÉPENDANCES
# ─────────────────────────────────────────────
# Exécute ces commandes dans ton terminal avant de lancer le script :
#
#   pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
#
# ─────────────────────────────────────────────

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import json
import time

# ─────────────────────────────────────────────
# ÉTAPE 1 : CONFIGURATION GLOBALE
# ─────────────────────────────────────────────

# ⚠️  MODIFIE CE CHEMIN selon l'emplacement de ton dataset PlantVillage
DATASET_PATH = "./PlantVillage"   # <-- Change ici si nécessaire

IMG_SIZE     = (128, 128)          # Taille des images d'entrée
BATCH_SIZE   = 32                  # Nombre d'images par batch
EPOCHS       = 30                  # Nombre d'epochs max (EarlyStopping arrêtera avant)
NUM_CLASSES  = 38                  # 38 classes PlantVillage
SEED         = 42                  # Reproductibilité

MODEL_SAVE_PATH = "./models/cnn_scratch_plant_disease.keras"
HISTORY_SAVE    = "./models/cnn_scratch_history.json"

os.makedirs("./models", exist_ok=True)
os.makedirs("./results", exist_ok=True)

print("="*60)
print("  CNN FROM SCRATCH — PlantVillage 38 classes")
print("="*60)
print(f"  TensorFlow version : {tf.__version__}")
print(f"  GPU disponible     : {tf.config.list_physical_devices('GPU')}")
print("="*60)

# ─────────────────────────────────────────────
# ÉTAPE 2 : CHARGEMENT ET AUGMENTATION DES DONNÉES
# ─────────────────────────────────────────────

# Générateur pour l'entraînement avec Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,           # Normalisation [0,1]
    validation_split=0.2,          # 80% train / 20% validation
    rotation_range=20,             # Rotation aléatoire ±20°
    width_shift_range=0.15,        # Décalage horizontal
    height_shift_range=0.15,       # Décalage vertical
    shear_range=0.1,               # Cisaillement
    zoom_range=0.15,               # Zoom aléatoire
    horizontal_flip=True,          # Miroir horizontal
    fill_mode='nearest'            # Remplissage des pixels manquants
)

# Générateur pour la validation (sans augmentation)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# Chargement des données d'entraînement
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED,
    shuffle=True
)

# Chargement des données de validation
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=False
)

# Sauvegarde des noms de classes (utile pour l'app Streamlit)
class_names = list(train_generator.class_indices.keys())
with open("./models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print(f"\n✅ Classes trouvées      : {train_generator.num_classes}")
print(f"✅ Images d'entraînement : {train_generator.samples}")
print(f"✅ Images de validation  : {val_generator.samples}")
print(f"✅ Noms de classes sauvegardés dans ./models/class_names.json")

# ─────────────────────────────────────────────
# ÉTAPE 3 : ARCHITECTURE CNN FROM SCRATCH
# ─────────────────────────────────────────────
# Architecture : 4 blocs Conv → Flatten → Dense → Softmax
# Simple, entraînable de zéro, bon point de départ

def build_cnn_scratch(input_shape=(128, 128, 3), num_classes=38):
    """
    Construit un CNN from scratch.
    Architecture :
      Bloc 1 : Conv(32) → BN → ReLU → MaxPool → Dropout
      Bloc 2 : Conv(64) → BN → ReLU → MaxPool → Dropout
      Bloc 3 : Conv(128) → BN → ReLU → MaxPool → Dropout
      Bloc 4 : Conv(256) → BN → ReLU → MaxPool → Dropout
      Tête   : Flatten → Dense(512) → Dropout → Dense(num_classes, softmax)
    """
    model = models.Sequential(name="CNN_From_Scratch")

    # ── Bloc 1 ──────────────────────────────
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            input_shape=(*input_shape,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Bloc 2 ──────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Bloc 3 ──────────────────────────────
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    # ── Bloc 4 ──────────────────────────────
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    # ── Tête de classification ───────────────
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

model = build_cnn_scratch(input_shape=(128, 128, 3), num_classes=NUM_CLASSES)
model.summary()

# ─────────────────────────────────────────────
# ÉTAPE 4 : COMPILATION
# ─────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✅ Modèle compilé avec Adam lr=0.001")

# ─────────────────────────────────────────────
# ÉTAPE 5 : CALLBACKS
# ─────────────────────────────────────────────

callbacks = [
    # Arrête l'entraînement si val_accuracy ne s'améliore plus pendant 5 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Sauvegarde automatiquement le meilleur modèle
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Réduit le learning rate si la perte stagne
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ─────────────────────────────────────────────
# ÉTAPE 6 : ENTRAÎNEMENT
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  DÉBUT DE L'ENTRAÎNEMENT — CNN FROM SCRATCH")
print("="*60)

start_time = time.time()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\n⏱️  Temps d'entraînement : {training_time/60:.1f} minutes")

# Sauvegarde de l'historique
history_dict = {
    "accuracy":     history.history['accuracy'],
    "val_accuracy": history.history['val_accuracy'],
    "loss":         history.history['loss'],
    "val_loss":     history.history['val_loss'],
    "training_time_seconds": training_time
}
with open(HISTORY_SAVE, "w") as f:
    json.dump(history_dict, f, indent=2)

print(f"✅ Historique sauvegardé : {HISTORY_SAVE}")

# ─────────────────────────────────────────────
# ÉTAPE 7 : ÉVALUATION
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  ÉVALUATION FINALE")
print("="*60)

val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"  Val Loss     : {val_loss:.4f}")
print(f"  Val Accuracy : {val_accuracy*100:.2f}%")

# ─────────────────────────────────────────────
# ÉTAPE 8 : GRAPHIQUES — Accuracy & Loss
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CNN From Scratch — Courbes d'entraînement", fontsize=14, fontweight='bold')

epochs_range = range(1, len(history.history['accuracy']) + 1)

# Courbe Accuracy
axes[0].plot(epochs_range, history.history['accuracy'],     'b-o', label='Train Accuracy', markersize=4)
axes[0].plot(epochs_range, history.history['val_accuracy'], 'r-o', label='Val Accuracy',   markersize=4)
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Courbe Loss
axes[1].plot(epochs_range, history.history['loss'],     'b-o', label='Train Loss', markersize=4)
axes[1].plot(epochs_range, history.history['val_loss'], 'r-o', label='Val Loss',   markersize=4)
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./results/cnn_scratch_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Graphiques sauvegardés : ./results/cnn_scratch_curves.png")

# ─────────────────────────────────────────────
# ÉTAPE 9 : RAPPORT DE CLASSIFICATION
# ─────────────────────────────────────────────

print("\n  Génération du rapport de classification...")

# Prédictions sur la validation
val_generator.reset()
y_pred_proba = model.predict(val_generator, verbose=0)
y_pred        = np.argmax(y_pred_proba, axis=1)
y_true        = val_generator.classes

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
with open("./results/cnn_scratch_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(classification_report(y_true, y_pred, target_names=class_names))
print("✅ Rapport sauvegardé : ./results/cnn_scratch_report.json")

# ─────────────────────────────────────────────
# ÉTAPE 10 : RÉSUMÉ FINAL
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  RÉSUMÉ CNN FROM SCRATCH")
print("="*60)
print(f"  Epochs effectuées    : {len(history.history['accuracy'])}")
print(f"  Meilleure Val Acc    : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"  Meilleure Val Loss   : {min(history.history['val_loss']):.4f}")
print(f"  Temps entraînement   : {training_time/60:.1f} min")
print(f"  Modèle sauvegardé    : {MODEL_SAVE_PATH}")
print("="*60)
print("\n✅ Phase 6 - Modèle 1 terminé. Lance maintenant : phase6_02_mobilenetv2.py")
