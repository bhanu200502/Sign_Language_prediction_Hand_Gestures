# ============================================
# STAGE 5: Model Training
# Goal: Build and train Neural Network on
#       63 landmark features, save model
# ============================================

import numpy as np
import os
import time

# ── Suppress TensorFlow warnings ──────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import tensorflow as tf
from tensorflow                      import keras
from tensorflow.keras.models         import Sequential
from tensorflow.keras.layers         import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks      import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils          import to_categorical
from sklearn.model_selection         import train_test_split
from sklearn.preprocessing           import LabelEncoder
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt

# ── SETTINGS ──────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data")      # where X.npy, y.npy are
MODEL_PATH  = os.path.join(BASE_DIR, "models")    # where to save model
MODEL_NAME  = "sign_model.h5"

EPOCHS      = 50          # training rounds
BATCH_SIZE  = 32          # samples per batch
TEST_SIZE   = 0.2         # 20% for testing
VAL_SIZE    = 0.1         # 10% for validation
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────


def load_data():
    """
    Load X.npy, y.npy, labels.npy
    from data/ folder
    """
    print("\n" + "="*55)
    print("📂 LOADING DATA")
    print("="*55)

    X_path      = os.path.join(DATA_PATH, "X.npy")
    y_path      = os.path.join(DATA_PATH, "y.npy")
    labels_path = os.path.join(DATA_PATH, "labels.npy")

    # Check files exist
    for path in [X_path, y_path, labels_path]:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            print(f"   Run Stage 4 first:")
            print(f"   python src/data_preprocessing.py")
            return None, None, None

    # Load files
    X      = np.load(X_path)
    y      = np.load(y_path)
    labels = np.load(labels_path)

    print(f"  ✅ X shape      : {X.shape}")
    print(f"  ✅ y shape      : {y.shape}")
    print(f"  ✅ Labels       : {list(labels)}")
    print(f"  ✅ Classes      : {len(labels)}")
    print(f"  ✅ Features     : {X.shape[1]}")

    # Show class distribution
    print(f"\n  📊 SAMPLES PER CLASS:")
    unique, counts = np.unique(y, return_counts=True)
    for idx, count in zip(unique, counts):
        bar = "█" * (count // 10)
        print(f"     {labels[idx]} → {bar} ({count})")

    return X, y, labels


def prepare_data(X, y, n_classes):
    """
    Split into train/val/test sets
    and convert labels to one-hot encoding
    """
    print("\n" + "="*55)
    print("✂️  SPLITTING DATA")
    print("="*55)

    # Step 1: Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size   = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify    = y    # keep class balance
    )

    # Step 2: Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size    = VAL_SIZE / (1 - TEST_SIZE),
        random_state = RANDOM_SEED,
        stratify     = y_trainval
    )

    print(f"  ✅ Train samples : {len(X_train)}")
    print(f"  ✅ Val samples   : {len(X_val)}")
    print(f"  ✅ Test samples  : {len(X_test)}")

    # Step 3: Convert labels to one-hot
    # e.g. label 2 → [0, 0, 1, 0, 0 ...]
    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat   = to_categorical(y_val,   n_classes)
    y_test_cat  = to_categorical(y_test,  n_classes)

    print(f"\n  ✅ y_train shape : {y_train_cat.shape}")
    print(f"  ✅ y_val shape   : {y_val_cat.shape}")
    print(f"  ✅ y_test shape  : {y_test_cat.shape}")

    return (X_train, X_val, X_test,
            y_train_cat, y_val_cat, y_test_cat,
            y_test)


def build_model(input_size, n_classes):
    """
    Build Dense Neural Network

    Architecture:
    Input(63) → Dense(256) → BN → Dropout
              → Dense(128) → BN → Dropout
              → Dense(64)  → BN → Dropout
              → Output(n_classes, Softmax)
    """
    print("\n" + "="*55)
    print("🏗️  BUILDING MODEL")
    print("="*55)

    model = Sequential([

        # ── Input → Hidden Layer 1 ────────────
        Dense(256,
              activation = 'relu',
              input_shape = (input_size,),
              name = 'dense_1'),
        BatchNormalization(name='bn_1'),
        Dropout(0.3, name='dropout_1'),

        # ── Hidden Layer 2 ────────────────────
        Dense(128,
              activation = 'relu',
              name = 'dense_2'),
        BatchNormalization(name='bn_2'),
        Dropout(0.3, name='dropout_2'),

        # ── Hidden Layer 3 ────────────────────
        Dense(64,
              activation = 'relu',
              name = 'dense_3'),
        BatchNormalization(name='bn_3'),
        Dropout(0.2, name='dropout_3'),

        # ── Output Layer ──────────────────────
        Dense(n_classes,
              activation = 'softmax',
              name = 'output')
    ])

    # ── Compile model ─────────────────────────
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )

    # ── Print summary ─────────────────────────
    model.summary()

    print(f"\n  ✅ Input size  : {input_size}")
    print(f"  ✅ Output size : {n_classes} classes")

    return model


def train_model(model, X_train, X_val,
                y_train, y_val):
    """
    Train the model with callbacks:
    - ModelCheckpoint: save best model
    - EarlyStopping: stop if not improving
    - ReduceLROnPlateau: reduce learning rate
    """
    print("\n" + "="*55)
    print("🚀 TRAINING STARTED")
    print("="*55)
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Train size : {len(X_train)}")
    print(f"  Val size   : {len(X_val)}")
    print("="*55)

    # ── Create models folder ──────────────────
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_save_path = os.path.join(MODEL_PATH, MODEL_NAME)

    # ── Callbacks ─────────────────────────────

    # 1. Save best model automatically
    checkpoint = ModelCheckpoint(
        filepath          = model_save_path,
        monitor           = 'val_accuracy',
        save_best_only    = True,
        verbose           = 1
    )

    # 2. Stop training if val_accuracy
    #    doesn't improve for 10 epochs
    early_stop = EarlyStopping(
        monitor   = 'val_accuracy',
        patience  = 10,
        verbose   = 1,
        restore_best_weights = True
    )

    # 3. Reduce learning rate if stuck
    reduce_lr = ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 5,
        verbose  = 1,
        min_lr   = 0.00001
    )

    # ── Train! ────────────────────────────────
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        callbacks       = [checkpoint, early_stop, reduce_lr],
        verbose         = 1
    )

    elapsed = time.time() - start_time
    print(f"\n  ⏱️  Training time : {elapsed:.1f} seconds")

    return history, model_save_path


def evaluate_model(model, X_test, y_test_cat,
                   y_test, labels):
    """
    Test the model on unseen test data
    """
    print("\n" + "="*55)
    print("📊 EVALUATING MODEL")
    print("="*55)

    # Get test accuracy
    test_loss, test_acc = model.evaluate(
        X_test, y_test_cat, verbose=0
    )

    print(f"  ✅ Test Accuracy : {test_acc*100:.2f}%")
    print(f"  ✅ Test Loss     : {test_loss:.4f}")

    # ── Show some predictions ─────────────────
    print(f"\n  🔍 SAMPLE PREDICTIONS:")
    print(f"  {'Actual':<10} {'Predicted':<10} {'Result'}")
    print(f"  {'-'*35}")

    predictions = model.predict(X_test[:10], verbose=0)

    for i in range(10):
        actual    = labels[y_test[i]]
        predicted = labels[np.argmax(predictions[i])]
        correct   = "✅" if actual == predicted else "❌"
        print(f"  {actual:<10} {predicted:<10} {correct}")

    return test_acc


def save_training_plot(history):
    """
    Save accuracy and loss graphs
    as training_plot.png
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Accuracy plot
        ax1.plot(history.history['accuracy'],
                 label='Train', color='blue', linewidth=2.8)
        ax1.plot(history.history['val_accuracy'],
                 label='Validation', color='orange', linewidth=2.8)
        ax1.set_title('Model Accuracy', fontsize=20, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=13, width=1.5, length=6)
        ax1.legend(fontsize=13, frameon=True)
        ax1.grid(True, alpha=0.3, linewidth=1)

        # Loss plot
        ax2.plot(history.history['loss'],
                 label='Train', color='blue', linewidth=2.8)
        ax2.plot(history.history['val_loss'],
                 label='Validation', color='orange', linewidth=2.8)
        ax2.set_title('Model Loss', fontsize=20, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=13, width=1.5, length=6)
        ax2.legend(fontsize=13, frameon=True)
        ax2.grid(True, alpha=0.3, linewidth=1)

        fig.tight_layout()
        plot_path = os.path.join(MODEL_PATH, "training_plot.png")
        plot_svg_path = os.path.join(MODEL_PATH, "training_plot.svg")
        fig.savefig(plot_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        fig.savefig(plot_svg_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)

        print(f"\n  📈 Training plot saved → {plot_path}")

    except Exception as e:
        print(f"  ⚠️  Could not save plot: {e}")


# ── MAIN ──────────────────────────────────────────────
if __name__ == "__main__":

    print("="*55)
    print("  STAGE 5 — MODEL TRAINING")
    print("="*55)

    # Check running from root folder
    if not os.path.exists(DATA_PATH):
        print("❌ Run from project root:")
        print("   cd D:\\SIGN_LANGUAGE")
        print("   python src/model_training.py")
        exit()

    # Step 1: Load data
    X, y, labels = load_data()
    if X is None:
        exit()

    n_classes  = len(labels)
    input_size = X.shape[1]   # = 63

    # Step 2: Prepare data
    (X_train, X_val, X_test,
     y_train, y_val, y_test_cat,
     y_test_raw) = prepare_data(X, y, n_classes)

    # Step 3: Build model
    model = build_model(input_size, n_classes)

    # Step 4: Train model
    history, model_path = train_model(
        model, X_train, X_val, y_train, y_val
    )

    # Step 5: Evaluate model
    test_acc = evaluate_model(
        model, X_test, y_test_cat, y_test_raw, labels
    )

    # Step 6: Save training plot
    save_training_plot(history)

    # Step 7: Save test accuracy
    accuracy_path = os.path.join(MODEL_PATH, "test_accuracy.npy")
    np.save(accuracy_path, np.array(test_acc))
    print(f"\n  📊 Test accuracy saved → {accuracy_path}")

    # ── Final Summary ──────────────────────────
    print("\n" + "="*55)
    print("🎉 TRAINING COMPLETE!")
    print("="*55)
    print(f"  ✅ Model saved   → {model_path}")
    print(f"  ✅ Test Accuracy → {test_acc*100:.2f}%")
    print(f"  ✅ Classes       → {list(labels)}")
    print(f"\n  👉 Now run Stage 6:")
    print(f"     python src/prediction.py")
    print("="*55)
