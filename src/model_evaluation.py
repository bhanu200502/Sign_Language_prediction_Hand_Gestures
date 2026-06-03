# ============================================
# STAGE 6: Model Performance Evaluation
# Goal: Generate confusion matrix, ROC curve,
#       and precision/recall metrics
# ============================================

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Suppress warnings ─────────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score,
                             precision_recall_curve, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ── SETTINGS ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models")
MODEL_NAME = "sign_model.h5"

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42

CHART_COLORS = [
    'blue', 'red', 'green', 'orange', 'purple', 'brown',
    'pink', 'gray', 'olive', 'cyan', 'magenta', 'gold',
    'navy', 'teal', 'maroon', 'lime', 'aqua', 'coral',
    'indigo', 'khaki', 'lightblue', 'lightgreen', 'salmon',
    'violet', 'turquoise', 'slateblue'
]
# ──────────────────────────────────────────────────────


def load_data():
    """Load training data"""
    print("\n" + "="*60)
    print("📂 LOADING DATA")
    print("="*60)

    X_path = os.path.join(DATA_PATH, "X.npy")
    y_path = os.path.join(DATA_PATH, "y.npy")
    labels_path = os.path.join(DATA_PATH, "labels.npy")

    for path in [X_path, y_path, labels_path]:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return None, None, None

    X = np.load(X_path)
    y = np.load(y_path)
    labels = np.load(labels_path)

    print(f"  ✅ X shape      : {X.shape}")
    print(f"  ✅ y shape      : {y.shape}")
    print(f"  ✅ Labels       : {list(labels)}")
    print(f"  ✅ Classes      : {len(labels)}")

    return X, y, labels


def prepare_test_set(X, y):
    """Prepare test set (same split logic as training)"""
    print("\n" + "="*60)
    print("✂️  PREPARING TEST SET")
    print("="*60)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED,
        stratify=y_trainval
    )

    print(f"  ✅ Train samples : {len(X_train)}")
    print(f"  ✅ Val samples   : {len(X_val)}")
    print(f"  ✅ Test samples  : {len(X_test)}")

    return X_test, y_test


def load_model():
    """Load trained model"""
    print("\n" + "="*60)
    print("📂 LOADING MODEL")
    print("="*60)

    model_path = os.path.join(MODEL_PATH, MODEL_NAME)

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"   Run Stage 5 first:")
        print(f"   python src/model_training.py")
        return None

    print(f"  ⏳ Loading model...")
    model = tf.keras.models.load_model(model_path)

    print(f"  ✅ Model loaded")
    print(f"  ✅ Input shape    : {model.input_shape}")
    print(f"  ✅ Output shape   : {model.output_shape}")

    return model


def generate_predictions(model, X_test):
    """Generate predictions and probabilities"""
    print("\n" + "="*60)
    print("🔮 GENERATING PREDICTIONS")
    print("="*60)

    predictions_proba = model.predict(X_test, verbose=0)
    predictions = np.argmax(predictions_proba, axis=1)

    print(f"  ✅ Predictions generated for {len(X_test)} samples")

    return predictions, predictions_proba


def plot_confusion_matrix(y_true, y_pred, labels):
    """Create and save confusion matrix in HD quality"""
    print("\n" + "="*60)
    print("📊 CREATING CONFUSION MATRIX (HD FORMAT)")
    print("="*60)

    cm = confusion_matrix(y_true, y_pred)

    # Create larger figure for HD quality
    fig, ax = plt.subplots(figsize=(28, 26))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1.2,
        linecolor='white',
        square=True,
        cbar_kws={'label': 'Count', 'shrink': 0.88, 'pad': 0.02},
        annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'},
        ax=ax,
        vmin=0,
        vmax=cm.max()
    )

    # Enhanced title
    ax.set_title(
        'Confusion Matrix - ASL Sign Language Recognition',
        fontsize=32,
        fontweight='bold',
        pad=30
    )
    
    # Enhanced axis labels
    ax.set_ylabel('Actual', fontsize=24, fontweight='bold', labelpad=15)
    ax.set_xlabel('Predicted', fontsize=24, fontweight='bold', labelpad=15)
    
    # Enhanced tick parameters for better readability
    ax.tick_params(axis='x', labelrotation=45, labelsize=18, width=2, length=6)
    ax.tick_params(axis='y', labelrotation=0, labelsize=18, width=2, length=6)

    # Enhanced colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16, width=2, length=6)
    cbar.set_label('Count', size=20, weight='bold', labelpad=15)

    fig.tight_layout()

    # Save files with highest quality
    cm_path = os.path.join(MODEL_PATH, "confusion_matrix.png")
    cm_hd_path = os.path.join(MODEL_PATH, "confusion_matrix_HD.png")
    cm_svg_path = os.path.join(MODEL_PATH, "confusion_matrix.svg")
    
    fig.savefig(cm_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(cm_hd_path, dpi=800, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(cm_svg_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"  ✅ Confusion matrix (600 DPI) → {cm_path}")
    print(f"  ✅ Confusion matrix (800 DPI HD) → {cm_hd_path}")
    print(f"  ✅ Vector confusion matrix → {cm_svg_path}")

    return cm


def plot_roc_curve(y_true, y_proba, labels):
    """Create and save ROC curve in HD quality"""
    print("\n" + "="*60)
    print("📈 CREATING ROC CURVE")
    print("="*60)

    n_classes = len(labels)

    # Binarize labels for multi-class ROC
    y_bin = label_binarize(y_true, classes=range(n_classes))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_bin.ravel(), y_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    # Plot 1: ROC curves for each class
    colors = cycle(CHART_COLORS)

    for i, color in zip(range(n_classes), colors):
        ax1.plot(fpr[i], tpr[i],
                color=color, lw=2.8,
                label=f'{labels[i]} (AUC = {roc_auc[i]:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2.5, label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    ax1.set_title('ROC Curves - Per Class', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3, linewidth=1)

    # Plot 2: Micro-average ROC
    ax2.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            color='deeppink', lw=3.2)
    ax2.plot([0, 1], [0, 1], 'k--', lw=2.5, label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    ax2.set_title('Micro-average ROC', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    ax2.legend(loc="lower right", fontsize=13, frameon=True)
    ax2.grid(True, alpha=0.3, linewidth=1)

    fig.tight_layout()
    roc_path = os.path.join(MODEL_PATH, "roc_curve.png")
    roc_svg_path = os.path.join(MODEL_PATH, "roc_curve.svg")
    fig.savefig(roc_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(roc_svg_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"  ✅ ROC curve saved → {roc_path}")
    print(f"  ✅ Micro-average AUC: {roc_auc['micro']:.4f}")

    return roc_auc


def plot_precision_recall_curve(y_true, y_proba, labels):
    """Create and save Precision-Recall curve in HD quality"""
    print("\n" + "="*60)
    print("🎯 CREATING PRECISION-RECALL CURVE")
    print("="*60)

    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=range(n_classes))

    precision = dict()
    recall = dict()
    pr_auc = dict()

    fig, ax = plt.subplots(figsize=(22, 16))

    colors = cycle(CHART_COLORS)

    for i, color in zip(range(n_classes), colors):
        precision[i], recall[i], _ = precision_recall_curve(
            y_bin[:, i], y_proba[:, i]
        )
        pr_auc[i] = auc(recall[i], precision[i])

        ax.plot(recall[i], precision[i],
                color=color, lw=2.8,
                label=f'{labels[i]} (AUC = {pr_auc[i]:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=18, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=18, fontweight='bold')
    ax.set_title('Precision-Recall Curves - ASL Sign Language Recognition',
                 fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=1)
    fig.tight_layout()

    pr_path = os.path.join(MODEL_PATH, "precision_recall_curve.png")
    pr_svg_path = os.path.join(MODEL_PATH, "precision_recall_curve.svg")
    fig.savefig(pr_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(pr_svg_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"  ✅ Precision-Recall curve saved → {pr_path}")

    return pr_auc


def calculate_metrics(y_true, y_pred, labels):
    """Calculate and display performance metrics"""
    print("\n" + "="*60)
    print("📊 PERFORMANCE METRICS")
    print("="*60)

    # Overall metrics
    accuracy = np.mean(y_true == y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n  🎯 OVERALL METRICS:")
    print(f"  ├─ Accuracy          : {accuracy*100:.2f}%")
    print(f"  ├─ Macro Precision   : {macro_precision:.4f}")
    print(f"  ├─ Macro Recall      : {macro_recall:.4f}")
    print(f"  ├─ Macro F1-Score    : {macro_f1:.4f}")
    print(f"  ├─ Weighted Precision: {weighted_precision:.4f}")
    print(f"  ├─ Weighted Recall   : {weighted_recall:.4f}")
    print(f"  └─ Weighted F1-Score : {weighted_f1:.4f}")

    # Per-class metrics
    print(f"\n  📋 PER-CLASS METRICS:")
    print(f"  {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"  {'-'*60}")

    per_class_metrics = {}
    for i, label in enumerate(labels):
        mask = y_true == i
        if np.sum(mask) > 0:
            tp = np.sum((y_pred == i) & (y_true == i))
            fp = np.sum((y_pred == i) & (y_true != i))
            fn = np.sum((y_pred != i) & (y_true == i))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = np.sum(mask)

            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(support)
            }

            print(f"  {label:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {int(support):<10}")

    # Print detailed classification report
    print(f"\n  📄 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=labels,
                              digits=4, zero_division=0))

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': per_class_metrics
    }


def save_metrics_summary(metrics, labels, roc_auc_micro):
    """Save metrics summary to file"""
    summary_path = os.path.join(MODEL_PATH, "performance_metrics.txt")

    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL PERFORMANCE EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write("OVERALL METRICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy:            {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Macro Precision:     {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:        {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score:      {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted Precision:  {metrics['weighted_precision']:.4f}\n")
        f.write(f"Weighted Recall:     {metrics['weighted_recall']:.4f}\n")
        f.write(f"Weighted F1-Score:   {metrics['weighted_f1']:.4f}\n")
        f.write(f"Micro-average AUC:   {roc_auc_micro:.4f}\n\n")

        f.write("PER-CLASS METRICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}\n")
        f.write("-"*70 + "\n")

        for label in labels:
            if label in metrics['per_class']:
                m = metrics['per_class'][label]
                f.write(f"{label:<15} {m['precision']:<15.4f} {m['recall']:<15.4f} "
                       f"{m['f1']:<15.4f} {m['support']:<10}\n")

    print(f"\n  ✅ Metrics summary saved → {summary_path}")


# ── MAIN ──────────────────────────────────────────────
if __name__ == "__main__":

    print("="*60)
    print("  STAGE 6 — MODEL PERFORMANCE EVALUATION")
    print("="*60)

    # Check running from root folder
    if not os.path.exists(DATA_PATH):
        print("❌ Run from project root:")
        print("   cd D:\\SIGN_LANGUAGE")
        print("   python src/model_evaluation.py")
        exit()

    # Step 1: Load data
    X, y, labels = load_data()
    if X is None:
        exit()

    # Step 2: Prepare test set
    X_test, y_test = prepare_test_set(X, y)

    # Step 3: Load model
    model = load_model()
    if model is None:
        exit()

    # Step 4: Generate predictions
    y_pred, y_proba = generate_predictions(model, X_test)

    # Step 5: Create visualizations
    confusion_matrix_result = plot_confusion_matrix(y_test, y_pred, labels)
    roc_auc = plot_roc_curve(y_test, y_proba, labels)
    pr_auc = plot_precision_recall_curve(y_test, y_proba, labels)

    # Step 6: Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, labels)

    # Step 7: Save summary
    save_metrics_summary(metrics, labels, roc_auc["micro"])

    # ── Final Summary ──────────────────────────
    print("\n" + "="*60)
    print("🎉 EVALUATION COMPLETE!")
    print("="*60)
    print(f"  ✅ Confusion Matrix saved")
    print(f"  ✅ ROC Curves saved")
    print(f"  ✅ Precision-Recall Curves saved")
    print(f"  ✅ Performance metrics saved")
    print("\n  📊 Check models/ folder for visualizations:")
    print(f"     • confusion_matrix.png")
    print(f"     • roc_curve.png")
    print(f"     • precision_recall_curve.png")
    print(f"     • performance_metrics.txt")
    print("="*60 + "\n")
