import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from efficientnet_model import create_model

# ====== Load Test Data ======
x_test = np.load("LungCancerFLData1/client1/x_test.npy")
y_test = np.load("LungCancerFLData1/client1/y_test.npy")

# Ensure labels have shape (N, 1)
if len(y_test.shape) == 1:
    y_test = np.expand_dims(y_test, axis=-1)

# ====== Load and Compile Model ======
model = create_model()
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.load_weights("saved_weights/final_model_39200.weights.h5")  # ✅ Make sure path is correct

# ====== Evaluate ======
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# ====== Predict ======
y_probs = model.predict(x_test)
y_preds = (y_probs > 0.5).astype(int)

# ====== Classification Report ======
print("\n📋 Classification Report:")
print(classification_report(y_test, y_preds, digits=4))

# ====== Confusion Matrix ======
cm = confusion_matrix(y_test, y_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("🔍 Confusion Matrix")
plt.tight_layout()
plt.show()

# ====== ROC Curve ======
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("📈 ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
