import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnet_model import create_model
from tensorflow.keras.models import load_model

# Load the test data
x_test = np.load("LungCancerFLData1/client1/x_test.npy")
y_test = np.load("LungCancerFLData1/client1/y_test.npy")

# Load your trained model (assumes you're saving weights or a final model)
model = create_model()
model.load_weights("final_model_weights.h5")  # or use load_model("model.h5") if you saved it

# Predict probabilities and classes
y_prob = model.predict(x_test)
y_pred = (y_prob > 0.5).astype("int32")

# -------------------------------
# 🧾 Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 📉 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 📊 ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.grid()
plt.show()
