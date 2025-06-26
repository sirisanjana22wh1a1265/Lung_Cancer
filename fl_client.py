import flwr as fl
import numpy as np
from Efficient_model import create_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

class LungCancerClient(fl.client.NumPyClient):
    def __init__(self, client_path):
        self.model = create_model()
        self.model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

        self.x_train = np.load(f"{client_path}/x_train.npy")
        self.y_train = np.load(f"{client_path}/y_train.npy")
        self.x_test = np.load(f"{client_path}/x_test.npy")
        self.y_test = np.load(f"{client_path}/y_test.npy")

        # ✅ Ensure labels are shaped correctly: (N, 1)
        if len(self.y_train.shape) == 1:
            self.y_train = np.expand_dims(self.y_train, axis=-1)
        if len(self.y_test.shape) == 1:
            self.y_test = np.expand_dims(self.y_test, axis=-1)
        print("Label distribution (train):", np.unique(self.y_train, return_counts=True))
        print("Label distribution (test):", np.unique(self.y_test, return_counts=True))

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
        ]

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=10,
            batch_size=16,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

        train_loss, train_acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print(f"Train Accuracy: {train_acc:.4f}")

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        y_pred_prob = self.model.predict(self.x_test)
        y_pred = (y_pred_prob > 0.5).astype("int32")

        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, digits=4))

        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        return loss, len(self.x_test), {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }