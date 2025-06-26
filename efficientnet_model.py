from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, regularizers

def create_model(input_shape=(224, 224, 3)):
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

    model = models.Sequential()
    model.add(base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))  # ⬅️ Increase dropout
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)))  # ⬅️ Stronger L2
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
