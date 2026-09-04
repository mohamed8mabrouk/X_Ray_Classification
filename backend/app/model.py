import numpy as np
import tensorflow as tf
from PIL import Image


class ImageClassifier:
    def __init__(self, model_path: str):
        self.labels = ["Fractured","Not Fractured"]
        self.model = self.build_model()
        self.model.load_weights(model_path)
    def build_model(self):


        return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    


    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize((150, 150))
        array = tf.keras.utils.img_to_array(image)
        array = np.expand_dims(array, axis=0)
        return array

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self.model.predict(tensor, verbose=0)

    def postprocess(self, raw_output: np.ndarray) -> dict:
     probability = float(raw_output[0][0])

     if probability >= 0.5:
        label = self.labels=[1]
       
     else:
        label = self.labels[0]
       

     return {
        "label": label,
        
    }

    def predict(self, image: Image.Image) -> dict:
        tensor = self.preprocess(image)
        raw_output = self.infer(tensor)
        return self.postprocess(raw_output)
