from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np


class FeatureExtractor:
    def __init__(self):
        # menggunakan pre-trained model VGG16 dengan imageNet
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input,
                           outputs=base_model.get_layer('fc1').output)  # output shape (1, 4096)

    def extract(self, img):
        """
        Mengekstrak feature dari gambar

        Gambar diambil dg PIL.Image.open(path) atau tensorflow.keras.preprocessing.image.load_img(path)

        Mengembalikan feature dengan shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG menerima gambar dg ukuran 224x224
        img = img.convert('RGB')  # memastikan gambar berwarna/RGB
        x = image.img_to_array(img)  # mengubah gambar ke array
        x = np.expand_dims(x, axis=0)  # expand array
        x = preprocess_input(x)  # preprocess input
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # melakukan normalisasi
