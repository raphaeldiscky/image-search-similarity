from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()  # mengekstrak fitur gambar

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)
        # memanggil fungsi extract() dari feature_extractor.py
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")
        np.save(feature_path, feature)  # save fitur ke folder feature
