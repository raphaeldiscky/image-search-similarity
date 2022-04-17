import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# membaca fitur gambar
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # menyimpan gambar query
        img = Image.open(file.stream)
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":",
                                                                                    ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # melakukan pencarian terhadap gambar input / query
        query = fe.extract(img)
        # mencari distance query ke semua feature yang telah diupload dg euclidean distance
        dists = np.linalg.norm(features-query, axis=1)
        ids = np.argsort(dists)[:200]  # menampilkan 200 data termirip
        scores = [(dists[id], img_paths[id])
                  for id in ids]  # menampilkan score

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)  # render image dan score pada browser
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
    app.run("0.0.0.0")
