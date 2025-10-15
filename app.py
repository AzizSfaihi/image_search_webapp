from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os, uuid

from descriptors import color_histogram, gray_histogram, correlogram, vgg16_descriptor
from index_search import index_images, search_similar

app = Flask(__name__, static_folder="static")
CORS(app)

DATASET_FOLDER = "static/images_dataset"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DESCRIPTORS = {
    "1": ("Histogramme Couleur", color_histogram),
    "2": ("Histogramme Niveaux de Gris", gray_histogram),
    "3": ("Corrélogramme", correlogram),
    "4": ("Descripteur Profond VGG16", vgg16_descriptor)
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/search", methods=["POST"])
def search():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["image"]
    descriptor_key = request.form.get("descriptor")

    if descriptor_key not in DESCRIPTORS:
        return jsonify({"error": "Descripteur invalide"}), 400

    filename = f"{uuid.uuid4()}.jpg"
    query_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(query_path)

    descriptor_fn = DESCRIPTORS[descriptor_key][1]
    index = index_images(DATASET_FOLDER, descriptor_fn)
    query_signature = descriptor_fn(query_path)
    results = search_similar(index, query_signature, top_k=6)
    results = [(f, d) for f, d in results if f != filename][:5]

    return jsonify({
        "results": [{
            "filename": f,
            "distance": round(float(d), 3),
            "url": f"/static/images_dataset/{f.replace('\\', '/')}"
        } for f, d in results]
    })

if __name__ == "__main__":
    app.run(debug=True)
