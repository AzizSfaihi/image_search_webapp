# 🖼️ Image Search Web App

A simple image similarity search engine using Python + Flask + Deep Learning (VGG16).

## 🚀 Features
- Search similar images by color, grayscale, correlogram, or VGG16.
- Upload your own query image from the browser.
- View top 5 most similar images.

## 🧩 Installation

git clone https://github.com/yourusername/image-search-webapp.git
cd image-search-webapp
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate
pip install -r requirements.txt
python app.py



Note: The `static/images_dataset/` and `static/uploads/` folders contain `.gitkeep` files to preserve their structure in Git.  
Please add your own dataset images in `images_dataset/` before running the app.
