import os, numpy as np

def chi2_distance(h1, h2, eps=1e-10):
    return 0.5 * np.sum(((h1 - h2)**2) / (h1 + h2 + eps))


def index_images(folder_path, descriptor_function):
    index = {}
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                try:
                    signature = descriptor_function(image_path)
                    # relative path to keep folder info (e.g. cars/img1.jpg)
                    rel_path = os.path.relpath(image_path, folder_path)
                    index[rel_path] = signature
                except Exception as e:
                    print(f"[Warning] Failed to process {image_path}: {e}")
    return index

def search_similar(idx, q_sig, top_k=5):
    dists = []
    for f, sig in idx.items():
        if isinstance(sig, np.ndarray):
            d = chi2_distance(q_sig, sig)
        else:
            d = abs(q_sig - sig)
        dists.append((f,d))
    dists.sort(key=lambda x: x[1])
    return dists[:top_k]
