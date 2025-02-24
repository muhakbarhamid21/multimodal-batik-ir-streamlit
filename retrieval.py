# retrieval.py
import numpy as np

def combine_embeddings(text_embedding=None, image_embedding=None):
    if text_embedding is not None and image_embedding is None:
        # Query teks saja: tambahkan 256 nol untuk bagian gambar
        combined = np.concatenate((text_embedding, np.zeros(256)))
    elif image_embedding is not None and text_embedding is None:
        # Query gambar saja: tambahkan 256 nol untuk bagian teks
        combined = np.concatenate((np.zeros(256), image_embedding))
    elif text_embedding is not None and image_embedding is not None:
        combined = np.concatenate((text_embedding, image_embedding))
    else:
        combined = None
    return combined
