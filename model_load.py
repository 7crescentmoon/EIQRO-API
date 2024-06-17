import os
import tensorflow as tf
from pathlib import Path
from google.cloud import storage 

model_dir = "model"

model_filename = "model_mm_v9_20.h5"

model_path = os.path.join(model_dir, model_filename)

prod = os.environ.get('PRODUCTION', "False").lower() == "true"

if not os.path.exists(model_path):
    os.makedirs(model_dir, exist_ok=True)

    cloud_storage_url = "https://storage.cloud.google.com/hijaiyah_model/model_mm_v9_20.h5"

    if prod:
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(Path('venv/serviceAccount.json').resolve())
    bucket = client.bucket("hijaiyah_model")
    blob = bucket.blob(model_filename)
    blob.download_to_filename(model_path)
    print(f"Model downloaded to {model_path}")
else:
    print(f"Model already exists at {model_path}, loading without download.")

model = tf.keras.models.load_model(model_path)

class_names = [ 'ain', 'alif', 'ba', 'dal', 'dhod', 'dzal',
                'dzho', 'fa', 'ghoin', 'ha', 'ha\'', 'hamzah', 'jim',
                'kaf', 'kho', 'lam', 'lamalif', 'mim', 'nun', 'qof',
                'ro', 'shod', 'sin', 'syin', 'ta', 'tho', 'tsa',
                'wawu', 'ya', 'zain']
