import os
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import io
from PIL import Image
import numpy as np
import firebase_admin 
import datetime
from model_load import model, class_names
from google.cloud import storage, firestore
from auth_middleware import firebase_authentication_middleware

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

prod = os.environ.get('PRODUCTION', "False").lower() == "true"

cred = None

if prod:
    cred = firebase_admin.credentials.ApplicationDefault()
else:
    cred = firebase_admin.credentials.Certificate('venv/serviceAccount.json')


firebase_admin.initialize_app(cred)

def upload_image_to_gcs(bucket_name, file_stream, destination_blob_name):
    if prod:
        storage_client = storage.Client()
    else:
        storage_client = storage.Client.from_service_account_json('venv/serviceAccount.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_file(file_stream, content_type='image/jpeg')
    
    # Mendapatkan URL file
    url = blob.public_url
    print(f"File uploaded to {url}.")
    return url

def save_prediction_to_firestore(predicted_class, confidence, image_url, user_id):
    if prod:
        db = firestore.Client()
    else:
        db = firestore.Client.from_service_account_json('venv/serviceAccount.json')
    
    # Buat dokumen baru di koleksi 'history'
    new_prediction_ref = db.collection('history').document()
    
    # Data untuk disimpan
    data = {
        'uid': user_id,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'image_url': image_url,
        'timestamp': datetime.datetime.now()
    }
    
    # Simpan data ke dalam Firestore
    new_prediction_ref.set(data)

    print(f"Prediction saved to Firestore with image URL: {image_url}")

def preprocess_image_as_array(image):
    im = Image.open(image).convert('RGB')
    im = im.resize((224, 224))
    return np.asarray(im)

def predict_image_class(model, img_array, class_names):
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
        return predicted_class
    else:
        return None


@app.route('/v1/predict', methods=['POST'])
@firebase_authentication_middleware
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        img_array = preprocess_image_as_array(file)
        predicted_class = predict_image_class(model, img_array, class_names)

        if predicted_class:
            confidence = np.max(model.predict(np.expand_dims(img_array, axis=0))) * 100
            bucket_name = 'images_from_predict'
            destination_blob_name = "image_predict_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
            
            # Membaca file langsung ke dalam memori
            file.seek(0)
            file_stream = io.BytesIO(file.read())
            file_stream.seek(0)  # Pastikan file stream berada di posisi awal sebelum mengunggah
            image_url = upload_image_to_gcs(bucket_name, file_stream, destination_blob_name)
            
            user_id = g.uid
            save_prediction_to_firestore(predicted_class, confidence, image_url, user_id)
            return jsonify({'result': predicted_class, 'confidence': confidence, 'uid' : user_id, 'image_url' : image_url})
        else:
            return jsonify({'error': 'Prediction error'})
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
