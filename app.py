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
from flask_swagger_ui import get_swaggerui_blueprint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

prod = os.environ.get('PRODUCTION', "False").lower() == "true"

cred = None

if prod:
    cred = firebase_admin.credentials.ApplicationDefault()
else:
    cred = firebase_admin.credentials.Certificate('venv/serviceAccount.json')


firebase_admin.initialize_app(cred)

SWAGGER_URL = '/api-docs'  
API_URL = '/static/openapi.json' 
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'E-iqro': "API Docs"
    }
)

app.register_blueprint(swaggerui_blueprint)

def upload_image_to_gcs(bucket_name, file_stream, destination_blob_name):
    if prod:
        storage_client = storage.Client()
    else:
        storage_client = storage.Client.from_service_account_json('venv/serviceAccount.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_file(file_stream, content_type='image/jpeg')
    
    url = blob.public_url
    print(f"File uploaded to {url}.")
    return url

def save_prediction_to_firestore(predicted_class, confidence, image_url, user_id):
    if prod:
        db = firestore.Client()
    else:
        db = firestore.Client.from_service_account_json('venv/serviceAccount.json')
    
    new_prediction_ref = db.collection('history').document()
    
    data = {
        'uid': user_id,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'image_url': image_url,
        'timestamp': datetime.datetime.now()
    }
    
    new_prediction_ref.set(data)

    print(f"Prediction saved to Firestore with image URL: {image_url}")

def preprocess_image_as_array(image):
    im = Image.open(image).convert('RGB')
    im = im.resize((224, 224))
    return np.asarray(im)

def predict_image_class(model, img_array, class_names, threshold=0.5):
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)[0]

    predicted_class_index = np.argmax(predictions)
    predicted_class_score = predictions[predicted_class_index]
    predicted_class = class_names[predicted_class_index]

    if predicted_class_score >= threshold:
        result = {
            "predicted_class": predicted_class,
            "confidence": float(predicted_class_score),
        }
        return result
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
        predicted_class = predict_image_class(model, img_array, class_names, threshold=0.5)

        if predicted_class:
            confidence = predicted_class['confidence'] * 100
            predict_model = predicted_class['predicted_class']
            bucket_name = 'images_from_predict'
            destination_blob_name = "image_predict_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
            
            file.seek(0)
            file_stream = io.BytesIO(file.read())
            file_stream.seek(0)
            image_url = upload_image_to_gcs(bucket_name, file_stream, destination_blob_name)
            
            user_id = g.uid
            save_prediction_to_firestore(predicted_class, confidence, image_url, user_id)
            return jsonify({'result': predict_model, 'confidence': confidence,'uid' : user_id, 'image_url' : image_url})
        else:
            return jsonify({'error': 'prediction invalid'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/v1/history', methods=['GET'])
@firebase_authentication_middleware
def get_history():
    try:
        user_id = g.uid
        if not user_id:
            return jsonify({'error': 'No user ID provided'}), 400 
        if prod:
            db = firestore.Client()
        else:
            db = firestore.Client.from_service_account_json('venv/serviceAccount.json')
        
        history_ref = db.collection('history').where('uid', '==', user_id)
        history_docs = history_ref.stream()

        history_list = []
        for doc in history_docs:
            history_list.append(doc.to_dict())
        
        return jsonify({'history': history_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
