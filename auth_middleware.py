from firebase_admin import auth
from flask import request, g
from functools import wraps

def firebase_authentication_middleware(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return {'error': 'Authorization header is missing'}, 401

        auth_parts = auth_header.split()

        if len(auth_parts) != 2 or auth_parts[0].lower() != 'bearer':
            return {'error': 'Invalid authorization header format'}, 401

        id_token = auth_parts[1]

        try:
            decoded_token = auth.verify_id_token(id_token)
            user_id = decoded_token['uid']
            user_record = auth.get_user(user_id)
            g.uid = user_record.uid

        except Exception as e:
            return {'error': f'Authentication failed: {str(e)}'}, 401

        return f(*args, **kwargs)

    return decorated_function

