import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'

from flask import (
    Flask, flash, render_template, Response,
    request, redirect, url_for, session
)
import datetime
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import gc

UPLOAD_FOLDER = './static/Data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MODEL_PATH = os.environ.get('MODEL_PATH', '../backend/models/model.h5')

app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app.config['MODEL_PATH'] = MODEL_PATH
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production-2026')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
capture = 0
camera = None
latest_capture = None

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model():
    global model
    if model is None:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        tf.config.set_visible_devices([], 'GPU')

        gc.collect()

        model_path = app.config['MODEL_PATH']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', run_eagerly=False)

        dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
        model.predict(dummy, verbose=0, batch_size=1)

        del dummy
        gc.collect()

    return model

def load_image(img_path: str) -> np.ndarray:
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor.astype(np.float32) / 255.0
    return img_tensor

def predict_path(img_path: str) -> str:
    try:
        import tensorflow as tf

        get_model()
        new_image = load_image(img_path)

        with tf.device('/CPU:0'):
            pred = model.predict(new_image, verbose=0, batch_size=1)[0][0]

        del new_image
        gc.collect()

        if pred < 0.5:
            return f"Skin Disease Detected (Confidence: {(1-pred)*100:.1f}%) - Please consult a dermatologist."
        else:
            return f"No Skin Disease Detected (Confidence: {pred*100:.1f}%) - Regular checkups recommended."
    except Exception as e:
        gc.collect()
        return f"Analysis error: {str(e)}"

def init_camera():
    global camera
    if os.environ.get('RENDER') or os.environ.get('FLASK_ENV') == 'production':
        return False

    try:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return False
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True
    except Exception as e:
        return False

def gen_frames():
    global capture, latest_capture
    if not init_camera():
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Not Available", (100, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    while True:
        try:
            success, frame = camera.read()
            if not success:
                continue

            if capture:
                capture = 0
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(filepath, frame)
                latest_capture = filename

            frame_flipped = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', frame_flipped)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            continue

@app.route("/", methods=['GET'])
@app.route("/index", methods=['GET'])
def index():
    if request.method == 'GET' and 'from_prediction' not in session:
        session.clear()
    session.pop('from_prediction', None)
    return render_template('index.html')

@app.route("/predicts", methods=['GET', 'POST'])
def predicts():
    if request.method == 'GET':
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            product = predict_path(file_path)
            user_image = url_for('static', filename=f'Data/{unique_filename}')

            session['from_prediction'] = True

            return render_template('index.html',
                                 product=product,
                                 user_image=user_image)
        except Exception as e:
            flash(f'Error: {str(e)}')
            return redirect(url_for('index'))
        finally:
            gc.collect()

    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return Response(status=503)

@app.route('/tasks', methods=['POST'])
def tasks():
    global capture, latest_capture

    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            capture = 1

            import time
            time.sleep(1.0)

            if latest_capture:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_capture)

                if os.path.exists(img_path):
                    product = predict_path(img_path)
                    user_image = url_for('static', filename=f'Data/{latest_capture}')

                    session['from_prediction'] = True

                    return render_template('index.html',
                                         product=product,
                                         user_image=user_image)
                else:
                    flash('Capture failed')
            else:
                flash('No image captured')

    return redirect(url_for('index'))

@app.route('/health')
def health():
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
            'timestamp': datetime.datetime.now().isoformat()
        }
        return status, 200
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}, 503

@app.teardown_appcontext
def cleanup(error):
    global camera
    try:
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None
    except Exception as e:
        pass

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
