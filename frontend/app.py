# app.py - Multi-Disease Skin Classifier (HAM10000 + Monkeypox)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '2'

from flask import (
    Flask, flash, render_template, Response,
    request, redirect, url_for, session
)
import datetime
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import gc

# ========== CONFIGURATION ==========
UPLOAD_FOLDER = './static/Data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# ✅ Multiple model configurations with correct paths
MODELS = {
    'ham10000': {
        'path': '/Users/jsgbn/Desktop/skin-identification/backend/models/HAM10000_MobileNetV2_TF2_15.h5',
        'type': 'h5',
        'classes': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
        'descriptions': {
            'akiec': 'Actinic Keratosis',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevus (Mole)',
            'vasc': 'Vascular Lesion'
        },
        'accuracy': {
            'akiec': 45.00, 'bcc': 58.37, 'bkl': 52.27,
            'df': 31.17, 'mel': 42.24, 'nv': 85.36, 'vasc': 40.00
        },
        'overall_accuracy': 71.64,
        'input_size': 128,
        'name': 'HAM10000 Skin Lesion Classifier'
    },
    'monkeypox': {
        'path': '/Users/jsgbn/Desktop/skin-identification/backend/models/model.tflite',
        'type': 'tflite',
        'classes': ['Monkeypox', 'Others'],
        'descriptions': {
            'Monkeypox': 'Monkeypox Virus Infection',
            'Others': 'Other Skin Conditions'
        },
        'accuracy': {
            'Monkeypox': 95.0,
            'Others': 95.0
        },
        'overall_accuracy': 95.0,
        'input_size': 256,  # ✅ Changed from 224 to 256
        'name': 'Monkeypox Detection System'
    }
}


app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production-2026')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model caches
loaded_models = {}
capture = 0
camera = None
latest_capture = None

# ========== HELPER FUNCTIONS ==========
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model(model_type: str):
    """Load and cache model (H5 or TFLite) with multiple fallback strategies"""
    global loaded_models

    if model_type not in loaded_models:
        model_config = MODELS[model_type]
        model_path = model_config['path']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"📦 Loading {model_type} model from: {model_path}")

        if model_config['type'] == 'h5':
            import tensorflow as tf

            # ✅ Try multiple loading strategies
            model = None
            strategies = [
                ("compile=False", lambda: tf.keras.models.load_model(model_path, compile=False)),
                ("compile=False + safe_mode=False", lambda: tf.keras.models.load_model(model_path, compile=False, safe_mode=False)),
                ("TF1 compatibility mode", lambda: load_with_tf1_compat(model_path)),
            ]

            last_error = None
            for strategy_name, load_fn in strategies:
                try:
                    print(f"   Trying: {strategy_name}...")
                    model = load_fn()
                    print(f"   ✅ Success with {strategy_name}!")
                    break
                except Exception as e:
                    last_error = e
                    print(f"   ❌ Failed: {str(e)[:150]}")

            if model is None:
                raise Exception(f"All loading strategies failed. Last error: {last_error}")

            # Warm-up prediction
            input_size = model_config['input_size']
            dummy = np.zeros((1, input_size, input_size, 3), dtype=np.float32)
            _ = model.predict(dummy, verbose=0)
            del dummy

            loaded_models[model_type] = {
                'model': model,
                'type': 'h5'
            }

            print(f"✅ H5 model loaded: {model_config['name']}")
            print(f"   Accuracy: {model_config['overall_accuracy']}%")
            print(f"   Classes: {len(model_config['classes'])}")
            print(f"   Input size: {input_size}x{input_size}")

        elif model_config['type'] == 'tflite':
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Auto-detect actual input size
            input_shape = input_details[0]['shape']
            actual_input_size = input_shape[1]

            print(f"   Detected input shape: {input_shape}")

            if actual_input_size != model_config['input_size']:
                print(f"   ⚠️ Size mismatch! Config: {model_config['input_size']}, Actual: {actual_input_size}")
                print(f"   Using actual size: {actual_input_size}")
                model_config['input_size'] = actual_input_size

            # Warm-up
            dummy = np.zeros(input_shape, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], dummy)
            interpreter.invoke()
            del dummy

            loaded_models[model_type] = {
                'interpreter': interpreter,
                'input_details': input_details,
                'output_details': output_details,
                'type': 'tflite'
            }

            print(f"✅ TFLite model loaded: {model_config['name']}")
            print(f"   Input shape: {input_shape}")
            print(f"   Classes: {len(model_config['classes'])}")

        gc.collect()

    return loaded_models[model_type]


def load_with_tf1_compat(model_path):
    """Try loading with TF1 compatibility mode"""
    import tensorflow as tf

    # Disable TF2 behavior temporarily
    try:
        import tensorflow.compat.v1 as tf1
        tf1.disable_v2_behavior()
    except:
        pass

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=None
        )
        return model
    finally:
        # Re-enable TF2 behavior
        try:
            import tensorflow.compat.v1 as tf1
            tf1.enable_v2_behavior()
        except:
            pass

def load_image(img_path: str, input_size: int) -> np.ndarray:
    """Preprocess image for model input"""
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((input_size, input_size))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def predict_path(img_path: str, model_type: str) -> str:
    """Universal prediction function for both models"""
    try:
        import time
        start = time.time()

        model_config = MODELS[model_type]
        model_cache = get_model(model_type)
        input_size = model_config['input_size']

        print(f"🔍 Prediction details:")
        print(f"   Model: {model_type}")
        print(f"   Input size: {input_size}x{input_size}")

        # Load and preprocess image
        img = load_image(img_path, input_size)
        print(f"   Preprocessed image shape: {img.shape}")

        # Run prediction based on model type
        if model_cache['type'] == 'h5':
            predictions = model_cache['model'].predict(img, verbose=0)[0]
        elif model_cache['type'] == 'tflite':
            interpreter = model_cache['interpreter']
            input_details = model_cache['input_details']
            output_details = model_cache['output_details']

            # ✅ Verify shape match
            expected_shape = input_details[0]['shape']
            print(f"   Expected TFLite shape: {expected_shape}")

            if img.shape != tuple(expected_shape):
                raise ValueError(f"Shape mismatch! Got {img.shape}, expected {expected_shape}")

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        elapsed = time.time() - start
        print(f"⚡ {model_type} prediction: {elapsed:.3f}s")

        del img
        gc.collect()

        # Format results
        result = format_prediction_result(predictions, model_config, model_type)
        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Prediction error:\n{error_details}")
        gc.collect()
        return f"<div style='background: #f8d7da; padding: 15px; border-radius: 10px; color: #721c24;'><b>Analysis error:</b> {str(e)}</div>"
def format_prediction_result(predictions: np.ndarray, model_config: dict, model_type: str) -> str:
    """Format prediction results with perfect centering"""

    # Validation checks
    if predictions is None or len(predictions) == 0:
        return format_error("Empty predictions received from model")

    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    class_names = model_config['classes']
    descriptions = model_config['descriptions']
    accuracies = model_config['accuracy']

    if len(predictions) != len(class_names):
        return format_error(f"Prediction mismatch: expected {len(class_names)}, got {len(predictions)}")

    try:
        pred_class_idx = np.argmax(predictions)
    except Exception as e:
        return format_error(f"Could not determine prediction: {str(e)}")

    if pred_class_idx >= len(class_names):
        return format_error(f"Invalid prediction index: {pred_class_idx}")

    pred_class = class_names[pred_class_idx]
    confidence = predictions[pred_class_idx] * 100
    description = descriptions[pred_class]
    class_acc = accuracies[pred_class]

    # ✅ PERFECTLY CENTERED CONTAINER
    result = """
    <div style='max-width: 600px; width: 100%; margin: 0 auto; padding: 0 20px; box-sizing: border-box; text-align: center;'>
    """

    # ✅ Model badge - centered
    result += f"""
    <div style='display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 20px;
                background: #f0fdfa; border: 1px solid rgba(20, 184, 166, 0.2); margin-bottom: 24px;'>
        <svg style='width: 14px; height: 14px; color: #14b8a6; flex-shrink: 0;' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2'
                  d='M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z'/>
        </svg>
        <span style='font-size: 13px; font-weight: 400; color: #14b8a6; white-space: nowrap;'>{model_config['name']}</span>
    </div>
    """

    # ✅ Diagnosis - centered
    result += f"""
    <h3 style='font-size: 24px; font-weight: 400; color: #0a0a0a; margin: 0 0 8px 0;
               letter-spacing: -0.5px; line-height: 1.3; text-align: center;'>
        {description}
    </h3>
    <p style='font-size: 13px; color: #737373; margin: 0 0 16px 0; font-weight: 300; line-height: 1.5; text-align: center;'>
        Accuracy: {class_acc}% • Confidence: {confidence:.1f}%
    </p>

    <div style='font-size: 56px; font-weight: 300; color: #14b8a6; line-height: 1; margin-bottom: 24px; text-align: center;'>
        {confidence:.1f}<span style='font-size: 28px; color: #737373;'>%</span>
    </div>
    """

    # ✅ Risk assessment - centered
    if model_type == 'ham10000':
        if pred_class in ['mel', 'bcc']:
            result += """
            <div style='background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                        padding: 16px 20px; border-radius: 12px; border-left: 4px solid #dc2626;
                        margin-bottom: 24px; text-align: left;'>
                <div style='display: flex; align-items: flex-start; gap: 12px;'>
                    <span style='font-size: 20px; flex-shrink: 0; line-height: 1;'>⚠️</span>
                    <div style='flex: 1; min-width: 0;'>
                        <p style='margin: 0; font-size: 14px; font-weight: 600; color: #7f1d1d; line-height: 1.4;'>
                            HIGH RISK
                        </p>
                        <p style='margin: 6px 0 0 0; font-size: 13px; font-weight: 400; color: #991b1b; line-height: 1.6;'>
                            Immediate dermatologist consultation strongly recommended.
                        </p>
                    </div>
                </div>
            </div>
            """
        elif pred_class in ['akiec', 'df']:
            result += """
            <div style='background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                        padding: 16px 20px; border-radius: 12px; border-left: 4px solid #f59e0b;
                        margin-bottom: 24px; text-align: left;'>
                <div style='display: flex; align-items: flex-start; gap: 12px;'>
                    <span style='font-size: 20px; flex-shrink: 0; line-height: 1;'>⚠️</span>
                    <div style='flex: 1; min-width: 0;'>
                        <p style='margin: 0; font-size: 14px; font-weight: 600; color: #78350f; line-height: 1.4;'>
                            MODERATE RISK
                        </p>
                        <p style='margin: 6px 0 0 0; font-size: 13px; font-weight: 400; color: #92400e; line-height: 1.6;'>
                            Schedule dermatologist checkup within 2-4 weeks.
                        </p>
                    </div>
                </div>
            </div>
            """
        else:
            result += """
            <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                        padding: 16px 20px; border-radius: 12px; border-left: 4px solid #16a34a;
                        margin-bottom: 24px; text-align: left;'>
                <div style='display: flex; align-items: flex-start; gap: 12px;'>
                    <span style='font-size: 20px; flex-shrink: 0; line-height: 1;'>✅</span>
                    <div style='flex: 1; min-width: 0;'>
                        <p style='margin: 0; font-size: 14px; font-weight: 600; color: #14532d; line-height: 1.4;'>
                            LOW RISK
                        </p>
                        <p style='margin: 6px 0 0 0; font-size: 13px; font-weight: 400; color: #166534; line-height: 1.6;'>
                            Continue regular monitoring. Consult if changes occur.
                        </p>
                    </div>
                </div>
            </div>
            """

    elif model_type == 'monkeypox':
        if pred_class == 'Monkeypox':
            result += """
            <div style='background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                        padding: 16px 20px; border-radius: 12px; border-left: 4px solid #dc2626;
                        margin-bottom: 24px; text-align: left;'>
                <div style='display: flex; align-items: flex-start; gap: 12px;'>
                    <span style='font-size: 20px; flex-shrink: 0; line-height: 1;'>⚠️</span>
                    <div style='flex: 1; min-width: 0;'>
                        <p style='margin: 0; font-size: 14px; font-weight: 600; color: #7f1d1d; line-height: 1.4;'>
                            POSITIVE DETECTION
                        </p>
                        <p style='margin: 6px 0 0 0; font-size: 13px; font-weight: 400; color: #991b1b; line-height: 1.6;'>
                            Seek immediate medical attention. Isolate and contact healthcare provider.
                        </p>
                    </div>
                </div>
            </div>
            """
        else:
            result += """
            <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                        padding: 16px 20px; border-radius: 12px; border-left: 4px solid #16a34a;
                        margin-bottom: 24px; text-align: left;'>
                <div style='display: flex; align-items: flex-start; gap: 12px;'>
                    <span style='font-size: 20px; flex-shrink: 0; line-height: 1;'>✅</span>
                    <div style='flex: 1; min-width: 0;'>
                        <p style='margin: 0; font-size: 14px; font-weight: 600; color: #14532d; line-height: 1.4;'>
                            NEGATIVE
                        </p>
                        <p style='margin: 6px 0 0 0; font-size: 13px; font-weight: 400; color: #166534; line-height: 1.6;'>
                            No monkeypox detected. Consult doctor if symptoms persist.
                        </p>
                    </div>
                </div>
            </div>
            """

    # ✅ Alternative predictions - left-aligned content, centered section
    if len(class_names) > 2:
        top_3_idx = np.argsort(predictions)[::-1][:3]

        result += """
        <div style='margin-bottom: 24px; text-align: left;'>
            <h4 style='font-size: 15px; font-weight: 500; color: #525252; margin: 0 0 16px 0;'>
                Alternative Possibilities
            </h4>
        """

        for idx in top_3_idx[1:]:
            alt_class = class_names[idx]
            alt_desc = descriptions[alt_class]
            alt_conf = predictions[idx] * 100
            alt_acc = accuracies[alt_class]

            result += f"""
            <div style='display: flex; justify-content: space-between; align-items: center;
                        padding: 12px 16px; margin-bottom: 8px; background: #fafafa;
                        border-radius: 10px; border: 1px solid #f5f5f5; gap: 16px;'>
                <div style='flex: 1; min-width: 0;'>
                    <p style='margin: 0; font-size: 14px; font-weight: 400; color: #0a0a0a; line-height: 1.4;'>
                        {alt_desc}
                    </p>
                    <p style='margin: 4px 0 0 0; font-size: 12px; font-weight: 300; color: #737373; line-height: 1.3;'>
                        Accuracy: {alt_acc}%
                    </p>
                </div>
                <div style='font-size: 18px; font-weight: 400; color: #14b8a6; flex-shrink: 0; white-space: nowrap;'>
                    {alt_conf:.1f}<span style='font-size: 13px; color: #737373;'>%</span>
                </div>
            </div>
            """

        result += "</div>"

    # ✅ Binary probabilities
    if len(class_names) == 2:
        result += """
        <div style='margin-bottom: 24px; text-align: left;'>
            <h4 style='font-size: 15px; font-weight: 500; color: #525252; margin: 0 0 16px 0;'>
                Detection Probabilities
            </h4>
        """

        for i, class_name in enumerate(class_names):
            prob = predictions[i] * 100
            desc = descriptions[class_name]

            if i == pred_class_idx:
                bg_style = "background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); border: 1px solid rgba(20, 184, 166, 0.3);"
                text_color = "#14b8a6"
            else:
                bg_style = "background: #fafafa; border: 1px solid #f5f5f5;"
                text_color = "#737373"

            result += f"""
            <div style='display: flex; justify-content: space-between; align-items: center;
                        padding: 12px 16px; margin-bottom: 8px; border-radius: 10px; {bg_style} gap: 16px;'>
                <p style='margin: 0; font-size: 14px; font-weight: 400; color: #0a0a0a; flex: 1; min-width: 0;'>
                    {desc}
                </p>
                <p style='margin: 0; font-size: 18px; font-weight: 400; color: {text_color}; flex-shrink: 0; white-space: nowrap;'>
                    {prob:.1f}<span style='font-size: 13px;'>%</span>
                </p>
            </div>
            """

        result += "</div>"

    # ✅ Disclaimer
    result += f"""
    <div style='background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                padding: 16px 20px; border-radius: 12px; border-left: 4px solid #f59e0b;
                margin-top: 24px; text-align: left;'>
        <div style='display: flex; gap: 12px; align-items: flex-start;'>
            <span style='font-size: 20px; flex-shrink: 0; line-height: 1;'>⚠️</span>
            <div style='flex: 1; min-width: 0;'>
                <p style='margin: 0 0 6px 0; font-size: 13px; font-weight: 600; color: #78350f; line-height: 1.4;'>
                    Medical Disclaimer
                </p>
                <p style='margin: 0; font-size: 12px; font-weight: 400; color: #92400e; line-height: 1.6;'>
                    This AI tool has {model_config['overall_accuracy']}% accuracy and is for educational/screening purposes only.
                    It is NOT a diagnostic tool. Always consult a qualified healthcare provider for proper medical evaluation.
                </p>
            </div>
        </div>
    </div>
    </div>
    """

    return result


def gen_frames():
    """Generate camera frames for video feed"""
    global capture, latest_capture

    if not init_camera():
        # Return placeholder image when camera not available
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Not Available", (120, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, "Check camera permissions", (120, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.putText(placeholder, "in System Settings", (120, 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

        ret, buffer = cv2.imencode('.jpg', placeholder)
        frame_bytes = buffer.tobytes()

        # Keep sending the same placeholder
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            import time
            time.sleep(0.1)
        return

    frame_count = 0
    while True:
        try:
            success, frame = camera.read()

            if not success or frame is None:
                print("⚠️ Failed to read frame, reinitializing camera...")
                if init_camera():
                    continue
                else:
                    # Camera lost, return placeholder
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Camera Connection Lost", (100, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    import time
                    time.sleep(1)
                    continue

            # Handle capture request
            if capture:
                capture = 0
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Save the non-flipped frame for analysis
                cv2.imwrite(filepath, frame)
                latest_capture = filename
                print(f"📸 Captured: {filename}")

            # Flip frame for display (mirror effect)
            frame_flipped = cv2.flip(frame, 1)

            # Add frame counter overlay (optional)
            frame_count += 1
            cv2.putText(frame_flipped, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', frame_flipped, [cv2.IMWRITE_JPEG_QUALITY, 85])

            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except GeneratorExit:
            print("📷 Video feed closed by client")
            break
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            import time
            time.sleep(0.1)
            continue


def release_camera():
    """Safely release camera resources"""
    global camera
    try:
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None
            print("📷 Camera released")
    except Exception as e:
        print(f"⚠️ Error releasing camera: {e}")
        camera = None

# ========== ROUTES ==========
@app.route("/", methods=['GET'])
@app.route("/index", methods=['GET'])
def index():
    if request.method == 'GET' and 'from_prediction' not in session:
        session.clear()
    session.pop('from_prediction', None)
    return render_template('index.html', models=MODELS)

@app.route("/predicts", methods=['GET', 'POST'])
def predicts():
    if request.method == 'GET':
        return redirect(url_for('index'))

    # ✅ Get selected model type
    model_type = request.form.get('model_type', 'ham10000')

    if model_type not in MODELS:
        flash('Invalid model selected')
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

            # ✅ Use selected model
            product = predict_path(file_path, model_type)
            user_image = url_for('static', filename=f'Data/{unique_filename}')

            session['from_prediction'] = True

            return render_template('index.html',
                                 product=product,
                                 user_image=user_image,
                                 models=MODELS,
                                 selected_model=model_type)
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
            model_type = request.form.get('model_type', 'ham10000')

            # Check if camera is available
            if not init_camera():
                flash('Camera not available. Please check permissions in System Settings.')
                return redirect(url_for('index'))

            # Trigger capture
            capture = 1

            # Wait for capture to complete
            import time
            max_wait = 3.0  # Wait up to 3 seconds
            wait_time = 0
            while latest_capture is None and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1

            if latest_capture:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_capture)

                if os.path.exists(img_path):
                    product = predict_path(img_path, model_type)
                    user_image = url_for('static', filename=f'Data/{latest_capture}')

                    session['from_prediction'] = True

                    # Reset latest_capture for next use
                    latest_capture = None

                    return render_template('index.html',
                                         product=product,
                                         user_image=user_image,
                                         models=MODELS,
                                         selected_model=model_type)
                else:
                    flash('Capture failed: File not saved')
            else:
                flash('Capture timeout: No image captured')

    return redirect(url_for('index'))


@app.route('/health')
def health():
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        status = {
            'status': 'healthy',
            'loaded_models': list(loaded_models.keys()),
            'available_models': list(MODELS.keys()),
            'model_configs': {
                name: {
                    'type': config['type'],
                    'accuracy': config['overall_accuracy'],
                    'classes': len(config['classes']),
                    'path': config['path']
                }
                for name, config in MODELS.items()
            },
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
    return render_template('index.html', models=MODELS), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', models=MODELS), 500

# Pre-load models at startup (optional)
with app.app_context():
    print("\n" + "="*60)
    print("🚀 MULTI-DISEASE SKIN CLASSIFIER")
    print("="*60)
    print("\n📋 Available Models:")
    for name, config in MODELS.items():
        exists = "✅" if os.path.exists(config['path']) else "❌"
        print(f"   {exists} {config['name']}")
        print(f"     Type: {config['type'].upper()}")
        print(f"     Accuracy: {config['overall_accuracy']}%")
        print(f"     Classes: {len(config['classes'])}")
        print(f"     Path: {config['path']}")
    print("\n💡 Models will load on first prediction")
    print("="*60 + "\n")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🌐 Server starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
