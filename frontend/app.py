# app.py - Multi-Disease Skin Classifier (HAM10000 + Monkeypox)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '2'

from flask import Flask, flash, render_template, request, redirect, url_for, session
import datetime
import numpy as np
from werkzeug.utils import secure_filename
import gc
from pathlib import Path
import base64

# ========== CONFIGURATION ==========
UPLOAD_FOLDER = './static/Data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def find_models_dir():
    app_path = Path(__file__).resolve()
    candidates = [
        app_path.parent / 'backend' / 'models',
        app_path.parent.parent / 'backend' / 'models',
    ]
    for path in candidates:
        if path.exists() and (any(path.glob('*.h5')) or any(path.glob('*.tflite'))):
            return path
    return candidates[0]

MODELS_DIR = find_models_dir()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    'ham10000': {
        'path': str(MODELS_DIR / 'cancer_model.h5'),
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
        'path': str(MODELS_DIR / 'model.tflite'),
        'type': 'tflite',
        'binary_output': True,  # ✅ NEW: Indicates single output value
        'threshold': 0.5,  # ✅ NEW: Threshold for classification
        'classes': ['Monkeypox', 'Not Monkeypox'],
        'descriptions': {
            'Monkeypox': 'Monkeypox Virus Infection',
            'Not Monkeypox': 'Other Skin Conditions'
        },
        'accuracy': {'Monkeypox': 95.0, 'Not Monkeypox': 95.0},
        'overall_accuracy': 95.0,
        'input_size': 256,
        'name': 'Monkeypox Detection System'
    }
}

app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production-2026')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
loaded_models = {}

# ========== HELPER FUNCTIONS ==========
def format_error(error_message: str) -> str:
    return f"""
    <div style='max-width: 600px; width: 100%; margin: 0 auto; padding: 20px; box-sizing: border-box; text-align: center;'>
        <div style='background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                    padding: 24px; border-radius: 16px; border-left: 4px solid #dc2626;'>
            <div style='display: flex; flex-direction: column; align-items: center; gap: 16px;'>
                <svg style='width: 56px; height: 56px; color: #dc2626;' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                    <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2'
                          d='M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'/>
                </svg>
                <div>
                    <h3 style='font-size: 20px; font-weight: 600; color: #7f1d1d; margin: 0 0 8px 0;'>Analysis Error</h3>
                    <p style='font-size: 14px; color: #991b1b; margin: 0; line-height: 1.6;'>{error_message}</p>
                </div>
            </div>
        </div>
        <div style='margin-top: 20px; text-align: center;'>
            <button onclick='window.location.href="/"'
                    style='padding: 12px 24px; background: linear-gradient(to right, #14b8a6, #0d9488);
                           color: white; border: none; border-radius: 10px; font-size: 14px;
                           font-weight: 500; cursor: pointer;'>
                Try Another Image
            </button>
        </div>
    </div>
    """

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model(model_type: str):
    global loaded_models

    if model_type not in loaded_models:
        model_config = MODELS[model_type]
        model_path = model_config['path']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_config['type'] == 'h5':
            import tensorflow as tf
            print(f"📦 Loading {model_type} H5 model from: {model_path}")

            model = None
            strategies = [
                lambda: tf.keras.models.load_model(model_path, compile=False),
                lambda: tf.keras.models.load_model(model_path, compile=False, safe_mode=False),
            ]

            for i, load_fn in enumerate(strategies):
                try:
                    model = load_fn()
                    print(f"✅ H5 model loaded successfully")
                    break
                except Exception as e:
                    print(f"⚠️ Strategy {i+1} failed: {str(e)}")
                    continue

            if model is None:
                raise Exception("Failed to load H5 model")

            input_size = model_config['input_size']
            dummy = np.zeros((1, input_size, input_size, 3), dtype=np.float32)
            _ = model.predict(dummy, verbose=0)
            del dummy

            loaded_models[model_type] = {'model': model, 'type': 'h5'}

        elif model_config['type'] == 'tflite':
            import tensorflow as tf
            print(f"📦 Loading {model_type} TFLite model from: {model_path}")

            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]['shape']

            print(f"✅ TFLite model loaded")
            print(f"   Input shape: {input_shape}")
            print(f"   Output shape: {output_details[0]['shape']}")

            if input_shape[1] != model_config['input_size']:
                print(f"⚠️ Adjusting input size: {model_config['input_size']} → {input_shape[1]}")
                model_config['input_size'] = input_shape[1]

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

        gc.collect()

    return loaded_models[model_type]

def load_image(img_path: str, input_size: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((input_size, input_size))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_path(img_path: str, model_type: str) -> str:
    try:
        print(f"\n{'='*60}")
        print(f"🔍 Prediction Request: {model_type}")
        print(f"{'='*60}")

        model_config = MODELS[model_type]
        model_cache = get_model(model_type)
        input_size = model_config['input_size']

        img = load_image(img_path, input_size)
        print(f"✅ Image preprocessed: {img.shape}")

        if model_cache['type'] == 'h5':
            predictions = model_cache['model'].predict(img, verbose=0)[0]
            print(f"✅ H5 Predictions: {predictions}")

        elif model_cache['type'] == 'tflite':
            interpreter = model_cache['interpreter']
            input_details = model_cache['input_details']
            output_details = model_cache['output_details']

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            print(f"✅ TFLite Raw Predictions: {predictions}")
            print(f"   Type: {type(predictions)}, Shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")

        del img
        gc.collect()

        # ✅ FIXED: Handle binary output (single value)
        if model_config.get('binary_output', False):
            print(f"🔍 Processing binary output...")

            # Robust extraction of single value
            predictions_array = np.array(predictions).flatten()

            if len(predictions_array) == 1:
                # Single output value
                prob = float(predictions_array[0])
                print(f"✅ Binary prediction value: {prob}")

                # Convert to [Monkeypox, Not Monkeypox] probabilities
                # If prob < 0.5: Monkeypox (index 0), else: Not Monkeypox (index 1)
                predictions = np.array([prob, 1.0 - prob])
                print(f"✅ Converted to probabilities: {predictions}")
            else:
                # Already has 2 or more values
                print(f"⚠️ Expected 1 output, got {len(predictions_array)}")
                if len(predictions_array) >= 2:
                    predictions = predictions_array[:2]
                else:
                    raise ValueError(f"Invalid binary model output: {predictions_array}")

        print(f"✅ Final predictions: {predictions} (shape: {predictions.shape})")
        print(f"{'='*60}\n")
        return format_prediction_result(predictions, model_config, model_type)

    except FileNotFoundError as e:
        print(f"❌ File Error: {str(e)}")
        return format_error(f"Model file not found: {str(e)}")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Prediction Error:\n{error_trace}")
        gc.collect()
        return format_error(f"Prediction failed: {str(e)}")

def format_prediction_result(predictions: np.ndarray, model_config: dict, model_type: str) -> str:
    try:
        if predictions is None or len(predictions) == 0:
            return format_error("Empty predictions received from model")

        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        class_names = model_config['classes']
        descriptions = model_config['descriptions']
        accuracies = model_config['accuracy']

        pred_class_idx = np.argmax(predictions)
        pred_class = class_names[pred_class_idx]
        confidence = predictions[pred_class_idx] * 100
        description = descriptions[pred_class]
        class_acc = accuracies[pred_class]

        result = f"""
        <div style='max-width: 600px; width: 100%; margin: 0 auto; padding: 0 20px; box-sizing: border-box; text-align: center;'>
            <div style='display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 20px;
                        background: #f0fdfa; border: 1px solid rgba(20, 184, 166, 0.2); margin-bottom: 24px;'>
                <svg style='width: 14px; height: 14px; color: #14b8a6; flex-shrink: 0;' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                    <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2'
                          d='M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z'/>
                </svg>
                <span style='font-size: 13px; font-weight: 400; color: #14b8a6;'>{model_config['name']}</span>
            </div>

            <h3 style='font-size: 24px; font-weight: 400; color: #0a0a0a; margin: 0 0 8px 0;'>{description}</h3>
            <p style='font-size: 13px; color: #737373; margin: 0 0 16px 0;'>Accuracy: {class_acc}% • Confidence: {confidence:.1f}%</p>

            <div style='font-size: 56px; font-weight: 300; color: #14b8a6; margin-bottom: 24px;'>
                {confidence:.1f}<span style='font-size: 28px; color: #737373;'>%</span>
            </div>
        """

        # Risk assessment
        if model_type == 'ham10000':
            if pred_class in ['mel', 'bcc']:
                result += """
                <div style='background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                            padding: 16px 20px; border-radius: 12px; border-left: 4px solid #dc2626; margin-bottom: 24px; text-align: left;'>
                    <div style='display: flex; gap: 12px;'>
                        <span style='font-size: 20px;'>⚠️</span>
                        <div>
                            <p style='margin: 0; font-size: 14px; font-weight: 600; color: #7f1d1d;'>HIGH RISK</p>
                            <p style='margin: 6px 0 0 0; font-size: 13px; color: #991b1b;'>Immediate dermatologist consultation strongly recommended.</p>
                        </div>
                    </div>
                </div>
                """
            elif pred_class in ['akiec', 'df']:
                result += """
                <div style='background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                            padding: 16px 20px; border-radius: 12px; border-left: 4px solid #f59e0b; margin-bottom: 24px; text-align: left;'>
                    <div style='display: flex; gap: 12px;'>
                        <span style='font-size: 20px;'>⚠️</span>
                        <div>
                            <p style='margin: 0; font-size: 14px; font-weight: 600; color: #78350f;'>MODERATE RISK</p>
                            <p style='margin: 6px 0 0 0; font-size: 13px; color: #92400e;'>Schedule dermatologist checkup within 2-4 weeks.</p>
                        </div>
                    </div>
                </div>
                """
            else:
                result += """
                <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                            padding: 16px 20px; border-radius: 12px; border-left: 4px solid #16a34a; margin-bottom: 24px; text-align: left;'>
                    <div style='display: flex; gap: 12px;'>
                        <span style='font-size: 20px;'>✅</span>
                        <div>
                            <p style='margin: 0; font-size: 14px; font-weight: 600; color: #14532d;'>LOW RISK</p>
                            <p style='margin: 6px 0 0 0; font-size: 13px; color: #166534;'>Continue regular monitoring. Consult if changes occur.</p>
                        </div>
                    </div>
                </div>
                """
        elif model_type == 'monkeypox':
            if pred_class == 'Monkeypox':
                result += """
                <div style='background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                            padding: 16px 20px; border-radius: 12px; border-left: 4px solid #dc2626; margin-bottom: 24px; text-align: left;'>
                    <div style='display: flex; gap: 12px;'>
                        <span style='font-size: 20px;'>⚠️</span>
                        <div>
                            <p style='margin: 0; font-size: 14px; font-weight: 600; color: #7f1d1d;'>POSITIVE DETECTION</p>
                            <p style='margin: 6px 0 0 0; font-size: 13px; color: #991b1b;'>It might be Monkeypox. You should visit a specialist immediately.</p>
                        </div>
                    </div>
                </div>
                """
            else:
                result += """
                <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                            padding: 16px 20px; border-radius: 12px; border-left: 4px solid #16a34a; margin-bottom: 24px; text-align: left;'>
                    <div style='display: flex; gap: 12px;'>
                        <span style='font-size: 20px;'>✅</span>
                        <div>
                            <p style='margin: 0; font-size: 14px; font-weight: 600; color: #14532d;'>NEGATIVE</p>
                            <p style='margin: 6px 0 0 0; font-size: 13px; color: #166534;'>It's most probably not monkeypox, but still you should visit a skin specialist.</p>
                        </div>
                    </div>
                </div>
                """

        # Alternative predictions (only for multi-class)
        if len(class_names) > 2:
            top_3_idx = np.argsort(predictions)[::-1][:3]
            result += "<div style='margin-bottom: 24px; text-align: left;'><h4 style='font-size: 15px; font-weight: 500; color: #525252; margin: 0 0 16px 0;'>Alternative Possibilities</h4>"

            for idx in top_3_idx[1:]:
                alt_class = class_names[idx]
                alt_desc = descriptions[alt_class]
                alt_conf = predictions[idx] * 100
                alt_acc = accuracies[alt_class]

                result += f"""
                <div style='display: flex; justify-content: space-between; padding: 12px 16px; margin-bottom: 8px;
                            background: #fafafa; border-radius: 10px; border: 1px solid #f5f5f5;'>
                    <div>
                        <p style='margin: 0; font-size: 14px; color: #0a0a0a;'>{alt_desc}</p>
                        <p style='margin: 4px 0 0 0; font-size: 12px; color: #737373;'>Accuracy: {alt_acc}%</p>
                    </div>
                    <div style='font-size: 18px; color: #14b8a6;'>{alt_conf:.1f}<span style='font-size: 13px; color: #737373;'>%</span></div>
                </div>
                """
            result += "</div>"

        # Binary probabilities
        if len(class_names) == 2:
            result += "<div style='margin-bottom: 24px; text-align: left;'><h4 style='font-size: 15px; font-weight: 500; color: #525252; margin: 0 0 16px 0;'>Detection Probabilities</h4>"

            for i, class_name in enumerate(class_names):
                prob = predictions[i] * 100
                desc = descriptions[class_name]

                bg = "background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);" if i == pred_class_idx else "background: #fafafa;"
                color = "#14b8a6" if i == pred_class_idx else "#737373"

                result += f"""
                <div style='display: flex; justify-content: space-between; padding: 12px 16px; margin-bottom: 8px;
                            {bg} border-radius: 10px; border: 1px solid #f5f5f5;'>
                    <p style='margin: 0; font-size: 14px; color: #0a0a0a;'>{desc}</p>
                    <p style='margin: 0; font-size: 18px; color: {color};'>{prob:.1f}<span style='font-size: 13px;'>%</span></p>
                </div>
                """
            result += "</div>"

        # Disclaimer
        result += """
        <div style='background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                    padding: 16px 20px; border-radius: 12px; border-left: 4px solid #f59e0b; margin-top: 24px; text-align: left;'>
            <div style='display: flex; gap: 12px;'>
                <span style='font-size: 20px;'>⚠️</span>
                <div>
                    <p style='margin: 0 0 6px 0; font-size: 13px; font-weight: 600; color: #78350f;'>Medical Disclaimer</p>
                    <p style='margin: 0; font-size: 12px; color: #92400e;'>
                        This AI tool is for educational purposes only. Always consult a healthcare provider for proper medical evaluation.
                    </p>
                </div>
            </div>
        </div>
        </div>
        """

        return result

    except Exception as e:
        import traceback
        print(f"❌ Format Error:\n{traceback.format_exc()}")
        return format_error(f"Failed to format results: {str(e)}")

# ========== ROUTES (same as before) ==========
@app.route("/", methods=['GET'])
@app.route("/index", methods=['GET'])
def index():
    if 'from_prediction' not in session:
        session.clear()
    session.pop('from_prediction', None)
    return render_template('index.html', models=MODELS)

@app.route("/predicts", methods=['GET', 'POST'])
def predicts():
    if request.method == 'GET':
        return redirect(url_for('index'))

    model_type = request.form.get('model_type', 'ham10000')

    if model_type not in MODELS:
        flash('Invalid model selected')
        return redirect(url_for('index'))

    image_data = request.form.get('image_data')
    if image_data:
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            if ',' in image_data:
                header, encoded = image_data.split(',', 1)
            else:
                encoded = image_data

            image_bytes = base64.b64decode(encoded)
            with open(file_path, 'wb') as f:
                f.write(image_bytes)

            product = predict_path(file_path, model_type)
            user_image = url_for('static', filename=f'Data/{filename}')
            session['from_prediction'] = True

            return render_template('index.html', product=product, user_image=user_image,
                                 models=MODELS, selected_model=model_type)
        except Exception as e:
            return render_template('index.html', product=format_error(f"Camera capture failed: {str(e)}"),
                                 models=MODELS), 500

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

            product = predict_path(file_path, model_type)
            user_image = url_for('static', filename=f'Data/{unique_filename}')
            session['from_prediction'] = True

            return render_template('index.html', product=product, user_image=user_image,
                                 models=MODELS, selected_model=model_type)
        except MemoryError:
            return render_template('index.html', product=format_error("Out of memory. Try a smaller image."),
                                 models=MODELS), 500
        except Exception as e:
            return render_template('index.html', product=format_error(f"Upload failed: {str(e)}"),
                                 models=MODELS), 500
        finally:
            gc.collect()

    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/health')
def health():
    try:
        import psutil
        memory_info = psutil.Process().memory_info()

        return {
            'status': 'healthy',
            'loaded_models': list(loaded_models.keys()),
            'available_models': list(MODELS.keys()),
            'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
            'timestamp': datetime.datetime.now().isoformat()
        }, 200
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}, 503

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', models=MODELS), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', product=format_error("Internal server error."), models=MODELS), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"\n{'='*60}")
    print(f"🚀 MULTI-DISEASE SKIN CLASSIFIER")
    print(f"{'='*60}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    print(f"📦 Available Models:")
    for model_name, config in MODELS.items():
        exists = "✅" if Path(config['path']).exists() else "❌"
        file_type = config['type'].upper()
        binary = " (Binary)" if config.get('binary_output') else ""
        num_classes = len(config['classes'])
        print(f"   {exists} {model_name} ({file_type}{binary}, {num_classes} classes): {Path(config['path']).name}")
    print(f"🌐 Server starting on port {port}...")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
