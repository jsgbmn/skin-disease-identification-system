# convert_ham10000.py - Convert HAM10000 model to compatible format
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

print("="*70)
print("HAM10000 MODEL CONVERTER")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print("="*70)

# Paths
h5_path = './HAM10000_MobileNetV2_FINAL.h5'
output_dir = './'

print(f"\n📦 Loading model from: {h5_path}")

if not os.path.exists(h5_path):
    print(f"❌ File not found: {h5_path}")
    exit(1)

model = None
successful_strategy = None

# ===== Strategy 1: Direct load with compile=False =====
print("\n🔄 Strategy 1: Direct load (compile=False)")
try:
    model = tf.keras.models.load_model(h5_path, compile=False)
    successful_strategy = "Strategy 1: Direct load"
    print("   ✅ Success!")
except Exception as e:
    print(f"   ❌ Failed: {str(e)[:150]}")

# ===== Strategy 2: Load with safe_mode=False =====
if model is None:
    print("\n🔄 Strategy 2: Load with safe_mode=False")
    try:
        model = tf.keras.models.load_model(h5_path, compile=False, safe_mode=False)
        successful_strategy = "Strategy 2: safe_mode=False"
        print("   ✅ Success!")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:150]}")

# ===== Strategy 3: Rebuild architecture + load weights =====
if model is None:
    print("\n🔄 Strategy 3: Rebuild architecture + load weights")
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model

        print("   Building MobileNetV2 architecture...")
        base_model = MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(128, 128, 3)
        )

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        predictions = Dense(7, activation='softmax', name='predictions')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        print("   Loading weights...")
        model.load_weights(h5_path)
        successful_strategy = "Strategy 3: Architecture rebuild"
        print("   ✅ Success!")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:150]}")

# Check if model loaded
if model is None:
    print("\n" + "="*70)
    print("❌ ALL STRATEGIES FAILED")
    print("="*70)
    print("\nPossible solutions:")
    print("1. Try TensorFlow 2.13: pip install tensorflow==2.13.0")
    print("2. Try TensorFlow 2.10: pip install tensorflow==2.10.0")
    print("3. Re-train the model with current TensorFlow version")
    exit(1)

# ===== Model loaded successfully =====
print("\n" + "="*70)
print(f"✅ MODEL LOADED SUCCESSFULLY ({successful_strategy})")
print("="*70)

print(f"\n📊 Model Information:")
print(f"   Input shape:  {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Parameters:   {model.count_params():,}")
print(f"   Layers:       {len(model.layers)}")

# ===== Test prediction =====
print("\n🧪 Testing prediction...")
try:
    test_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
    predictions = model.predict(test_input, verbose=0)
    print(f"   ✅ Prediction works! Output shape: {predictions.shape}")
    print(f"   Sample output: {predictions[0][:3]}... (sum={predictions[0].sum():.4f})")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    exit(1)

# ===== Save in multiple formats =====
print("\n💾 Saving in multiple formats...")

saved_formats = []

# Format 1: Re-save as H5 with TF 2.15
print("\n1️⃣  H5 format (re-saved with TF 2.15)")
h5_new_path = os.path.join(output_dir, 'HAM10000_MobileNetV2_TF2_15.h5')
try:
    model.save(h5_new_path, save_format='h5')
    print(f"   ✅ Saved: {h5_new_path}")
    saved_formats.append(('H5 (TF 2.15)', h5_new_path, 'h5'))
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Format 2: Keras format (.keras)
print("\n2️⃣  Keras format (.keras)")
keras_path = os.path.join(output_dir, 'HAM10000_MobileNetV2_TF2_15.keras')
try:
    model.save(keras_path, save_format='keras')
    print(f"   ✅ Saved: {keras_path}")
    saved_formats.append(('Keras', keras_path, 'keras'))
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Format 3: SavedModel format
print("\n3️⃣  SavedModel format")
saved_model_path = os.path.join(output_dir, 'HAM10000_saved_model')
try:
    tf.saved_model.save(model, saved_model_path)
    print(f"   ✅ Saved: {saved_model_path}")
    saved_formats.append(('SavedModel', saved_model_path, 'saved_model'))
except Exception as e:
    print(f"   ❌ Failed: {e}")

# ===== Summary =====
print("\n" + "="*70)
print("✅ CONVERSION COMPLETE")
print("="*70)

if saved_formats:
    print("\n📁 Saved formats:")
    for i, (format_name, path, format_type) in enumerate(saved_formats, 1):
        print(f"\n{i}. {format_name}")
        print(f"   Path: {path}")
        print(f"   Type: '{format_type}'")

    print("\n" + "="*70)
    print("📝 UPDATE YOUR app.py:")
    print("="*70)
    recommended = saved_formats[0]  # Use first successful format
    print(f"\nMODELS = {{")
    print(f"    'ham10000': {{")
    print(f"        'path': '{recommended[1]}',")
    print(f"        'type': '{recommended[2]}',")
    print(f"        # ... rest of config")
    print(f"    }}")
    print(f"}}")
    print("="*70)
else:
    print("\n❌ No formats were saved successfully!")

print("\n🎉 Done!")
