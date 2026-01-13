import tensorflow as tf
import os
import numpy as np

# Use absolute path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'models')
h5_path = os.path.join(model_dir, 'model.h5')
tflite_path = os.path.join(model_dir, 'model.tflite')

print(f"H5 Model: {h5_path}")
print(f"TFLite Output: {tflite_path}")

# Load H5 model
print(f"\nLoading H5 model...")
try:
    model = tf.keras.models.load_model(h5_path, compile=False)
    print("✅ Model loaded successfully")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Test model
try:
    dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
    output = model.predict(dummy, verbose=0)
    print(f"✅ Model test passed: {output.shape}")
except Exception as e:
    print(f"❌ Model test failed: {e}")
    exit(1)

# Convert to TFLite
print("\nConverting to TFLite...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()
    print("✅ Conversion successful")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    exit(1)

# Save TFLite model (NOT model.h5!)
try:
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite saved to: {tflite_path}")
except Exception as e:
    print(f"❌ Failed to save: {e}")
    exit(1)

# Verify
print("\nVerifying TFLite model...")
try:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"✅ Verified")
    print(f"   Input: {input_details[0]['shape']}")
    print(f"   Output: {output_details[0]['shape']}")

    # Test
    test_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    print(f"✅ Test prediction passed")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    exit(1)

# Show sizes
h5_size = os.path.getsize(h5_path) / (1024 * 1024)
tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"\n📦 H5 Model: {h5_size:.2f} MB")
print(f"📦 TFLite Model: {tflite_size:.2f} MB")
print(f"\n✅ Conversion complete! Both files preserved.")
