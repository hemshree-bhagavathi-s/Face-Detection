import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("face_detector.h5")

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Reduce model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("face_detector.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion completed!")
print("Saved as face_detector.tflite")