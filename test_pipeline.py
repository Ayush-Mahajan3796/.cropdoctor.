import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from remedies_data import remedies
from main import CLASS_NAMES, load_keras_model, analyze_soil_visual

sys.stdout.reconfigure(encoding='utf-8')

def run_test_diagnostic(leaf_path, soil_path=None):
    print("--- CropDoctor Diagnostic Pipeline Test ---")
    
    # 1. Load leaf image
    if not os.path.exists(leaf_path):
        print(f"Error: Leaf path {leaf_path} does not exist.")
        return
    
    leaf_img = Image.open(leaf_path).convert("RGB")
    print(f"Leaf image loaded: format={leaf_img.format}, size={leaf_img.size}")
    
    # 2. Run CNN Model Prediction
    model = load_keras_model()
    if model is None:
        print("Model is None!")
        return
    
    resized = leaf_img.resize((128, 128))
    img_arr = tf.keras.preprocessing.image.img_to_array(resized)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    predictions = model.predict(img_arr)
    result_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[result_idx]
    confidence = float(predictions[0][result_idx])
    
    print(f"\n[CNN Prediction]")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # 3. Fetch Remedies
    remedy_info = remedies.get(predicted_class, None)
    if remedy_info:
        print(f"\n[Remedies Info]")
        print(f"Category: {remedy_info.get('category')}")
        print(f"Symptoms: {remedy_info.get('symptoms')}")
        print(f"Organic Remedies: {remedy_info.get('organic')[:2]}")
        print(f"Chemical Treatments: {remedy_info.get('chemical')[:2]}")
    else:
        print(f"\n[Remedies Info] No remedies found for class '{predicted_class}'")
        
    # 4. Soil Telemetry if provided
    if soil_path and os.path.exists(soil_path):
        soil_res = analyze_soil_visual(soil_path)
        print(f"\n[Soil Telemetry]")
        print(f"Moisture: {soil_res['moisture']} (Score: {soil_res['moisture_score']})")
        print(f"Composition: {soil_res['composition']}")
        print(f"Luminance: {soil_res['brightness']} px")
        
    print("\nDiagnostic pipeline completed successfully!")

if __name__ == "__main__":
    run_test_diagnostic("sample_leaf.png", "sample_soil.png")
