# pyrefly: ignore [missing-import]
import json
import logging
import base64
import io
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_helper")

def analyze_plant_with_gemini(
    api_key: str,
    leaf_image: Image.Image,
    soil_image: Image.Image = None,
    city: str = None,
    weather_data: dict = None,
    microclimate: dict = None,
    cnn_prediction: str = None
) -> dict:
    """
    Multimodal agronomic reasoning engine using Google's Gemini Vision API or Groq AI.
    Sends leaf (and soil) photo along with local weather and microclimate data
    to receive high-fidelity, structured JSON remedies.
    """
    if not api_key or api_key.strip() == "":
        raise ValueError("API Key is missing or invalid.")

    api_key_clean = api_key.strip()

    # Compile weather info
    weather_str = "None available"
    if weather_data:
        weather_str = f"Temp: {weather_data.get('temp', 25.0)}°C, Humidity: {weather_data.get('humidity', 60.0)}%, Conditions: {weather_data.get('desc', 'Clear')}"
        
    microclimate_str = "None available"
    if microclimate:
        microclimate_str = f"Sunlight: {microclimate.get('sunlight', 'N/A')}, Environment: {microclimate.get('environment', 'N/A')}, Soil Moisture: {microclimate.get('soil_moisture', 'N/A')}"

    # Build prompt
    prompt = f"""
You are an expert agronomic doctor, plant pathologist, and senior crop advisor.
Analyze the provided plant sample under these exact conditions:
- Target City / Location: {city or 'Unknown'}
- Ambient Weather Telemetry: {weather_str}
- Microclimate Calibration: {microclimate_str}
- Screen CNN Model Prediction: {cnn_prediction or 'None'}

Please execute a rigorous agronomic diagnostic check:
1. Identify the crop type and assess the foliage health.
2. Cross-reference the screen CNN model prediction with your deep generative computer vision reasoning.
3. Compute a final 'vigor_index' (0-100) reflecting the crop's dynamic vitality. Healthy plants score 85-100. Minor stress scores 60-80. Active pathogen outbreaks score 10-50.
4. Construct a relevant 'context_warning' if there is an environmental clash or pathogen propagation risk.
5. Generate real, high-quality, practical treatment remedies categorized into:
   - Biological & Organic Controls
   - Synthetic & Chemical Treatments
   - Cultural Management & Physical Prevention
   - Specialized Soil & Water Solutions

You MUST respond with a strictly formatted JSON object matching this schema.
Do NOT wrap in markdown code blocks like ```json. Output RAW JSON text directly.

JSON Schema:
{{
  "analyzed_crop": "string",
  "analyzed_disease": "string",
  "is_healthy": boolean,
  "confidence": float,
  "category": "string (Use EXACTLY one of: 'Fungal Outbreak 🍄', 'Bacterial Outbreak 🦠', 'Viral infection 🧬', 'Pest Propagation 🐛', 'Optimal Health ✅', 'Environmental Stress 🏜️')",
  "vigor_index": integer,
  "context_warning": {{
    "type": "string (one of: 'critical', 'anomaly', 'optimal', 'dehydration')",
    "title": "string (ALL CAPS short alert heading, e.g. CRITICAL FUNGAL OUTBREAK ACCELERATION)",
    "description": "string (1-2 sentences explaining contextual risk based on humidity, temperature, soil moisture)"
  }},
  "remedies": {{
    "organic": [
      "string (remedy 1)",
      "string (remedy 2)"
    ],
    "chemical": [
      "string (remedy 1)",
      "string (remedy 2)"
    ],
    "prevention": [
      "string (remedy 1)",
      "string (remedy 2)"
    ],
    "soil_water": [
      "string (remedy 1)",
      "string (remedy 2)"
    ]
  }}
}}
"""

    # Check API key type (Groq vs Gemini)
    if api_key_clean.startswith("gsk_"):
        try:
            import groq
            logger.info("Initializing Groq client with API key...")
            client = groq.Groq(api_key=api_key_clean)
            
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            text_response = response.choices[0].message.content.strip()
            if text_response.startswith("```"):
                lines = text_response.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                text_response = "\n".join(lines).strip()
                
            data = json.loads(text_response)
            logger.info("Successfully analyzed plant with Groq API.")
            return data
            
        except Exception as e:
            logger.error(f"Groq API invocation failed: {e}")
            raise RuntimeError(f"Groq AI reasoning failed: {str(e)}")
            
    else:
        # Standard Google Gemini API path
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key_clean)
            model_name = "gemini-2.5-flash"
            
            contents = [prompt]
            if isinstance(leaf_image, Image.Image):
                contents.append(leaf_image)
            if soil_image and isinstance(soil_image, Image.Image):
                contents.append(soil_image)

            logger.info(f"Initializing Gemini GenerativeModel with {model_name}...")
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                contents=contents,
                generation_config={"response_mime_type": "application/json"}
            )
            
            text_response = response.text.strip()
            if text_response.startswith("```"):
                lines = text_response.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                text_response = "\n".join(lines).strip()
                
            data = json.loads(text_response)
            logger.info("Successfully analyzed plant with Gemini API.")
            return data
            
        except Exception as e:
            logger.error(f"Gemini API invocation failed: {e}")
            raise RuntimeError(f"Gemini AI reasoning failed: {str(e)}")
