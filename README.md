# 🌿 CropDoctor Pro (.cropdoctor.)

Multimodal Agronomy Suite for Precision Plant Disease Detection and Soil Telemetry.

## ✨ Features
- **38-Class Plant Disease Recognition**: Deep learning CNN model screening.
- **Multimodal AI Reasoning**: Integration with Groq & Google Gemini for dynamic disease analysis and actionable remedies.
- **Soil Telemetry**: Computer vision analysis of soil moisture, structure, and luminance.
- **Weather Synchronization**: Real-time ambient weather integration via Open-Meteo REST API.
- **Actionable Treatment Plans**: Detailed organic, chemical, and physical prevention advice.

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Configure API Key
Create a `.env` file or `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY="your_groq_api_key_here"
# or
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 3. Run the Web Application
```bash
streamlit run main.py
```
