# Multilingual Assistive Image Captioning Model

This project provides a **multilingual assistive model** designed to help visually impaired users by describing images in multiple Indian languages and narrating the descriptions via audio. The system generates captions for images, translates them into a selected language, and produces speech output for accessibility.

---

## Key Features

1. **Image Captioning**: Automatically generate descriptive captions for uploaded images using a Vision Transformer (ViT) model.
2. **Translation**: Translate captions into multiple Indian languages using the IndicTrans2 model.
3. **Speech Synthesis**: Convert translated captions into audio using Google Text-to-Speech (gTTS), enabling accessibility for visually impaired users.

---

## Core Functionality: `caption_translate_speech`

This function integrates the image captioning, translation, and speech synthesis processes. Here's how it works:

### Function Definition:
```python
def caption_translate_speech(image_path, language='hin_Deva', speech_lang='hi'):
```

### Parameters:
- **`image_path`**: The file path or URL of the image to process.
- **`language`**: The target language for translation (default: Hindi in Devanagari script, `hin_Deva`).
- **`speech_lang`**: The language for speech synthesis (default: Hindi, `hi`).

### Steps Performed:
1. **Display the Image**:
   - The `load_image` function displays the uploaded image (local or URL).
   
2. **Generate Caption**:
   - Captions are created using a Vision Transformer (ViT) model.
   - Example: *"A man riding a bicycle on a busy street."*

3. **Translate Caption**:
   - The generated caption is translated into the selected Indian language using IndicTrans2.
   - Example: *"एक आदमी व्यस्त सड़क पर साइकिल चला रहा है।"*

4. **Convert to Speech**:
   - The translated caption is converted to audio using gTTS.
   - The audio output is returned for playback.

---

## Project Structure

```plaintext
Image-Captioning/
├── app.py                     # Flask application for API requests
├── requirements.txt           # Dependencies
├── static/                    
│   ├── uploads/               # Directory for uploaded images
│   └── audio/                 # Directory for generated audio files
└── templates/                 
    └── index.html             # Web interface for the application
```

---

## Installation Guide

### 1. Clone the Repository:
```bash
git clone https://github.com/shivamlth27/Image-captioning.git
cd Image-Captioning
```

### 2. Create a Virtual Environment (Optional but Recommended):
```bash
python3 -m venv aiml
source aiml/bin/activate  # On Windows: `venv\Scripts\activate`
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Start the Flask App:
```bash
python app.py
```
- The app will run locally at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

### Usage Instructions:
1. **Upload an Image**: Upload an image file or provide a URL.
2. **Generate Caption**: The app generates a caption for the uploaded image.
3. **Translate Caption**: Select a target language, and the caption is translated.
4. **Speech Output**: Download or play the audio of the translated caption.

---

## Models Used

- **Image Captioning**: 
  - VisionEncoderDecoderModel (ViT + GPT-2) from Hugging Face.
- **Translation**: 
  - IndicTrans for Indian language translations.
- **Speech Synthesis**: 
  - Google Text-to-Speech (gTTS) for audio output.

---

## Demo
Watch the demo video for a walkthrough of the functionality:  
*`Demo.Video.mp4`*

---

This project bridges the gap between technology and accessibility, making image descriptions available in multiple languages with speech synthesis support, specifically tailored for visually impaired users.
