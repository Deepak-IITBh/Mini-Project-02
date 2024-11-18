import os
import torch
from flask import Flask, request, render_template, send_file
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from PIL import Image
import requests
import io
import gTTS

app = Flask(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image Captioning Model Setup
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(DEVICE)
caption_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Translation Model Setup
translation_model_dir = "AI4Bharat/indic-trans-v2-en-indic"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_dir, trust_remote_code=True)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    translation_model_dir,
    trust_remote_code=True,
).to(DEVICE)
ip = IndicProcessor()

def check_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def load_image(image_path):
    if check_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
    else:
        raise ValueError("Invalid image path")

def get_caption(model, image_processor, tokenizer, image):
    # Preprocessing the Image
    img = image_processor(image, return_tensors="pt").to(DEVICE)

    # Generating captions
    output = model.generate(**img)

    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return caption

def translate_text(text, src_lang, tgt_lang):
    # Preprocess the text
    text = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)[0]

    # Tokenize and generate translations
    inputs = translation_tokenizer(
        [text],
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = translation_model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens
    with translation_tokenizer.as_target_tokenizer():
        translations = translation_tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess the translations
    translations = ip.postprocess_batch(translations, lang=tgt_lang)

    return translations[0]

def text_to_speech(text, language='en'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=language)
    
    # Save to a bytes buffer
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    return mp3_fp

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part', 400
        
        file = request.files['file']
        
        # If no file is selected, file.filename will be an empty string
        if file.filename == '':
            return 'No selected file', 400
        
        # Read the image
        image = Image.open(file.stream)
        
        # Get caption
        caption = get_caption(caption_model, image_processor, caption_tokenizer, image)
        
        # Language selection
        target_language = request.form.get('language', 'hi')  # Default to Hindi
        
        # Translate caption
        translated_caption = translate_text(caption, 'en', target_language)
        
        # Generate speech
        speech_file = text_to_speech(translated_caption, target_language)
        
        return render_template('result.html', 
                               original_caption=caption, 
                               translated_caption=translated_caption,
                               language=target_language)
    
    return render_template('index.html')

@app.route('/speech')
def get_speech():
    # Generate speech for the translated caption
    translated_caption = request.args.get('text', '')
    language = request.args.get('language', 'hi')
    
    speech_file = text_to_speech(translated_caption, language)
    
    return send_file(
        speech_file, 
        mimetype='audio/mp3',
        as_attachment=True,
        download_name='caption.mp3'
    )

if __name__ == '__main__':
    app.run(debug=True)