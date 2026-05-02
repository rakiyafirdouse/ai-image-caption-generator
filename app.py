from flask import Flask, request, jsonify, render_template
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# ============================
# LOAD BLIP MODEL
# ============================
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ============================
# FLASK APP
# ============================
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    path = "temp.jpg"
    file.save(path)

    try:
        image = Image.open(path).convert('RGB')

        inputs = processor(image, return_tensors="pt").to(device)

        out = model.generate(**inputs, max_length=30)

        caption = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({'caption': caption})

    except Exception as e:
        return jsonify({'error': str(e)})

    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == '__main__':
    app.run(debug=True)