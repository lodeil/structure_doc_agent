import os

os.system('pip install pyyaml==5.1')
# workaround: install old version of pytorch since detectron2 hasn't released packages for pytorch 1.9 (issue: https://github.com/facebookresearch/detectron2/issues/3158)
os.system('pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')

# install detectron2 that matches pytorch 1.8
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
os.system('pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html')

## install PyTesseract
os.system('pip install -q pytesseract')


import gradio as gr
import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'D:\\softwares\\Tesseract-OCR\\tesseract.exe'


processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")

# load image example
dataset = load_dataset("nielsr/funsd", split="test")
image = Image.open(dataset[0]["image_path"]).convert("RGB")
image = Image.open("./doc.png")
image.save("document.png")
# define id2label, label2color
labels = dataset.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

def process_image(image):
    width, height = image.size

    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)
    
    return image


title = "🧱 Document structure inference"
description = f""" ℹ️ \n Retrieve document structure from image. \n Tag detected boxes as QUESTION - ANSWER - HEADER - OTHER. \n Upload your own document or use the example below. 
"""
article = "layoutlmv2-base-uncased model"
examples =[['document.png']]

css = ".output-image, .input-image {height: 40rem !important; width: 100% !important;}"
#css = "@media screen and (max-width: 600px) { .output_image, .input_image {height:20rem !important; width: 100% !important;} }"
# css = ".output_image, .input_image {height: 600px !important}"

css = ".image-preview {height: auto !important;}"

iface = gr.Interface(fn=process_image, 
                     inputs=gr.inputs.Image(type="pil"), 
                     outputs=gr.outputs.Image(type="pil", label="annotated image"),
                     title=title,
                     description=description,
                     article=article,
                     examples=examples,
                     css=css,
                     enable_queue=True)
iface.launch(debug=True)