import io
import json
import os

import torch
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
class_label = json.load(open(os.path.join('inference', 'label.json')))
label_to_url = json.load(open(os.path.join('inference', 'label2url.json')))
model = torch.load(os.path.join('inference', 'model', 'model.pth'), map_location=torch.device('cpu'))
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    outputs = torch.softmax(outputs.squeeze(), dim=0)
    prob, indices = torch.topk(outputs, k=5)
    prob = prob.squeeze().detach().numpy()
    indices = indices.squeeze().numpy()
    return [{'class_name': class_label[indices[i]],
             'probability': "{0:.2f}".format(prob[i]),
             'inat_url': label_to_url[class_label[indices[i]]]} for i in range(5)]


@app.route('/')
def index():
    return 'DeepMushroom API'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        pred = get_prediction(image_bytes=img_bytes)
        return jsonify(pred)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0')
