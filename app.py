import base64
import os
import random
import string
from io import BytesIO


from flask import Flask, render_template
from flask import request
import torch
from PIL import Image


from ml.dataset_s import get_image_transformer
from ml.utils import try_gpu

def predict(torch_image):
    if len(torch_image.shape) == 3:
        torch_image = torch_image.unsqueeze(0)
    else:
        raise Exception('Wrong shape, must be 3,x,x')
    return classes[model(torch_image).argmax().item()]

app = Flask(__name__, template_folder='templates')

image_transforms = get_image_transformer(64,False,None)
classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
model_file = './ml/model.pt'
if not os.path.exists(model_file):
    Exception('Model file not found')
model = torch.jit.load(model_file, map_location=try_gpu())
# Put model in evaluation mode for inferencing
model.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'GET':
        return render_template('index.html',predicted_label=None)
    if request.method == 'POST':
        file = request.files['file']
        if not file.filename:
            return render_template('index.html',predicted_label=None)
        chars = string.ascii_letters + string.digits
        prefix = ''.join(random.choice(chars) for i in range(16))
        new_filename = f'{prefix}_{file.filename}'
        file.save(f'/tmp/{new_filename}')
        pil_image = Image.open(f'/tmp/{new_filename}').convert(mode='RGB')
        torch_image  = image_transforms(pil_image)
        torch_image = torch_image.to(try_gpu())
        label = predict(torch_image)

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return render_template('index.html',predicted_label=label,img = img_str)

if __name__ == '__main__':
    app.run(debug=True)
