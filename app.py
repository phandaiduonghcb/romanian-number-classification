import base64
import os
import random
import string
from io import BytesIO


from flask import Flask, render_template
from flask import request
import torch
from PIL import Image


from ml.dataset_s import get_image_transformer # Preprocessing the input image
from ml.utils import try_gpu # Try to use GPU if available

def predict(torch_image):
    # Function to get the predicted label of a pytorch image
    if len(torch_image.shape) == 3:
        torch_image = torch_image.unsqueeze(0)
    else:
        raise Exception('Wrong shape, must be 3,x,x')
    return classes[model(torch_image).argmax().item()]

# Initalize flask app
app = Flask(__name__, template_folder='templates')

image_transforms = get_image_transformer(64,False,None) # pytorch transforms for preprocessing the input image
classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x'] # Class names
model_file = './ml/model.pt' # Trained model file path
model = torch.jit.load(model_file, map_location=try_gpu()) # Load model
model.eval() # Put model in evaluation mode for inferencing


@app.route("/", methods=["GET", "POST"]) # Our web endpoint
def index():
    if request.method == 'GET':
        return render_template('index.html',predicted_label=None)
    if request.method == 'POST':

        # Get the uploaded file
        file = request.files['file']

        # If the file is empty, then there is no file uploaded, so we just refresh the page
        if not file.filename:
            return render_template('index.html',predicted_label=None)
        
        # If the file exists, save it to /tmp folder with a newly generated name
        chars = string.ascii_letters + string.digits
        prefix = ''.join(random.choice(chars) for i in range(16))
        new_filename = f'{prefix}_{file.filename}'
        file.save(f'/tmp/{new_filename}')

        # Load the image using PIL Image and preprocess it to get a pytorch image
        pil_image = Image.open(f'/tmp/{new_filename}').convert(mode='RGB')
        torch_image  = image_transforms(pil_image)
        torch_image = torch_image.to(try_gpu())

        # Get label of the torch image
        label = predict(torch_image)

        # Convert the uploaded image to base64 string to show it when inferencing sucessfully
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return render_template('index.html',predicted_label=label,img = img_str)

if __name__ == '__main__':
    app.run(debug=True)
