import torch
import torchvision
import base64
import json
import numpy as np
import os

from PIL import Image
from io import BytesIO
from dataset_s import get_image_transformer

def cpu():
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')
def num_gpus():
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()
# Preprocessing steps for the image
image_transforms = get_image_transformer(64,False,None)
classes = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
model_file = '/opt/ml/model.pt'
if not os.path.exists(model_file):
    Exception('Model file not found')
model = torch.jit.load(model_file, map_location=try_gpu())

# Put model in evaluation mode for inferencing
model.eval()

def predict(torch_image):
    if len(torch_image.shape) == 3:
        torch_image = torch_image.unsqueeze(0)
    else:
        raise Exception('Wrong shape, must be 3,x,x')
    return classes[model(torch_image).argmax().item()]

def lambda_handler(event, context):
    print(event)
    image_bytes = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='RGB')
    image  = image_transforms(image)

    label = predict(image)
    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": label,
            }
        )
    }

if __name__ == '__main__':
    print(predict())
