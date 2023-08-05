# Romanian number classification
## Installation
```sh
# Clone the git repo
git clone https://github.com/phandaiduonghcb/romanian-number-classification
cd romanian-number-classification

# Create a python environment and install dependencies
sudo apt update
sudo apt install ffmpeg libsm6 libxext6 python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --no-cache-dir

# Run flask on localhost using port 8080
flask run --host 0.0.0.0 --port 8080
```
