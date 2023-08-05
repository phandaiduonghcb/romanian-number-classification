# Romanian number classification
## Installation
```sh
git clone https://github.com/phandaiduonghcb/romanian-number-classification
cd romanian-number-classification
sudo apt update
sudo apt install ffmpeg libsm6 libxext6 python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --no-cache-dir
flask run --host 0.0.0.0 --port 8080
```
