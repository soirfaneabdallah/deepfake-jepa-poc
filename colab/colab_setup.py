import os
import subprocess
from pathlib import Path

def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')

def download_dataset():
    subprocess.run("pip install -q kagglehub", shell=True, check=True)
    import kagglehub
    return kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")

def setup():
    mount_drive()
    dataset_path = download_dataset()
    os.environ['DATASET_PATH'] = dataset_path
    os.environ['MODEL_SAVE_PATH'] = '/content/drive/MyDrive/deepfake_models'
    Path('/content/drive/MyDrive/deepfake_models').mkdir(parents=True, exist_ok=True)
    Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
    return dataset_path

if __name__ == "__main__":
    setup()