import os
import sys
from pathlib import Path
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def main():
    from google.colab import drive
    drive.mount('/content/drive')
    
    run("pip install -q kagglehub")
    
    import kagglehub
    dataset_path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
    
    os.environ['DATASET_PATH'] = dataset_path
    os.environ['MODEL_SAVE_PATH'] = '/content/drive/MyDrive/deepfake_models'
    
    Path('/content/drive/MyDrive/deepfake_models').mkdir(parents=True, exist_ok=True)
    Path('/content/deepfake-forensics-jepa/results/checkpoints').mkdir(parents=True, exist_ok=True)
    
    print(dataset_path)

if __name__ == "__main__":
    main()