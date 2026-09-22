"""Run from any directory: python backend/download_waste_model.py --size n|s."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=['n', 's'], default='s')
    args = parser.parse_args()
    repo = f'ditoow/trashscan8{args.size}'
    with urlopen(f'https://huggingface.co/api/models/{repo}') as response:
        revision = json.load(response)['sha']
    target = Path(__file__).parent / 'weights' / f'trashscan8{args.size}.onnx'
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.part')
    digest = hashlib.sha256()
    with urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/best.onnx', timeout=180) as response, temporary.open('wb') as output:
        while chunk := response.read(1024*1024):
            digest.update(chunk)
            output.write(chunk)
    temporary.replace(target)
    print(f'Model: {target}\nSource: {repo}@{revision}\nSHA256: {digest.hexdigest()}')

if __name__ == '__main__':
    main()
