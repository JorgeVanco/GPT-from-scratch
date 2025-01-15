import os
import requests


def download_shakespeare(input_file_path: str = None) -> None:
    # download the tiny shakespeare dataset
    if input_file_path is None:
        input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    if not os.path.exists(input_file_path):
        if not os.path.exists(os.path.dirname(input_file_path)):
            os.makedirs(os.path.dirname(input_file_path))
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)
