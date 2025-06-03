import urllib.request
import os


def download_the_verdict(file_path: str = "data/the-verdict.txt") -> None:
    """Download the verdict text file from the specified URL if it does not exist."""
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    if not os.path.exists(file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        urllib.request.urlretrieve(url, file_path)
