import urllib.request


def download_the_verdict(file_path: str = "data/the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )

    urllib.request.urlretrieve(url, file_path)
