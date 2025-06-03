from torch.utils.data import DataLoader, Dataset
import torch
import tiktoken


def train_test_split(data: str, train_ratio: float) -> tuple[str, str]:
    """Split the text data into training and testing sets based on the given ratio."""
    n = int(len(data) * train_ratio)
    training_text = data[:n]
    testing_text = data[n:]
    return training_text, testing_text


class TextDataset(Dataset):
    """A PyTorch Dataset for text data, encoding it into input-target pairs."""

    def __init__(self, text, max_length, stride, encoder="gpt2") -> None:
        self.data_text = text
        self.max_length = max_length
        self.stride = stride
        self.encoder = tiktoken.get_encoding(encoder)

        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        self.preprocess_dataset()

    def preprocess_dataset(self) -> None:
        """Preprocess the dataset by encoding the text and creating input-target pairs."""
        encoded_data = self.encoder.encode(self.data_text)

        for i in range(0, len(encoded_data) - self.max_length, self.stride):
            self.input_ids.append(torch.tensor(encoded_data[i : i + self.max_length]))
            self.target_ids.append(
                torch.tensor(encoded_data[i + 1 : i + self.max_length + 1])
            )

    def __getitem__(self, index) -> tuple:
        return self.input_ids[index], self.target_ids[index]

    def __len__(self) -> int:
        return len(self.input_ids)


def create_dataloader(
    text,
    max_length,
    stride,
    batch_size,
    tokenizer="gpt2",
    shuffle=True,
    drop_last=True,
    num_workers=0,
) -> DataLoader:
    """Create a DataLoader for the text dataset."""
    dataset = TextDataset(text, max_length, stride, tokenizer)
    return DataLoader(
        dataset, batch_size, shuffle, num_workers=num_workers, drop_last=drop_last
    )
