import torch
from model import GPTModel
from text_generation import generate_text
from config import BASE_CONFIG

if __name__ == "__main__":
    gpt = GPTModel(BASE_CONFIG)
    gpt.load_state_dict(torch.load("trained_models/gpt_model.pth"))
    gpt.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)

    with torch.no_grad():
        print("Generated text:", generate_text(gpt, "You are all resolved", 10, device))
