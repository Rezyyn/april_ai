<<<<<<< HEAD
import torch

def test_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch is using: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    test_torch_device()
=======
import torch

def test_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch is using: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    test_torch_device()
>>>>>>> e3f865b49b5649cd25509beb419db0b6fe201ff1
