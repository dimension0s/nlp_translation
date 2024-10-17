import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'There are{torch.cuda.device_count()} GPU(s) available.')
    print(f"Device name:", torch.cuda.get_device_name(0))

else:
    device = torch.device("cpu")
    print("No GPU available,using the CPU instead.")