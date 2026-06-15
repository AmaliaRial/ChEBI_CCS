import torch

print(torch.cuda.is_available())  # Debe ser True
print(torch.cuda.get_device_name(0))  # Debe mostrar tu GPU