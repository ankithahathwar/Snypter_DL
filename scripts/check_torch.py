import torch

print(f"PyTorch Version: {torch.__version__}")

# In 2026, Intel GPU support is built-in as 'xpu'
if torch.xpu.is_available():
    print(" SUCCESS: Intel Arc GPU is Active!")
    print(f"Device Name: {torch.xpu.get_device_name(0)}")
else:
    print("GPU not detected. Running on CPU.")
    # Quick tip: Ensure your Intel Graphics drivers are updated!