import torch
import intel_extension_for_pytorch as ipex # The critical "bridge"

print("Intel GPU Status Report")

# IPEX adds the 'xpu' capabilities to the torch namespace
gpu_ready = torch.xpu.is_available() 

print(f"XPU Backend Loaded: {ipex.__version__}")
print(f"GPU available? {gpu_ready}")

if gpu_ready:
    print(f"Device Name: {torch.xpu.get_device_name(0)}")
    print("Success! Your Intel Arc GPU is online.")
else:
    print("GPU still not detected. Ensure oneAPI is initialized.")