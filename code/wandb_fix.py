import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the current memory usage
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
    cached_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # in GB
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
    free_memory = total_memory - allocated_memory  # in GB

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Used Memory: {allocated_memory:.2f} GB")
    print(f"Free Memory: {free_memory:.2f} GB")
    print(f"Cached Memory: {cached_memory:.2f} GB")

else:
    print("CUDA is not available.")
