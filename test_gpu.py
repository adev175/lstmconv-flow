import torch


def debug_memory_usage(batch_size):
    print(f"\n🔍 Memory Analysis - Batch Size {batch_size}")

    before_memory = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"Before batch: {before_memory:.2f} GB")

    # Simulate your training step
    inputs = torch.randn(batch_size, 6, 256, 256).cuda()

    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    current_memory = torch.cuda.memory_allocated() / 1024 ** 3

    print(f"Peak memory: {peak_memory:.2f} GB")
    print(f"Current memory: {current_memory:.2f} GB")
    print(f"Memory per sample: {(current_memory - before_memory) / batch_size:.3f} GB")


# Test different batch sizes
for bs in [1, 4, 8, 16]:
    debug_memory_usage(bs)
    torch.cuda.empty_cache()