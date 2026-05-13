import torch


def get_runtime_device(config_mode="auto"):
    """
    Resolve runtime device from configuration only.
    """

    selected_mode = str(config_mode).strip().lower()
    if selected_mode not in {"auto", "cpu", "gpu"}:
        raise ValueError("Invalid device mode. Use one of: auto, cpu, gpu")

    cuda_available = torch.cuda.is_available()

    if selected_mode == "gpu" and not cuda_available:
        raise RuntimeError("DDNET device mode is set to 'gpu' but CUDA is unavailable.")

    use_cuda = selected_mode == "gpu" or (selected_mode == "auto" and cuda_available)
    device = torch.device("cuda" if use_cuda else "cpu")

    return device, use_cuda, selected_mode
