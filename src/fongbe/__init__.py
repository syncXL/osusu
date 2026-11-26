import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")