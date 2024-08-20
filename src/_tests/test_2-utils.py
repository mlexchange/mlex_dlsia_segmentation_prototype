import os

import torch

from utils import find_device


def test_find_device_cuda_available(monkeypatch):
    # Monkey patch torch.cuda.is_available to return True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = find_device()
    assert device.type == "cuda", "Device should be cuda when CUDA is available"


def test_find_device_cuda_not_available(monkeypatch):
    # Monkey patch torch.cuda.is_available to return False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = find_device()
    assert device.type == "cpu", "Device should be cpu when CUDA is not available"


def test_dir_creation(model_directory):
    assert os.path.exists(model_directory)
    assert os.path.isdir(model_directory)
