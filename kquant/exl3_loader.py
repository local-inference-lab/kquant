"""Load kquant's QSRT encoder atop an unmodified ExLlamaV3 dependency."""

from __future__ import annotations

import gc
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
import torch


_BACKEND_MODULE = "exllamav3.modules.quant.exl3_lib.kquant_qsrt_encoder"


def load_qsrt_encoder(exllamav3_root: str | Path) -> types.ModuleType:
    """Load kquant's QSRT encoder using an unmodified ExLlamaV3 checkout.

    kquant owns the mixed-rate LDLQ implementation. ExLlamaV3 supplies its
    unmodified extension plus Hadamard and tensor utilities. Importing the
    package root would also require serving, tokenizer, and generation
    dependencies that are intentionally absent from kquant's environment.
    """

    root = Path(exllamav3_root).resolve() / "exllamav3"
    packages = (
        ("exllamav3", root),
        ("exllamav3.modules", root / "modules"),
        ("exllamav3.modules.quant", root / "modules" / "quant"),
        (
            "exllamav3.modules.quant.exl3_lib",
            root / "modules" / "quant" / "exl3_lib",
        ),
        ("exllamav3.util", root / "util"),
    )
    for name, path in packages:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = module.__path__
        module.__spec__ = spec
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)

    util = sys.modules["exllamav3.util"]

    def cuda_sync_active() -> None:
        for device_id in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{device_id}")
            if torch.cuda.memory_allocated(device) > 0:
                torch.cuda.synchronize(device)

    util.cuda_sync_active = cuda_sync_active

    memory = types.ModuleType("exllamav3.util.memory")

    def free_mem() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    memory.free_mem = free_mem
    memory.list_gpu_tensors = lambda *args, **kwargs: None
    sys.modules[memory.__name__] = memory
    util.memory = memory

    progress = types.ModuleType("exllamav3.util.progress")

    class ProgressBar:
        def __init__(self, text: str, count: int, transient: bool = True):
            self.text = text
            self.count = count
            self.transient = transient

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def update(self, value: int) -> None:
            return None

        def new_task(self, text: str, count: int) -> None:
            self.text = text
            self.count = count

    progress.ProgressBar = ProgressBar
    sys.modules[progress.__name__] = progress
    util.progress = progress

    module = sys.modules.get(_BACKEND_MODULE)
    if module is None:
        backend_path = Path(__file__).with_name("exl3_encoder_backend.py")
        spec = importlib.util.spec_from_file_location(_BACKEND_MODULE, backend_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load QSRT encoder backend from {backend_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_BACKEND_MODULE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_BACKEND_MODULE, None)
            raise
        setattr(sys.modules["exllamav3.modules.quant.exl3_lib"], "kquant_qsrt_encoder", module)
    return module
