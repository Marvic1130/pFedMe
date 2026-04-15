"""Wrappers exposing pFedMe-family baselines to the pipeline ExtAdapter."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from src.core import AppConfig

PFED_ROOT = Path(__file__).resolve().parent
if str(PFED_ROOT) not in sys.path:
    sys.path.insert(0, str(PFED_ROOT))

from main import main as pfedme_entry


class _PFedBaseWrapper:
    def __init__(self, args: Optional[Any] = None, cfg: Optional[AppConfig] = None) -> None:
        if args is None:
            self.args = SimpleNamespace()
        elif isinstance(args, dict):
            self.args = SimpleNamespace(**args)
        else:
            self.args = args
        self.cfg = cfg

    def _resolve(self, name: str, default: Any) -> Any:
        if hasattr(self.args, name):
            return getattr(self.args, name)
        return default

    def _extract_common_kwargs(self) -> dict[str, Any]:
        cfg = self.cfg
        dataset_default = cfg.dataset.name if cfg else "femnist"
        model_default = "cnn"
        batch_default = cfg.client.batch_size if cfg else 32
        # Decouple default LR from cfg.server.inner_lr which is for HyperQLoRA
        lr_default = 5e-3
        glob_iters_default = cfg.num_rounds if cfg else 800
        # User requested local_epochs = 5
        local_epochs_default = 5
        
        # Full participation by default:
        # - If cfg.dataset.clients_per_round is set, use it.
        # - Else if cfg.dataset.num_clients is set, use it.
        # - Else use 0 as a sentinel meaning "all available clients" (resolved in external/pFedMe/main.py).
        numusers_default = 0
        if cfg:
            numusers_default = cfg.dataset.clients_per_round or cfg.dataset.num_clients or 0

        return {
            "dataset": self._resolve("dataset", dataset_default),
            "model": self._resolve("model", model_default),
            "batch_size": self._resolve("batch_size", batch_default),
            "learning_rate": self._resolve("learning_rate", lr_default),
            "beta": self._resolve("beta", 1.0),
            "lamda": self._resolve("lamda", 15),
            "num_glob_iters": self._resolve("num_global_iters", glob_iters_default),
            "local_epochs": self._resolve("local_epochs", local_epochs_default),
            "optimizer": self._resolve("optimizer", "SGD"),
            "numusers": self._resolve("numusers", numusers_default),
            "K": self._resolve("K", 5),
            "personal_learning_rate": self._resolve("personal_learning_rate", 0.09),
            "times": self._resolve("times", 1),
            "gpu": self._resolve("gpu", 0),
        }

    def _run_with_algorithm(self, algorithm: str) -> None:
        common = self._extract_common_kwargs()
        pfedme_entry(
            dataset=common["dataset"],
            algorithm=algorithm,
            model=common["model"],
            batch_size=common["batch_size"],
            learning_rate=common["learning_rate"],
            beta=common["beta"],
            lamda=common["lamda"],
            num_glob_iters=common["num_glob_iters"],
            local_epochs=common["local_epochs"],
            optimizer=common["optimizer"],
            numusers=common["numusers"],
            K=common["K"],
            personal_learning_rate=common["personal_learning_rate"],
            times=common["times"],
            gpu=common["gpu"],
            cfg=self.cfg,
        )


class PFedMeWrapper(_PFedBaseWrapper):
    def run(self) -> None:
        self._run_with_algorithm("pFedMe")


class PerFedAvgWrapper(_PFedBaseWrapper):
    def run(self) -> None:
        # Per-FedAvg corresponds to "PerAvg" flag in upstream repo
        self._run_with_algorithm("PerAvg")


class PFedMeFedAvgWrapper(_PFedBaseWrapper):
    def run(self) -> None:
        self._run_with_algorithm("FedAvg")
