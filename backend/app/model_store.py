from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import ARTIFACTS_DIR, DISADVANTAGEOUS_DECKS, SITUATION_FEATURES
from .features import prepare_trials


class ModelStore:
    def __init__(self, artifacts_dir: Path | None = None):
        self.artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
        self._ensure_trained()
        self.models = {
            "svm": {
                "deck": joblib.load(self.artifacts_dir / "svm_deck.joblib"),
                "switch": joblib.load(self.artifacts_dir / "svm_switch.joblib"),
            },
            "pnn": {
                "deck": joblib.load(self.artifacts_dir / "pnn_deck.joblib"),
                "switch": joblib.load(self.artifacts_dir / "pnn_switch.joblib"),
            },
            "kl_pnn": {
                "deck": joblib.load(self.artifacts_dir / "kl_pnn_deck.joblib"),
                "switch": joblib.load(self.artifacts_dir / "kl_pnn_switch.joblib"),
            },
        }
        self.gmm = joblib.load(self.artifacts_dir / "gmm.joblib")
        self.person_scaler = joblib.load(self.artifacts_dir / "person_scaler.joblib")
        with open(self.artifacts_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.trials = prepare_trials()

    def _ensure_trained(self) -> None:
        required = [
            "svm_deck.joblib",
            "svm_switch.joblib",
            "pnn_deck.joblib",
            "pnn_switch.joblib",
            "kl_pnn_deck.joblib",
            "kl_pnn_switch.joblib",
            "gmm.joblib",
            "person_scaler.joblib",
            "meta.json",
        ]
        if not all((self.artifacts_dir / name).exists() for name in required):
            from .train import train_and_save

            train_and_save(self.artifacts_dir)

        # Ensure best-model metadata exists even for older artifact folders
        meta_path = self.artifacts_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        if "best_models" not in meta and "metrics" in meta:
            from .train import select_best_model

            best_deck = select_best_model(meta["metrics"]["deck"])
            best_switch = select_best_model(meta["metrics"]["switch"])
            meta["default_deck_model"] = best_deck
            meta["default_switch_model"] = best_switch
            meta["best_models"] = {
                "deck": best_deck,
                "switch": best_switch,
                "selection_rule": "max_accuracy then max_f1 then min_mean_kl",
                "deck_metrics": meta["metrics"]["deck"][best_deck],
                "switch_metrics": meta["metrics"]["switch"][best_switch],
            }
            meta["available_models"] = ["best", "svm", "pnn", "kl_pnn"]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    def resolve_name(self, name: str, target: str) -> str:
        if name in (None, "", "best", "auto"):
            if target == "deck":
                return self.meta.get(
                    "best_models", {}
                ).get("deck", self.meta.get("default_deck_model", "svm"))
            return self.meta.get("best_models", {}).get(
                "switch", self.meta.get("default_switch_model", "svm")
            )
        if name in self.models:
            return name
        return self.meta.get(f"default_{target}_model", "svm")

    def deck_model(self, name: str):
        return self.models[self.resolve_name(name, "deck")]["deck"]

    def switch_model(self, name: str):
        return self.models[self.resolve_name(name, "switch")]["switch"]

    def situation_frame(self, payload: dict) -> pd.DataFrame:
        return pd.DataFrame([{k: float(payload[k]) for k in SITUATION_FEATURES}])

    @staticmethod
    def is_suboptimal_deck(deck: int) -> bool:
        return int(deck) in DISADVANTAGEOUS_DECKS


def clear_store_cache() -> None:
    get_store.cache_clear()


@lru_cache(maxsize=1)
def get_store() -> ModelStore:
    return ModelStore()
