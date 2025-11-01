# -*- coding: utf-8 -*-
"""Randomized hyperparameter search with K-Fold CV."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import itertools
import random
from src.wrappers.finetune_cv import FinetuneCVWrapper, log


class FinetuneRandomSearchWrapper:
    """Randomized hyperparameter tuning using internal FinetuneCVWrapper."""

    def __init__(self, cfg: Any, param_grid: Dict[str, List[Any]], n_samples: int = 10):
        self.cfg = cfg
        self.param_grid = param_grid
        self.n_samples = n_samples

    def _sample_configs(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*self.param_grid.values()))
        selected = random.sample(combos, min(self.n_samples, len(combos)))
        return [dict(zip(keys, combo)) for combo in selected]

    def run_search(self) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], float]]]:
        """Run randomized grid search with CV on subset of data."""
        results = []
        sampled_configs = self._sample_configs()

        for i, params in enumerate(sampled_configs, start=1):
            log.info(f"\nRunning config {i}/{len(sampled_configs)}: {params}")

            # Apply sampled params to cfg
            cfg = self._apply_params_to_cfg(params)

            cv_wrapper = FinetuneCVWrapper(cfg)
            cv_results = cv_wrapper.train_cv()
            mean_metric = cv_results["mean_metric"]

            results.append((params, mean_metric))

        reverse = not str(self.cfg.train.metric_key).endswith("loss")
        best_params, best_metric = sorted(results, key=lambda x: x[1], reverse=reverse)[0]
        log.info(f"🏆 Best params: {best_params} -> {best_metric:.4f}")

        return best_params, results

    def _apply_params_to_cfg(self, params: Dict[str, Any]) -> Any:
        """Returns a deep-copied cfg with modified hyperparams."""
        import copy
        cfg = copy.deepcopy(self.cfg)
        if "lr" in params:
            cfg.train.learning_rate = params["lr"]
        if "weight_decay" in params:
            cfg.train.weight_decay = params["weight_decay"]
        if "batch_size" in params:
            cfg.data.batch_size = params["batch_size"]
        return cfg


def run(cfg: Any, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Entry point for random search, mirrors style of `run(cfg)` in FinetuneWrapper."""
    wrapper = FinetuneRandomSearchWrapper(cfg, param_grid)
    best_params, results = wrapper.run_search()
    return {"best_params": best_params, "results": results}
