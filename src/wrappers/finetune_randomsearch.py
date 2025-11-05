# -*- coding: utf-8 -*-
"""Randomized hyperparameter search with optional K-Fold CV."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import itertools
import random
import copy
import os
import json
from src.wrappers.finetune_cv import FinetuneCVWrapper, log


class FinetuneRandomSearchWrapper:
    """Randomized hyperparameter tuning with flexible CV strategy."""

    def __init__(
        self,
        cfg: Any,
        param_grid: Dict[str, List[Any]],
        n_samples: int = 10,
        use_cv: bool = False,
        search_subset_frac: float = 0.3,
    ):
        """
        Args:
            cfg: Configuration object
            param_grid: Dict mapping parameter names to lists of values to try
            n_samples: Number of random configurations to sample
            use_cv: If True, use K-Fold CV for each trial. If False, use single train/val split
            search_subset_frac: Fraction of data to use for hyperparameter search (0.0-1.0)
        """
        self.cfg = cfg
        self.param_grid = param_grid
        self.n_samples = n_samples
        self.use_cv = use_cv
        self.search_subset_frac = search_subset_frac
        
        # Save original settings to restore later
        self.original_subset_frac = getattr(cfg.train, "subset_frac", 1.0)
        self.original_k_folds = getattr(cfg.train, "k_folds", 5)
        
        # Setup output directory
        self.search_dir = os.path.join(
            getattr(cfg.runtime, "run_dir", "./runs/finetune"),
            "hyperparam_search"
        )
        os.makedirs(self.search_dir, exist_ok=True)

    def _sample_configs(self) -> List[Dict[str, Any]]:
        """Sample random hyperparameter configurations."""
        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*self.param_grid.values()))
        
        # Sample without replacement up to available combinations
        n_to_sample = min(self.n_samples, len(combos))
        selected = random.sample(combos, n_to_sample)
        
        return [dict(zip(keys, combo)) for combo in selected]

    def _apply_params_to_cfg(self, params: Dict[str, Any]) -> Any:
        """Apply hyperparameters to config. Returns deep copy with modifications."""
        cfg = copy.deepcopy(self.cfg)
        
        # Learning rate and optimizer params
        if "lr" in params:
            cfg.train.learning_rate = params["lr"]
        if "weight_decay" in params:
            cfg.train.weight_decay = params["weight_decay"]
        if "momentum" in params:
            if hasattr(cfg.train, "momentum"):
                cfg.train.momentum = params["momentum"]
        
        # Data params
        if "batch_size" in params:
            cfg.data.batch_size = params["batch_size"]
        if "label_smoothing" in params:
            cfg.loss.label_smoothing = params["label_smoothing"]
        
        # Model params (if applicable)
        if "dropout" in params:
            if hasattr(cfg.model, "dropout"):
                cfg.model.dropout = params["dropout"]
        
        # Augmentation params (if applicable)
        if "augmentation_strength" in params:
            if hasattr(cfg.data, "augmentation_strength"):
                cfg.data.augmentation_strength = params["augmentation_strength"]
        
        # Apply search-specific settings
        cfg.train.subset_frac = self.search_subset_frac
        
        if not self.use_cv:
            # For single split: use 1 "fold" which effectively does train/val split
            cfg.train.k_folds = 1
        
        return cfg

    def run_search(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run randomized hyperparameter search."""
        log.info(f"\n{'='*60}")
        log.info(f"Starting Hyperparameter Search")
        log.info(f"  Strategy: {'K-Fold CV' if self.use_cv else 'Single Train/Val Split'}")
        log.info(f"  Data subset: {self.search_subset_frac*100:.0f}%")
        log.info(f"  Configurations to try: {self.n_samples}")
        log.info(f"  K-folds: {self.original_k_folds if self.use_cv else 1}")
        log.info(f"{'='*60}\n")
        
        results = []
        sampled_configs = self._sample_configs()

        for i, params in enumerate(sampled_configs, start=1):
            log.info(f"\n{'─'*60}")
            log.info(f"Trial {i}/{len(sampled_configs)}: {params}")
            log.info(f"{'─'*60}")

            # Apply sampled params to cfg
            trial_cfg = self._apply_params_to_cfg(params)
            
            # Update run name for this trial
            trial_cfg.logging.run_name = f"{self.cfg.logging.run_name}_trial{i:02d}"

            try:
                cv_wrapper = FinetuneCVWrapper(trial_cfg)
                cv_results = cv_wrapper.train()  # Fixed: was train_cv()
                
                mean_metric = cv_results["mean_metric"]
                std_metric = cv_results.get("std_metric", 0.0)
                
                result_entry = {
                    "trial": i,
                    "params": params,
                    "mean_metric": mean_metric,
                    "std_metric": std_metric,
                    "fold_metrics": cv_results.get("fold_metrics", [mean_metric]),
                }
                results.append(result_entry)
                
                log.info(f"✓ Trial {i} completed: {mean_metric:.4f} ± {std_metric:.4f}")
                
            except Exception as e:
                log.error(f"✗ Trial {i} failed with error: {e}")
                result_entry = {
                    "trial": i,
                    "params": params,
                    "mean_metric": float('-inf'),
                    "std_metric": 0.0,
                    "error": str(e),
                }
                results.append(result_entry)

        # Find best configuration
        reverse = not str(self.cfg.train.metric_key).endswith("loss")
        valid_results = [r for r in results if r["mean_metric"] != float('-inf')]
        
        if not valid_results:
            raise RuntimeError("All hyperparameter trials failed!")
        
        best_result = sorted(valid_results, key=lambda x: x["mean_metric"], reverse=reverse)[0]
        best_params = best_result["params"]
        best_metric = best_result["mean_metric"]
        
        log.info(f"\n{'='*60}")
        log.info(f"🏆 Best Configuration Found:")
        log.info(f"  Params: {best_params}")
        log.info(f"  Metric: {best_metric:.4f}")
        log.info(f"  Trial: {best_result['trial']}")
        log.info(f"{'='*60}\n")
        
        # Save results to JSON
        self._save_results(results, best_result)
        
        return best_params, results

    def _save_results(self, results: List[Dict[str, Any]], best_result: Dict[str, Any]):
        """Save search results to JSON file."""
        output = {
            "search_config": {
                "param_grid": self.param_grid,
                "n_samples": self.n_samples,
                "use_cv": self.use_cv,
                "search_subset_frac": self.search_subset_frac,
                "k_folds": self.original_k_folds if self.use_cv else 1,
            },
            "best_result": best_result,
            "all_results": results,
        }
        
        output_path = os.path.join(self.search_dir, "search_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        log.info(f"Search results saved to {output_path}")


def run(
    cfg: Any,
    param_grid: Dict[str, List[Any]],
    n_samples: int = 10,
    use_cv: bool = False,
    search_subset_frac: float = 0.3,
) -> Dict[str, Any]:
    """
    Entry point for hyperparameter search.
    
    Args:
        cfg: Configuration object
        param_grid: Dict mapping parameter names to lists of values to try
        n_samples: Number of random configurations to sample
        use_cv: If True, use K-Fold CV. If False, use single train/val split (faster)
        search_subset_frac: Fraction of data to use for search (e.g., 0.3 = 30%)
    
    Returns:
        Dict with best_params and all results
    
    Example:
        >>> param_grid = {
        ...     "lr": [1e-4, 5e-4, 1e-3],
        ...     "weight_decay": [0.0, 0.01, 0.1],
        ...     "batch_size": [16, 32, 64],
        ... }
        >>> results = run(cfg, param_grid, n_samples=10, use_cv=False, search_subset_frac=0.3)
    """
    wrapper = FinetuneRandomSearchWrapper(
        cfg=cfg,
        param_grid=param_grid,
        n_samples=n_samples,
        use_cv=use_cv,
        search_subset_frac=search_subset_frac,
    )
    best_params, results = wrapper.run_search()
    
    return {
        "best_params": best_params,
        "results": results,
    }