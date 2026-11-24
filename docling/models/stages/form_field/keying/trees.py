# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A gradient-boosted tree ensemble for binary classification, evaluated with numpy.

The trees are stored as flat arrays in a ``.npz`` file, so no training
library is needed at run time. A sample goes left at a node when its
feature is at most the node's threshold, and a missing (NaN) feature goes to
the side the node learned for missing values. The probability is the
logistic function of the baseline plus the leaf values of all trees.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


@dataclass(frozen=True)
class TreeEnsemble:
    """Trees as arrays of shape (trees, nodes); unused nodes are leaves of value 0."""

    names: tuple[str, ...]  # Feature names, in column order.
    baseline: float
    feature: np.ndarray  # Feature column of each split node.
    threshold: np.ndarray
    left: np.ndarray  # Child node indices.
    right: np.ndarray
    missing_left: np.ndarray  # Where a NaN feature goes.
    leaf: np.ndarray
    value: np.ndarray  # Leaf values.
    no_key: float = 0.0  # Below this probability, a value keeps no key.
    levels: int = field(init=False, repr=False)  # Steps to reach every leaf.

    @property
    def depth(self) -> int:
        """Longest root-to-leaf path, in edges."""
        trees, nodes = self.leaf.shape
        depth = np.zeros((trees, nodes), dtype=np.int64)
        for t in range(trees):
            for n in range(nodes):
                if not self.leaf[t, n]:
                    depth[t, self.left[t, n]] = depth[t, n] + 1
                    depth[t, self.right[t, n]] = depth[t, n] + 1
        return int(depth.max())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Probability of the positive class for each row of ``x``."""
        if x.ndim != 2 or x.shape[1] != len(self.names):
            raise ValueError(
                f"expected rows of {len(self.names)} features, got shape {x.shape}"
            )
        if not len(x):
            return np.zeros(0)
        # The trees were fitted on float32 features.
        x = x.astype(np.float32).astype(np.float64)
        trees = self.leaf.shape[0]
        rows = np.arange(len(x))[:, None]
        column = np.arange(trees)[None, :]
        node = np.zeros((len(x), trees), dtype=np.int64)
        for _ in range(self.levels):
            at = self.feature[column, node]
            sample = x[rows, at]
            go_left = np.where(
                np.isnan(sample),
                self.missing_left[column, node],
                sample <= self.threshold[column, node],
            )
            child = np.where(go_left, self.left[column, node], self.right[column, node])
            node = np.where(self.leaf[column, node], node, child)
        raw = self.baseline + self.value[column, node].sum(axis=1)
        return 1.0 / (1.0 + np.exp(-raw))

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", self.depth)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            format_version=np.array(FORMAT_VERSION),
            names=np.array(json.dumps(list(self.names))),
            baseline=np.array(self.baseline),
            no_key=np.array(self.no_key),
            feature=self.feature,
            threshold=self.threshold,
            left=self.left,
            right=self.right,
            missing_left=self.missing_left,
            leaf=self.leaf,
            value=self.value,
        )

    @classmethod
    def from_sklearn(
        cls, model, names: Sequence[str], no_key: float = 0.0
    ) -> TreeEnsemble:
        """The trees of a fitted binary ``HistGradientBoostingClassifier``.

        Reads the fitted predictors directly, so scikit-learn is only needed
        to fit and export, never to predict. Categorical splits are not
        supported.
        """
        if any(len(it) != 1 for it in model._predictors):
            raise ValueError("not a binary classifier")
        predictors = [it[0] for it in model._predictors]
        if any(p.nodes["is_categorical"].any() for p in predictors):
            raise ValueError("categorical splits are not supported")
        if len(names) != model.n_features_in_:
            raise ValueError(f"{len(names)} names for {model.n_features_in_} features")
        shape = (len(predictors), max(len(p.nodes) for p in predictors))
        arrays = {
            "feature": np.zeros(shape, np.int64),
            "threshold": np.zeros(shape, np.float64),
            "left": np.zeros(shape, np.int64),
            "right": np.zeros(shape, np.int64),
            "missing_left": np.zeros(shape, bool),
            "leaf": np.ones(shape, bool),
            "value": np.zeros(shape, np.float64),
        }
        fields = {
            "feature": "feature_idx",
            "threshold": "num_threshold",
            "left": "left",
            "right": "right",
            "missing_left": "missing_go_to_left",
            "leaf": "is_leaf",
            "value": "value",
        }
        for t, predictor in enumerate(predictors):
            nodes = predictor.nodes
            for name, source in fields.items():
                arrays[name][t, : len(nodes)] = nodes[source]
        baseline = float(np.asarray(model._baseline_prediction).ravel()[0])
        return cls(tuple(names), baseline, no_key=no_key, **arrays)

    @classmethod
    def load(cls, path: Path) -> TreeEnsemble:
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{path}: tree format {version}, expected {FORMAT_VERSION}"
                )
            return cls(
                names=tuple(json.loads(str(data["names"]))),
                baseline=float(data["baseline"]),
                feature=data["feature"].astype(np.int64),
                threshold=data["threshold"].astype(np.float64),
                left=data["left"].astype(np.int64),
                right=data["right"].astype(np.int64),
                missing_left=data["missing_left"].astype(bool),
                leaf=data["leaf"].astype(bool),
                value=data["value"].astype(np.float64),
                no_key=float(data["no_key"]),
            )
