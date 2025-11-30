"""Dataset builders for probing memory and concept separability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

InputSequence = np.ndarray  # shape (T,)
OutputSequence = np.ndarray  # shape (T, d) or (T,)
SimulatorFn = Callable[[InputSequence], OutputSequence]


@dataclass
class MemoryDataset:
    inputs: np.ndarray
    outputs: np.ndarray
    dt: float = 1.0


@dataclass
class ConceptDataset:
    inputs: np.ndarray
    outputs: np.ndarray
    labels: np.ndarray
    dt: float = 1.0


def generate_random_drive_dataset(
    simulator: SimulatorFn,
    n_trials: int,
    n_steps: int,
    burn_in: int = 0,
    input_alphabet: Sequence[float] = (-1.0, 0.0, 1.0),
    input_probs: Sequence[float] | None = None,
    rng: np.random.Generator | None = None,
    dt: float = 1.0,
) -> MemoryDataset:
    """Generate random input driven dataset for memory profiling."""

    if rng is None:
        rng = np.random.default_rng()

    input_alphabet = np.asarray(input_alphabet, dtype=float)
    if input_probs is None:
        input_probs = np.ones(len(input_alphabet)) / len(input_alphabet)
    input_probs = np.asarray(input_probs, dtype=float)
    input_probs = input_probs / input_probs.sum()

    T_total = n_steps + burn_in
    all_inputs = np.empty((n_trials, n_steps), dtype=float)
    all_outputs: list[np.ndarray] = []

    for trial in range(n_trials):
        idx = rng.choice(len(input_alphabet), size=T_total, p=input_probs)
        u_full = input_alphabet[idx]
        y_full = simulator(u_full)
        if y_full.shape[0] != T_total:
            raise ValueError(
                f"Simulator returned length {y_full.shape[0]}, expected {T_total}"
            )
        u = u_full[burn_in:]
        y = y_full[burn_in:]
        all_inputs[trial] = u
        all_outputs.append(y if y.ndim == 2 else y[:, None])

    outputs_arr = np.stack(all_outputs, axis=0)
    return MemoryDataset(inputs=all_inputs, outputs=outputs_arr, dt=dt)


def generate_concept_dataset(
    simulator: SimulatorFn,
    patterns: Sequence[Sequence[float]],
    n_trials_per_concept: int,
    burn_in: int = 0,
    rng: np.random.Generator | None = None,
    dt: float = 1.0,
) -> ConceptDataset:
    """Generate concept dataset for discriminability tests."""

    if rng is None:
        rng = np.random.default_rng()

    patterns_arr = [np.asarray(p, dtype=float) for p in patterns]
    lengths = {len(p) for p in patterns_arr}
    if len(lengths) != 1:
        raise ValueError("All patterns must have the same length for now.")
    L = lengths.pop()

    n_concepts = len(patterns_arr)
    n_trials = n_concepts * n_trials_per_concept
    all_inputs = np.empty((n_trials, L), dtype=float)
    all_outputs: list[np.ndarray] = []
    labels = np.empty((n_trials,), dtype=int)

    trial_idx = 0
    for c, pattern in enumerate(patterns_arr):
        for _ in range(n_trials_per_concept):
            if burn_in > 0:
                pre = rng.normal(loc=0.0, scale=0.1, size=burn_in)
                u_full = np.concatenate([pre, pattern])
            else:
                u_full = pattern.copy()

            y_full = simulator(u_full)
            if y_full.shape[0] != u_full.shape[0]:
                raise ValueError("Simulator length mismatch in concept dataset.")

            u = u_full[-L:]
            y = y_full[-L:]
            all_inputs[trial_idx] = u
            all_outputs.append(y if y.ndim == 2 else y[:, None])
            labels[trial_idx] = c
            trial_idx += 1

    outputs_arr = np.stack(all_outputs, axis=0)
    return ConceptDataset(inputs=all_inputs, outputs=outputs_arr, labels=labels, dt=dt)
