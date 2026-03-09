"""Facade module for benchmark APIs.

Public benchmark functions are re-exported for backwards compatibility.
"""

from benchmarks.compression_benchmark import model_level_compression_experiment
from benchmarks.operator_profile import operator_level_profile
from benchmarks.pruning_benchmark import pruning_hardware_experiment
from benchmarks.repeated_benchmark import hardware_simulation, repeated_benchmark

__all__ = [
    "repeated_benchmark",
    "hardware_simulation",
    "pruning_hardware_experiment",
    "model_level_compression_experiment",
    "operator_level_profile",
]
