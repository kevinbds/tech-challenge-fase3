"""Reproducible experiments with a frozen temporal test."""


def run_experiments(*args, **kwargs):
    from .experiments import run_experiments as run
    return run(*args, **kwargs)


__all__ = ["run_experiments"]
