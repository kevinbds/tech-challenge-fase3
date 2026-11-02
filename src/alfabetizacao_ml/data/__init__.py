"""Official-source acquisition and private analytical marts."""

from .acquisition import acquire_sources


def prepare_data(cache_dir, manifest_path):
    """Build validated local Silver/Gold and modeling marts."""
    from .preparation import prepare_data as prepare

    return prepare(cache_dir, manifest_path)


__all__ = ["acquire_sources", "prepare_data"]
