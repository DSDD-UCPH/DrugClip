try:
    import unimol.tasks
    import unimol.data
    import unimol.models
    import unimol.losses
    import unimol.utils
except ImportError as e:
    # Unit tests and tools that only need turboquant / packed records should
    # still be importable when torch/unicore are not on PYTHONPATH.
    msg = str(e).lower()
    if "torch" in msg or "unicore" in msg:
        pass
    else:
        raise
