"""FreeToken inference runtime."""

import os

from freetoken.version import __version__

if os.name == "nt":
    # ROCm Windows fork: before anything imports transformers (see _fast_metadata).
    from freetoken import _fast_metadata

    _fast_metadata.install()

__all__ = ["__version__"]
