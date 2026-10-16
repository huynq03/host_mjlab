"""HoST on MJLab.

Initialize MJLab before importing task submodules so entry-point discovery is
independent of whether callers import MJLab or this package first.
"""

import mjlab as _mjlab  # noqa: F401
