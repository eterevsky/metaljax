"""Op handler registration. Importing this package registers all handlers."""

from metaljax.ops import (  # noqa: F401
    control,
    elementwise,
    gather,
    linalg,
    reduction,
    shape,
    sort,
)
