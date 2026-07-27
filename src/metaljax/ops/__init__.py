"""Op handler registration. Importing this package registers all handlers."""

from metaljax.ops import (  # noqa: F401
    control,
    conv,
    elementwise,
    gather,
    lapack,
    linalg,
    reduction,
    shape,
    sort,
)
