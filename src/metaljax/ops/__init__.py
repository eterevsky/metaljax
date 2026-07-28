"""Op handler registration. Importing this package registers all handlers."""

from metaljax.ops import (  # noqa: F401
    callbacks,
    collectives,
    control,
    conv,
    elementwise,
    gather,
    lapack,
    linalg,
    reduction,
    rng,
    shape,
    sort,
)
