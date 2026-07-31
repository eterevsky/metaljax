"""Validation of the compile options PJRT hands us at Compile time.

jax's `jit(..., compiler_options={...})` / `lowered.compile(compiler_options=
{...})` reach a backend as `env_option_overrides` inside the serialized
CompileOptionsProto, and XLA's backends *reject* what they cannot apply:
an unknown name is `No such compile option: '<name>'`, a value of the wrong
type is `While setting option <name>, '<value>' is not a valid <type> value.`
Silently ignoring them would be worse than useless -- a caller asking for
`xla_dump_to` would get no dump and no complaint -- and jax's own test suite
asserts the errors.

metaljax has no XLA flag surface at all: every option it accepts is accepted
*and ignored* (logged under METALJAX_DEBUG). What it must still do is tell
"a real XLA option this backend does nothing with" apart from "a typo".
Names are screened by XLA's debug-option naming (`xla_...`); types are
checked only for the options whose type we actually know, since inventing
a type for an unknown option risks rejecting something valid -- the safe
failure here is a missed diagnostic, not a refused compile.
METALJAX_COMPILE_OPTIONS=ignore turns the whole check off.
"""

from __future__ import annotations

import os
import struct

# --- minimal protobuf wire-format reader -------------------------------
#
# CompileOptionsProto.env_option_overrides is field 7, a
# map<string, OptionOverrideProto>: each entry is a submessage with the
# name in field 1 and an OptionOverrideProto in field 2, whose oneof is
# string=1, bool=2, int=3, double=4. Verified against
# xla_client.CompileOptions().SerializeAsString().

_ENV_OPTION_OVERRIDES_FIELD = 7


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf: bytes):
    """Yield (field_number, wire_type, payload) over a serialized message.

    payload is bytes for wire type 2, an int for varints and fixed-width
    fields (raw little-endian bits for the latter)."""
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _varint(buf, i)
        num, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _varint(buf, i)
            yield num, wire, val
        elif wire == 1:
            yield num, wire, int.from_bytes(buf[i:i + 8], "little")
            i += 8
        elif wire == 2:
            ln, i = _varint(buf, i)
            yield num, wire, buf[i:i + ln]
            i += ln
        elif wire == 5:
            yield num, wire, int.from_bytes(buf[i:i + 4], "little")
            i += 4
        else:  # groups (3/4): removed from proto3, never emitted here
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _decode_override(buf: bytes):
    value = None
    for num, wire, payload in _fields(buf):
        if num == 1 and wire == 2:
            value = payload.decode("utf-8", "replace")
        elif num == 2 and wire == 0:
            value = bool(payload)
        elif num == 3 and wire == 0:
            # int64 on the wire is zigzag-free: negatives arrive as the
            # two's-complement 64-bit pattern.
            value = payload - (1 << 64) if payload >= (1 << 63) else payload
        elif num == 4 and wire == 1:
            value = struct.unpack("<d", payload.to_bytes(8, "little"))[0]
    return value


def parse_env_option_overrides(blob: bytes) -> list[tuple[str, object]]:
    """The (name, value) overrides inside a serialized CompileOptionsProto."""
    out: list[tuple[str, object]] = []
    for num, wire, payload in _fields(blob):
        if num != _ENV_OPTION_OVERRIDES_FIELD or wire != 2:
            continue
        key = None
        value = None
        for k, kwire, kpayload in _fields(payload):
            if k == 1 and kwire == 2:
                key = kpayload.decode("utf-8", "replace")
            elif k == 2 and kwire == 2:
                value = _decode_override(kpayload)
        if key is not None:
            out.append((key, value))
    return out


# --- option table ------------------------------------------------------

# Types for the options we can check. jaxlib's DebugOptions binding exposes
# a few dozen fields with live values, so their types come straight from it;
# the rest of XLA's ~500 debug options are not reachable from Python, so the
# handful below are declared by hand (jax's own suite sets them).
_EXTRA_OPTION_TYPES: dict[str, type] = {
    "xla_embed_ir_in_executable": bool,
    "xla_gpu_auto_spmd_partitioning_memory_budget_ratio": float,
}

_option_types: dict[str, type] | None = None


def _types() -> dict[str, type]:
    global _option_types
    if _option_types is None:
        table: dict[str, type] = {}
        try:
            from jaxlib import xla_client as xc

            do = xc.CompileOptions().executable_build_options.debug_options
            for name in dir(do):
                if not name.startswith("xla_"):
                    continue
                v = getattr(do, name)
                if isinstance(v, (bool, int, float, str)):
                    table[name] = type(v)
        except Exception:
            pass
        table.update(_EXTRA_OPTION_TYPES)
        _option_types = table
    return _option_types


_TYPE_NAMES = {bool: "bool", int: "int", float: "float", str: "string"}

# absl::SimpleAtob's vocabulary -- what XLA accepts for a bool field given
# as a string.
_TRUE = {"1", "t", "true", "y", "yes"}
_FALSE = {"0", "f", "false", "n", "no"}


class CompileOptionError(ValueError):
    pass


def _coerce(name: str, want: type, value):
    """Raise unless `value` is usable for an option of type `want`."""
    if value is None:
        # proto3 omits default-valued fields, so an override set to False /
        # 0 / "" arrives with its oneof empty. Nothing to check.
        return
    if isinstance(value, str):
        # XLA parses string overrides into the field's type.
        if want is str:
            return
        text = value.strip()
        if want is bool:
            if text.lower() in _TRUE or text.lower() in _FALSE:
                return
        elif want is int:
            try:
                int(text, 0)
                return
            except ValueError:
                pass
        elif want is float:
            try:
                float(text)
                return
            except ValueError:
                pass
        raise CompileOptionError(
            f"While setting option {name}, '{value}' is not a valid "
            f"{_TYPE_NAMES[want]} value.")
    if want is str and not isinstance(value, str):
        raise CompileOptionError(
            f"While setting option {name}, '{value}' is not a valid "
            f"string value.")
    # Numeric/bool values widen the way XLA's reflection does
    # (bool -> int -> double); nothing to reject.


def validate(blob: bytes | None) -> list[tuple[str, object]]:
    """Check a serialized CompileOptionsProto's env_option_overrides.

    Returns the overrides (all of which metaljax ignores); raises
    CompileOptionError with XLA's wording for anything XLA would refuse.
    """
    if not blob or os.environ.get("METALJAX_COMPILE_OPTIONS", "") == "ignore":
        return []
    try:
        overrides = parse_env_option_overrides(blob)
    except Exception:
        # A proto we cannot read is not a reason to fail the compile: the
        # options are advisory for this backend.
        return []
    if not all(name.isidentifier() for name, _ in overrides):
        # Whatever we decoded, it is not a list of option names -- a future
        # CompileOptionsProto layout, most likely. Rejecting every compile
        # over a misparse would be far worse than skipping the check.
        return []
    types = _types()
    for name, value in overrides:
        # XLA looks the name up in the DebugOptions descriptor; ours is the
        # naming rule, since we cannot see that descriptor.
        if not (name.startswith("xla_") and len(name) > 4):
            raise CompileOptionError(f"No such compile option: '{name}'")
        want = types.get(name)
        if want is not None:
            _coerce(name, want, value)
    if overrides and os.environ.get("METALJAX_DEBUG", "") == "1":
        print(f"[metaljax] ignoring compile options: "
              f"{[n for n, _ in overrides]}", flush=True)
    return overrides
