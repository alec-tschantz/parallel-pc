from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax


@dataclass(frozen=True)
class Variable:
    """A mutable value tensor in the PC graph."""

    name: str
    shape: tuple[int, ...]


class Transform(eqx.Module):
    """A frozen transform mapping source variable(s) to target variable(s)."""

    id: str = eqx.field(static=True)
    module: eqx.Module
    src: tuple[str, ...] = eqx.field(static=True)
    tgt: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        id: str,
        module: eqx.Module,
        src: str | list[str] | tuple[str, ...],
        tgt: str | list[str] | tuple[str, ...],
    ):
        self.id = id
        self.module = module
        self.src = (src,) if isinstance(src, str) else tuple(src)
        self.tgt = (tgt,) if isinstance(tgt, str) else tuple(tgt)


@dataclass(frozen=True)
class Energy:
    """Wires a model-agnostic energy function to variable/transform IDs.

    args: each is a variable name, a transform ID (single-output),
          or "transform_id.target_name" (multi-output).
    """

    fn: Callable
    args: tuple[str, ...]

    def __init__(self, fn: Callable, args: list[str] | tuple[str, ...]):
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "args", tuple(args))


class State(eqx.Module):
    """Inference state: flat buffer + free mask."""

    flat: jax.Array
    free_mask: jax.Array
