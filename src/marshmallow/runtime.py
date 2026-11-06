# ruff: noqa: SLF001
"""Runtime (de)serialization policy stack.

This module holds a *runtime* stack of policy frames that is independent of
``class Meta`` options, the ``many``/``partial``/``unknown`` arguments of a
single ``load``/``dump`` call, the field-level ``validate`` lists and the
class registry. The stack is bound to the running collaboration scope
(a :class:`contextvars.ContextVar`): code reached from the scoped block --
including nested schema conversions invoked from field processing -- reads
the stack as it exists *at that moment*, never a snapshot taken when the
outer conversion started. Two concurrent scopes (threads or asyncio tasks)
therefore never share frames.

Each frame carries three independent gates:

``serdes_admit``
    Whether new ``load``/``loads``/``dump``/``dumps`` conversions may be
    admitted inside the frame.

    * :data:`HONOR_CFG` -- behave as without any frame (default).
    * :data:`DENY_NEW` -- reject every *new* conversion with
      :class:`marshmallow.exceptions.SerdesAdmissionError`. A conversion that
      already entered field processing before the frame was pushed is never
      cancelled or rewritten: only later admissions are gated.
    * :data:`ONE_MORE` -- admit exactly one more conversion; the next one
      after it is rejected.

``validate_gate``
    * :data:`HONOR_CFG` -- run field ``validate`` callables, ``@validates``
      and ``@validates_schema`` validators as configured.
    * :data:`SKIP_VALIDATORS` -- skip those validators. Required-missing
      detection, ``allow_none`` checks and type conversion still run.
    * :data:`FREEZE_ERRORS` -- once validation errors have been collected,
      their containers become immutable; attempts to delete, alter or clear
      them raise :class:`marshmallow.exceptions.FrozenErrorStoreError`.

``unknown_gate``
    * :data:`HONOR_CFG` -- use the ``unknown`` resolved from ``Meta`` and the
      call argument.
    * :data:`FORCE_EXCLUDE` -- unknown keys are excluded.
    * :data:`FORCE_RAISE` -- unknown keys raise.

Example::

    from marshmallow import Schema, fields
    from marshmallow.runtime import serdes_scope, DENY_NEW


    class MySchema(Schema):
        name = fields.Str()


    with serdes_scope(serdes_admit=DENY_NEW):
        MySchema().load({"name": "foo"})  # raises SerdesAdmissionError

Frames nest: pushing a frame shadows the outer one until it is popped, and an
exception raised inside a :class:`serdes_scope` block still pops the frame.
An empty stack is byte-for-byte equivalent to the documented API behavior.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import typing

from marshmallow.constants import EXCLUDE, RAISE
from marshmallow.exceptions import SerdesAdmissionError

if typing.TYPE_CHECKING:
    from marshmallow import types

#: ``serdes_admit`` / ``validate_gate`` / ``unknown_gate`` value meaning
#: "follow Meta options, call arguments and field configuration as usual".
HONOR_CFG: typing.Final = "honor_cfg"

#: ``serdes_admit``: reject every new load/loads/dump/dumps in this frame.
DENY_NEW: typing.Final = "deny_new"

#: ``serdes_admit``: admit exactly one more load/loads/dump/dumps, reject the
#: next one.
ONE_MORE: typing.Final = "one_more"

#: ``validate_gate``: skip field ``validate`` and schema-level validators.
SKIP_VALIDATORS: typing.Final = "skip_validators"

#: ``validate_gate``: freeze collected validation error containers.
FREEZE_ERRORS: typing.Final = "freeze_errors"

#: ``unknown_gate``: unknown keys are excluded.
FORCE_EXCLUDE: typing.Final = "force_exclude"

#: ``unknown_gate``: unknown keys raise.
FORCE_RAISE: typing.Final = "force_raise"

_SERDES_ADMIT_VALUES: typing.Final[frozenset[str]] = frozenset(
    {HONOR_CFG, DENY_NEW, ONE_MORE}
)
_VALIDATE_GATE_VALUES: typing.Final[frozenset[str]] = frozenset(
    {HONOR_CFG, SKIP_VALIDATORS, FREEZE_ERRORS}
)
_UNKNOWN_GATE_VALUES: typing.Final[frozenset[str]] = frozenset(
    {HONOR_CFG, FORCE_EXCLUDE, FORCE_RAISE}
)
_UNKNOWN_OVERRIDES: typing.Final[dict[str, types.UnknownOption]] = {
    FORCE_EXCLUDE: EXCLUDE,
    FORCE_RAISE: RAISE,
}


@dataclasses.dataclass(frozen=True)
class SerdesPolicyFrame:
    """An immutable triple of gate values plus internal admission bookkeeping.

    Instances are replaced (never mutated) when a ``one_more`` admission is
    consumed, so contexts that copied the enclosing stack -- e.g. asyncio
    tasks created inside an outer frame -- cannot consume each other's
    admissions.
    """

    serdes_admit: str = HONOR_CFG
    validate_gate: str = HONOR_CFG
    unknown_gate: str = HONOR_CFG
    _admitted: bool = False


_STACK: contextvars.ContextVar[tuple[SerdesPolicyFrame, ...]] = contextvars.ContextVar(
    "marshmallow_serdes_policy_stack", default=()
)


def _check_value(axis: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid {axis} value {value!r}. Valid values: {options}.")


def push_frame(
    *,
    serdes_admit: str = HONOR_CFG,
    validate_gate: str = HONOR_CFG,
    unknown_gate: str = HONOR_CFG,
) -> contextvars.Token[tuple[SerdesPolicyFrame, ...]]:
    """Push a policy frame onto the stack of the current collaboration scope.

    :return: A :class:`contextvars.Token` to pass to :func:`pop_frame`.
    """
    _check_value("serdes_admit", serdes_admit, _SERDES_ADMIT_VALUES)
    _check_value("validate_gate", validate_gate, _VALIDATE_GATE_VALUES)
    _check_value("unknown_gate", unknown_gate, _UNKNOWN_GATE_VALUES)
    frame = SerdesPolicyFrame(
        serdes_admit=serdes_admit,
        validate_gate=validate_gate,
        unknown_gate=unknown_gate,
    )
    return _STACK.set((*_STACK.get(), frame))


def pop_frame(token: contextvars.Token[tuple[SerdesPolicyFrame, ...]]) -> None:
    """Pop the frame pushed with ``token``, restoring the previous binding."""
    _STACK.reset(token)


@contextlib.contextmanager
def serdes_scope(
    *,
    serdes_admit: str = HONOR_CFG,
    validate_gate: str = HONOR_CFG,
    unknown_gate: str = HONOR_CFG,
) -> typing.Iterator[SerdesPolicyFrame]:
    """Context manager pushing a policy frame for the duration of the block.

    The frame is popped even if the block raises. Nested blocks see the
    innermost frame and restore the outer one on exit.
    """
    token = push_frame(
        serdes_admit=serdes_admit,
        validate_gate=validate_gate,
        unknown_gate=unknown_gate,
    )
    try:
        yield _STACK.get()[-1]
    finally:
        pop_frame(token)


def current_frame() -> SerdesPolicyFrame | None:
    """Return the innermost policy frame, or ``None`` when the stack is empty."""
    stack = _STACK.get()
    return stack[-1] if stack else None


def frame_depth() -> int:
    """Number of currently pushed frames in this collaboration scope."""
    return len(_STACK.get())


def admit_serdes(operation: str) -> None:
    """Gate the admission of a new top-level conversion.

    Called by ``Schema.load``/``loads``/``dump``/``dumps`` at entry. Raises
    :class:`SerdesAdmissionError` when the innermost frame denies the call.
    The check reads the stack at call time, so a ``one_more`` quota consumed
    by one conversion is visible to the next conversion immediately.
    """
    stack = _STACK.get()
    if not stack:
        return
    frame = stack[-1]
    if frame.serdes_admit == DENY_NEW:
        raise SerdesAdmissionError(operation=operation, gate=DENY_NEW)
    if frame.serdes_admit == ONE_MORE:
        if frame._admitted:
            raise SerdesAdmissionError(operation=operation, gate=ONE_MORE)
        _STACK.set((*stack[:-1], dataclasses.replace(frame, _admitted=True)))


def validators_skipped() -> bool:
    """Whether field and schema-level validators must be skipped right now."""
    stack = _STACK.get()
    return bool(stack) and stack[-1].validate_gate == SKIP_VALIDATORS


def errors_frozen() -> bool:
    """Whether collected validation errors must be frozen right now."""
    stack = _STACK.get()
    return bool(stack) and stack[-1].validate_gate == FREEZE_ERRORS


def effective_unknown(
    configured: types.UnknownOption,
) -> types.UnknownOption:
    """Return the unknown-key strategy merged from ``configured`` and the stack.

    ``configured`` is the value already resolved from ``Meta`` / the call
    argument (``EXCLUDE``, ``INCLUDE`` or ``RAISE``). A non-``honor_cfg``
    ``unknown_gate`` on the innermost frame overrides it; otherwise
    ``configured`` is returned untouched.
    """
    stack = _STACK.get()
    if stack:
        override = _UNKNOWN_OVERRIDES.get(stack[-1].unknown_gate)
        if override is not None:
            return override
    return configured


__all__ = [
    "DENY_NEW",
    "FORCE_EXCLUDE",
    "FORCE_RAISE",
    "FREEZE_ERRORS",
    "HONOR_CFG",
    "ONE_MORE",
    "SKIP_VALIDATORS",
    "SerdesPolicyFrame",
    "admit_serdes",
    "current_frame",
    "effective_unknown",
    "errors_frozen",
    "frame_depth",
    "pop_frame",
    "push_frame",
    "serdes_scope",
    "validators_skipped",
]
