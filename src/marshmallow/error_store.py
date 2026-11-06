"""Utilities for storing collections of error messages.

.. warning::

    This module is treated as private API.
    Users should not need to use this module directly.
"""

from __future__ import annotations

import typing

from marshmallow import runtime
from marshmallow.exceptions import SCHEMA, FrozenErrorStoreError


def _reject_mutation(*args, **kwargs):
    raise FrozenErrorStoreError(
        "Collected validation error messages are frozen by the serdes policy "
        "stack and cannot be modified or cleared."
    )


class _FrozenList(list):
    """List of collected error messages that cannot be mutated."""

    __slots__ = ()

    append = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    remove = _reject_mutation
    pop = _reject_mutation
    clear = _reject_mutation
    sort = _reject_mutation
    reverse = _reject_mutation
    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation


class _FrozenDict(dict):
    """Dict of collected error messages that cannot be mutated."""

    __slots__ = ()

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    clear = _reject_mutation
    update = _reject_mutation
    setdefault = _reject_mutation


def freeze_containers(value: typing.Any) -> typing.Any:
    """Recursively turn ``list``/``dict`` error containers into frozen ones.

    Already-frozen containers are returned untouched. Non-container values
    pass through unchanged.
    """
    if isinstance(value, (_FrozenList, _FrozenDict)):
        return value
    if isinstance(value, list):
        return _FrozenList(freeze_containers(item) for item in value)
    if isinstance(value, dict):
        return _FrozenDict(
            {key: freeze_containers(item) for key, item in value.items()}
        )
    return value


class ErrorStore:
    def __init__(self):
        #: Dictionary of errors stored during serialization
        self.errors: typing.Any = {}
        #: Whether ``errors`` is currently held as frozen containers
        self._frozen = False

    def store_error(self, messages, field_name=SCHEMA, index=None):
        # field error  -> store/merge error messages under field name key
        # schema error -> if string or list, store/merge under _schema key
        #              -> if dict, store/merge with other top-level keys
        messages = copy_containers(messages)
        if field_name != SCHEMA or not isinstance(messages, dict):
            messages = {field_name: messages}
        if index is not None:
            messages = {index: messages}
        # When the previous payload is frozen, merge against a mutable copy so
        # frozen containers handed out earlier can never be mutated in place.
        base = copy_containers(self.errors) if self._frozen else self.errors
        merged = merge_errors(base, messages)
        freeze = runtime.errors_frozen()
        self.errors = freeze_containers(merged) if freeze else merged
        self._frozen = freeze


def copy_containers(errors):
    if isinstance(errors, list):
        return [copy_containers(val) for val in errors]
    if isinstance(errors, dict):
        return {key: copy_containers(val) for key, val in errors.items()}
    return errors


def merge_errors(errors1, errors2):  # noqa: PLR0911
    """Deeply merge two error messages.

    The format of ``errors1`` and ``errors2`` matches the ``message``
    parameter of :exc:`marshmallow.exceptions.ValidationError`.
    """
    if not errors1:
        return errors2
    if not errors2:
        return errors1
    if isinstance(errors1, list):
        if isinstance(errors2, list):
            errors1.extend(errors2)
            return errors1
        if isinstance(errors2, dict):
            errors2[SCHEMA] = merge_errors(errors1, errors2.get(SCHEMA))
            return errors2
        errors1.append(errors2)
        return errors1
    if isinstance(errors1, dict):
        if isinstance(errors2, dict):
            for key, val in errors2.items():
                if key in errors1:
                    errors1[key] = merge_errors(errors1[key], val)
                else:
                    errors1[key] = val
            return errors1
        errors1[SCHEMA] = merge_errors(errors1.get(SCHEMA), errors2)
        return errors1
    if isinstance(errors2, list):
        return [errors1, *errors2]
    if isinstance(errors2, dict):
        errors2[SCHEMA] = merge_errors(errors1, errors2.get(SCHEMA))
        return errors2
    return [errors1, errors2]
