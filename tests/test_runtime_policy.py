"""Tests for the runtime (de)serialization policy stack (`marshmallow.runtime`)."""

from __future__ import annotations

import asyncio
import threading
import typing

import pytest

from marshmallow import (
    DENY_NEW,
    EXCLUDE,
    FORCE_EXCLUDE,
    FORCE_RAISE,
    FREEZE_ERRORS,
    HONOR_CFG,
    INCLUDE,
    ONE_MORE,
    RAISE,
    SKIP_VALIDATORS,
    FrozenErrorStoreError,
    Schema,
    SerdesAdmissionError,
    ValidationError,
    fields,
    post_dump,
    post_load,
    pre_dump,
    pre_load,
    validates,
    validates_schema,
)
from marshmallow.error_store import ErrorStore
from marshmallow.runtime import (
    current_frame,
    frame_depth,
    pop_frame,
    push_frame,
    serdes_scope,
)


class AuxSchema(Schema):
    n = fields.Int()


class FlatSchema(Schema):
    name = fields.Str(required=True)
    age = fields.Int()


def _reject_non_positive(value):
    if value <= 0:
        raise ValidationError("must be positive")


class ValidatedSchema(Schema):
    n = fields.Int(validate=_reject_non_positive)

    @validates("n")
    def validate_n(self, value, **kwargs):
        if value % 2 == 1:
            raise ValidationError("must be even")

    @validates_schema
    def validate_schema(self, data, **kwargs):
        if data.get("n") == 42:
            raise ValidationError("42 is forbidden")


class AlwaysSchemaErrorSchema(Schema):
    n = fields.Int(validate=_reject_non_positive)
    m = fields.Int()

    @validates_schema(skip_on_field_errors=False)
    def validate_schema(self, data, **kwargs):
        raise ValidationError("always schema error")


class RequiredSchema(Schema):
    n = fields.Int(required=True)
    m = fields.Int()


class NestedFlatSchema(Schema):
    n = fields.Int(validate=_reject_non_positive)


class HasNestedSchema(Schema):
    child = fields.Nested(NestedFlatSchema)


class HasNestedListSchema(Schema):
    items = fields.List(fields.Nested(NestedFlatSchema))


class TestStackMechanics:
    def test_empty_stack_is_none(self):
        assert current_frame() is None
        assert frame_depth() == 0

    def test_frame_values_visible_inside(self):
        with serdes_scope(
            serdes_admit=DENY_NEW,
            validate_gate=SKIP_VALIDATORS,
            unknown_gate=FORCE_RAISE,
        ) as frame:
            assert frame.serdes_admit == DENY_NEW
            assert frame.validate_gate == SKIP_VALIDATORS
            assert frame.unknown_gate == FORCE_RAISE
            assert frame_depth() == 1
        assert frame_depth() == 0

    def test_invalid_gate_values_rejected(self):
        with pytest.raises(ValueError, match="serdes_admit"):
            push_frame(serdes_admit="nope")
        with pytest.raises(ValueError, match="validate_gate"):
            push_frame(validate_gate="nope")
        with pytest.raises(ValueError, match="unknown_gate"):
            push_frame(unknown_gate="nope")
        assert frame_depth() == 0

    def test_inner_frame_shadows_then_restores_outer(self):
        with serdes_scope(serdes_admit=DENY_NEW):
            with pytest.raises(SerdesAdmissionError):
                AuxSchema().load({"n": 1})
            with serdes_scope(serdes_admit=HONOR_CFG):
                assert AuxSchema().load({"n": 1}) == {"n": 1}
            # Inner frame popped: outer gate active again
            with pytest.raises(SerdesAdmissionError):
                AuxSchema().load({"n": 1})
        assert frame_depth() == 0

    def test_exception_pops_frame_cleanly(self):
        with pytest.raises(RuntimeError, match="boom"):
            with serdes_scope(serdes_admit=DENY_NEW):
                raise RuntimeError("boom")
        assert frame_depth() == 0
        # Conversions work normally again
        assert AuxSchema().load({"n": 1}) == {"n": 1}

    def test_manual_push_pop(self):
        token = push_frame(serdes_admit=DENY_NEW)
        with pytest.raises(SerdesAdmissionError):
            AuxSchema().load({"n": 1})
        pop_frame(token)
        assert AuxSchema().load({"n": 1}) == {"n": 1}

    def test_inner_frame_restores_independent_axes(self):
        # Outer freezes errors; inner skips validators; after pop the outer
        # validate gate is observable again.
        with serdes_scope(validate_gate=FREEZE_ERRORS):
            with serdes_scope(validate_gate=SKIP_VALIDATORS):
                assert ValidatedSchema().load({"n": 1}) == {"n": 1}
            with pytest.raises(ValidationError):
                ValidatedSchema().load({"n": 1})


class TestSerdesAdmit:
    def test_deny_new_rejects_all_four_entries(self):
        s = AuxSchema()
        with serdes_scope(serdes_admit=DENY_NEW):
            with pytest.raises(SerdesAdmissionError) as exc:
                s.load({"n": 1})
            assert exc.value.operation == "load"
            assert exc.value.gate == DENY_NEW
            with pytest.raises(SerdesAdmissionError) as exc:
                s.loads('{"n": 1}')
            assert exc.value.operation == "loads"
            with pytest.raises(SerdesAdmissionError) as exc:
                s.dump({"n": 1})
            assert exc.value.operation == "dump"
            with pytest.raises(SerdesAdmissionError) as exc:
                s.dumps({"n": 1})
            assert exc.value.operation == "dumps"

    def test_loads_rejected_before_json_parsing(self):
        s = AuxSchema()
        with serdes_scope(serdes_admit=DENY_NEW):
            # If parsing ran first this would raise json.JSONDecodeError instead.
            with pytest.raises(SerdesAdmissionError):
                s.loads("not valid json {{{")

    def test_unpressed_conversion_in_same_process_unaffected(self):
        # The frame is scope-bound: code outside the block follows pin behavior.
        s = FlatSchema()
        with serdes_scope(serdes_admit=DENY_NEW):
            with pytest.raises(SerdesAdmissionError):
                s.load({"name": "a"})
        assert s.load({"name": "a", "extra": 1}, unknown=EXCLUDE) == {"name": "a"}

    def test_deny_new_pushed_during_field_processing_does_not_cancel_outer(self):
        class PushPopField(fields.Raw):
            def _deserialize(self, value, attr, data, **kwargs):
                token = push_frame(serdes_admit=DENY_NEW)
                try:
                    # A new conversion attempted after push is denied...
                    with pytest.raises(SerdesAdmissionError):
                        AuxSchema().load({"n": 9})
                finally:
                    pop_frame(token)
                # ...but the outer conversion keeps running with real data.
                return value

        class MidSchema(Schema):
            probe = PushPopField()
            tail = fields.Str()

        assert MidSchema().load({"probe": "ok", "tail": "end"}) == {
            "probe": "ok",
            "tail": "end",
        }

    def test_deny_new_pushed_in_preload_does_not_cancel_flat_load(self):
        class FlatWithHook(Schema):
            name = fields.Str()

            @pre_load
            def push_then_pop(self, data, **kwargs):
                token = push_frame(serdes_admit=DENY_NEW)
                pop_frame(token)
                return data

        assert FlatWithHook().load({"name": "a"}) == {"name": "a"}

    def test_deny_new_pushed_from_field_blocks_later_nested_field(self):
        tokens: list = []

        class PushField(fields.Raw):
            def _deserialize(self, value, attr, data, **kwargs):
                tokens.append(push_frame(serdes_admit=DENY_NEW))
                return value

        class Parent(Schema):
            gate = PushField()
            child = fields.Nested(NestedFlatSchema)

            @post_load
            def pop_gate(self, data, **kwargs):
                pop_frame(tokens.pop())
                return data

        try:
            # "gate" is processed first and pushes while field processing is
            # already running; the later nested load is a new conversion.
            with pytest.raises(SerdesAdmissionError):
                Parent().load({"gate": "x", "child": {"n": 1}})
        finally:
            if tokens:
                pop_frame(tokens.pop())
        assert frame_depth() == 0

    def test_deny_new_pushed_from_field_flat_conversion_completes(self):
        tokens: list = []

        class PushField(fields.Raw):
            def _deserialize(self, value, attr, data, **kwargs):
                tokens.append(push_frame(serdes_admit=DENY_NEW))
                return value

        class FlatPushSchema(Schema):
            gate = PushField()
            tail = fields.Str()

            @post_load
            def pop_gate(self, data, **kwargs):
                pop_frame(tokens.pop())
                return data

        # The already-running conversion is not cancelled or emptied.
        assert FlatPushSchema().load({"gate": "x", "tail": "y"}) == {
            "gate": "x",
            "tail": "y",
        }
        assert tokens == []

    def test_deny_new_rejects_nested_load_started_after_push(self):
        tokens: list = []

        class Parent(Schema):
            child = fields.Nested(NestedFlatSchema)

            @pre_load
            def push_deny(self, data, **kwargs):
                tokens.append(push_frame(serdes_admit=DENY_NEW))
                return data

        try:
            # The outer load was admitted before the push; the nested load is a
            # new conversion and is rejected (not swallowed as empty success).
            with pytest.raises(SerdesAdmissionError):
                Parent().load({"child": {"n": 1}})
        finally:
            pop_frame(tokens.pop())
        assert frame_depth() == 0

    def test_deny_new_pushed_in_predump_flat_dump_completes(self):
        class FlatDump(Schema):
            name = fields.Str()

            @pre_dump
            def push_then_pop(self, data, **kwargs):
                token = push_frame(serdes_admit=DENY_NEW)
                pop_frame(token)
                return data

        assert FlatDump().dump({"name": "a"}) == {"name": "a"}

    def test_deny_new_rejects_nested_dump_started_after_push(self):
        tokens: list = []

        class Parent(Schema):
            child = fields.Nested(NestedFlatSchema)

            @pre_dump
            def push_deny(self, data, **kwargs):
                tokens.append(push_frame(serdes_admit=DENY_NEW))
                return data

            @post_dump
            def pop_gate(self, data, **kwargs):
                pop_frame(tokens.pop())
                return data

        try:
            with pytest.raises(SerdesAdmissionError):
                Parent().dump({"child": {"n": 1}})
        finally:
            if tokens:
                pop_frame(tokens.pop())
        assert frame_depth() == 0

    def test_one_more_admits_exactly_one_new_conversion(self):
        s = AuxSchema()
        with serdes_scope(serdes_admit=ONE_MORE):
            assert s.load({"n": 1}) == {"n": 1}
            # The second conversion (third counting the one that pushed the
            # frame) must not enter.
            with pytest.raises(SerdesAdmissionError) as exc:
                s.load({"n": 2})
            assert exc.value.gate == ONE_MORE
            # The quota stays consumed.
            with pytest.raises(SerdesAdmissionError):
                s.dump({"n": 3})

    def test_one_more_re_reads_stack_top_per_entry(self):
        class OneMoreField(fields.Raw):
            def _deserialize(self, value, attr, data, **kwargs):
                aux = AuxSchema()
                with serdes_scope(serdes_admit=ONE_MORE):
                    assert aux.load({"n": 1}) == {"n": 1}
                    with pytest.raises(SerdesAdmissionError):
                        aux.load({"n": 2})
                    # A nested frame shadows the quota: re-reading the top now
                    # admits again.
                    with serdes_scope(serdes_admit=HONOR_CFG):
                        assert aux.load({"n": 3}) == {"n": 3}
                    # Inner frame popped: quota still exhausted.
                    with pytest.raises(SerdesAdmissionError):
                        aux.load({"n": 4})
                # Frame popped: back to normal.
                assert aux.load({"n": 5}) == {"n": 5}
                return value

        class OneMoreSchema(Schema):
            probe = OneMoreField()

        assert OneMoreSchema().load({"probe": "x"}) == {"probe": "x"}

    def test_one_more_counts_nested_loads_inside_running_conversion(self):
        tokens: list = []

        class Parent(Schema):
            items = fields.List(fields.Nested(NestedFlatSchema))

            @pre_load
            def push_one_more(self, data, **kwargs):
                tokens.append(push_frame(serdes_admit=ONE_MORE))
                return data

        try:
            # First nested item consumes the "one more"; the second is denied.
            with pytest.raises(SerdesAdmissionError):
                Parent().load({"items": [{"n": 1}, {"n": 2}]})
        finally:
            pop_frame(tokens.pop())

    def test_one_more_loads_counts_once(self):
        s = AuxSchema()
        with serdes_scope(serdes_admit=ONE_MORE):
            assert s.loads('{"n": 1}') == {"n": 1}
            with pytest.raises(SerdesAdmissionError):
                s.loads('{"n": 2}')

    def test_one_more_dumps_counts_once(self):
        s = AuxSchema()
        with serdes_scope(serdes_admit=ONE_MORE):
            assert s.dumps({"n": 1}) == '{"n": 1}'
            with pytest.raises(SerdesAdmissionError):
                s.dumps({"n": 2})


class TestSkipValidators:
    def test_field_validate_list_skipped(self):
        s = ValidatedSchema()
        with pytest.raises(ValidationError):
            s.load({"n": -1})
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            assert s.load({"n": -1}) == {"n": -1}

    def test_validates_decorator_skipped(self):
        s = ValidatedSchema()
        with pytest.raises(ValidationError):
            s.load({"n": 1})
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            assert s.load({"n": 1}) == {"n": 1}

    def test_validates_schema_skipped(self):
        s = ValidatedSchema()
        with pytest.raises(ValidationError):
            s.load({"n": 42})
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            assert s.load({"n": 42}) == {"n": 42}
            assert s.validate({"n": 42}) == {}

    def test_required_missing_still_detected(self):
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            with pytest.raises(ValidationError) as exc:
                RequiredSchema().load({})
        assert "n" in exc.value.messages

    def test_invalid_type_still_detected(self):
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            with pytest.raises(ValidationError) as exc:
                RequiredSchema().load({"n": "not an int"})
        assert "n" in exc.value.messages

    def test_null_still_rejected(self):
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            with pytest.raises(ValidationError) as exc:
                RequiredSchema().load({"n": None})
        assert "n" in exc.value.messages

    def test_nested_field_validators_skipped(self):
        s = HasNestedSchema()
        with pytest.raises(ValidationError):
            s.load({"child": {"n": -1}})
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            assert s.load({"child": {"n": -1}}) == {"child": {"n": -1}}

    def test_list_inner_field_validators_skipped(self):
        class ListSchema(Schema):
            nums = fields.List(fields.Int(validate=_reject_non_positive))

        s = ListSchema()
        with pytest.raises(ValidationError):
            s.load({"nums": [-1, -2]})
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            assert s.load({"nums": [-1, -2]}) == {"nums": [-1, -2]}

    def test_tuple_inner_field_validators_skipped(self):
        class TupleSchema(Schema):
            pair = fields.Tuple((fields.Int(validate=_reject_non_positive),))

        s = TupleSchema()
        with pytest.raises(ValidationError):
            s.load({"pair": [-1]})
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            assert s.load({"pair": [-1]}) == {"pair": (-1,)}

    def test_mapping_inner_field_validators_skipped(self):
        class DictSchema(Schema):
            mapping = fields.Dict(
                keys=fields.Str(), values=fields.Int(validate=_reject_non_positive)
            )

        s = DictSchema()
        with pytest.raises(ValidationError):
            s.load({"mapping": {"a": -1}})
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            assert s.load({"mapping": {"a": -1}}) == {"mapping": {"a": -1}}

    def test_post_load_processors_still_run(self):
        class PostSchema(Schema):
            n = fields.Int(
                validate=_reject_non_positive,
                post_load=lambda value: value * 10,
            )

        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            assert PostSchema().load({"n": -1}) == {"n": -10}

    def test_validate_list_not_mutated(self):
        s = ValidatedSchema()
        validators_before = s.fields["n"].validators
        with serdes_scope(validate_gate=SKIP_VALIDATORS):
            s.load({"n": -1})
        assert s.fields["n"].validators is validators_before
        assert s.fields["n"].validators == validators_before


class TestFreezeErrors:
    def test_failure_still_raised_with_all_errors_and_valid_data(self):
        s = AlwaysSchemaErrorSchema()
        with serdes_scope(validate_gate=FREEZE_ERRORS):
            with pytest.raises(ValidationError) as exc:
                s.load({"n": -1, "m": 2})
        assert "n" in exc.value.messages
        assert "_schema" in exc.value.messages
        assert exc.value.valid_data == {"m": 2}

    def test_success_is_not_faked(self):
        with serdes_scope(validate_gate=FREEZE_ERRORS):
            assert AuxSchema().load({"n": 1}) == {"n": 1}
            assert AuxSchema().dump({"n": 1}) == {"n": 1}

    def test_messages_dict_and_lists_are_immutable(self):
        with serdes_scope(validate_gate=FREEZE_ERRORS):
            with pytest.raises(ValidationError) as exc:
                ValidatedSchema().load({"n": -1})
        messages = typing.cast("dict", exc.value.messages)
        with pytest.raises(FrozenErrorStoreError):
            messages.clear()
        with pytest.raises(FrozenErrorStoreError):
            messages.pop("n")
        with pytest.raises(FrozenErrorStoreError):
            messages["n"].append("tampered")
        with pytest.raises(TypeError):
            messages["new"] = ["x"]

    def test_nested_indexed_messages_are_frozen_deeply(self):
        class Outer(Schema):
            items = fields.List(fields.Nested(RequiredSchema))

        with serdes_scope(validate_gate=FREEZE_ERRORS):
            with pytest.raises(ValidationError) as exc:
                Outer().load({"items": [{"n": 1}, {}]})
        messages = typing.cast("dict", exc.value.messages)
        assert messages == {"items": {1: {"n": ["Missing data for required field."]}}}
        with pytest.raises(FrozenErrorStoreError):
            messages["items"][1]["n"].append("tampered")
        with pytest.raises(FrozenErrorStoreError):
            messages["items"].pop(1)

    def test_pre_load_error_is_frozen(self):
        class PreSchema(Schema):
            @pre_load
            def boom(self, data, **kwargs):
                raise ValidationError("pre failure", "x")

        with serdes_scope(validate_gate=FREEZE_ERRORS):
            with pytest.raises(ValidationError) as exc:
                PreSchema().load({})
        with pytest.raises(FrozenErrorStoreError):
            exc.value.messages.clear()

    def test_post_load_error_is_frozen(self):
        class PostSchema(Schema):
            n = fields.Int()

            @post_load
            def boom(self, data, **kwargs):
                raise ValidationError("post failure", "x")

        with serdes_scope(validate_gate=FREEZE_ERRORS):
            with pytest.raises(ValidationError) as exc:
                PostSchema().load({"n": 1})
        with pytest.raises(FrozenErrorStoreError):
            exc.value.messages.clear()

    def test_error_store_collect_under_freeze_is_append_only(self):
        store = ErrorStore()
        with serdes_scope(validate_gate=FREEZE_ERRORS):
            store.store_error(["a"], "f1")
            snapshot = store.errors
            assert snapshot == {"f1": ["a"]}
            with pytest.raises(FrozenErrorStoreError):
                snapshot["f1"].append("x")
            with pytest.raises(FrozenErrorStoreError):
                snapshot.clear()
            # A later collector may add errors, but never mutates the snapshot.
            store.store_error(["b"], "f2")
            assert snapshot == {"f1": ["a"]}
            assert store.errors == {"f1": ["a"], "f2": ["b"]}
            with pytest.raises(FrozenErrorStoreError):
                store.errors["f1"].append("y")
        # Outside the frame a fresh store behaves like the pin API.
        plain = ErrorStore()
        plain.store_error(["a"], "f1")
        plain.errors["f1"].append("mutable")
        assert plain.errors == {"f1": ["a", "mutable"]}

    def test_error_store_thaws_after_frame_pops(self):
        store = ErrorStore()
        with serdes_scope(validate_gate=FREEZE_ERRORS):
            store.store_error(["a"], "f1")
        store.store_error(["b"], "f2")
        assert not store._frozen
        assert store.errors == {"f1": ["a"], "f2": ["b"]}
        # Mutable again.
        store.errors["f1"].append("c")
        assert store.errors["f1"] == ["a", "c"]


class RaiseUnknownSchema(Schema):
    n = fields.Int()

    class Meta:
        unknown = RAISE


class IncludeUnknownSchema(Schema):
    n = fields.Int()

    class Meta:
        unknown = INCLUDE


class TestUnknownGate:
    def test_force_exclude_overrides_meta_raise(self):
        s = RaiseUnknownSchema()
        with pytest.raises(ValidationError):
            s.load({"n": 1, "x": 2})
        with serdes_scope(unknown_gate=FORCE_EXCLUDE):
            assert s.load({"n": 1, "x": 2}) == {"n": 1}

    def test_force_raise_overrides_call_argument_include(self):
        s = RaiseUnknownSchema()
        assert s.load({"n": 1, "x": 2}, unknown=INCLUDE) == {"n": 1, "x": 2}
        with serdes_scope(unknown_gate=FORCE_RAISE):
            with pytest.raises(ValidationError) as exc:
                s.load({"n": 1, "x": 2}, unknown=INCLUDE)
        assert "x" in exc.value.messages

    def test_force_exclude_overrides_meta_include(self):
        s = IncludeUnknownSchema()
        assert s.load({"n": 1, "x": 2}) == {"n": 1, "x": 2}
        with serdes_scope(unknown_gate=FORCE_EXCLUDE):
            assert s.load({"n": 1, "x": 2}) == {"n": 1}

    def test_force_raise_in_nested_schema(self):
        class Parent(Schema):
            child = fields.Nested(RaiseUnknownSchema)

        with serdes_scope(unknown_gate=FORCE_EXCLUDE):
            assert Parent().load({"child": {"n": 1, "z": 2}}) == {"child": {"n": 1}}
        with serdes_scope(unknown_gate=FORCE_RAISE):
            with pytest.raises(ValidationError):
                Parent().load({"child": {"n": 1, "z": 2}}, unknown=EXCLUDE)

    def test_force_exclude_with_many(self):
        s = IncludeUnknownSchema()
        with serdes_scope(unknown_gate=FORCE_EXCLUDE):
            assert s.load([{"n": 1, "x": 2}, {"n": 2}], many=True) == [
                {"n": 1},
                {"n": 2},
            ]

    def test_gate_does_not_change_partial(self):
        class PartialSchema(Schema):
            n = fields.Int(required=True)

        with serdes_scope(unknown_gate=FORCE_RAISE):
            assert PartialSchema().load({}, partial=True) == {}
            with pytest.raises(ValidationError):
                PartialSchema().load({"z": 1}, partial=True)

    def test_honor_cfg_follows_call_argument(self):
        s = RaiseUnknownSchema()
        with serdes_scope(unknown_gate=HONOR_CFG):
            assert s.load({"n": 1, "x": 2}, unknown=INCLUDE) == {"n": 1, "x": 2}
            assert s.load({"n": 1, "x": 2}, unknown=EXCLUDE) == {"n": 1}

    def test_meta_and_instance_options_are_not_mutated(self):
        s = RaiseUnknownSchema()
        with serdes_scope(unknown_gate=FORCE_EXCLUDE):
            s.load({"n": 1, "x": 2})
        assert s.unknown == RAISE
        assert s.opts.unknown == RAISE


class TestScopeIsolation:
    def test_threads_do_not_share_stacks(self):
        barrier = threading.Barrier(3)
        results: dict[str, object] = {}

        def deny_worker():
            with serdes_scope(serdes_admit=DENY_NEW):
                barrier.wait()
                try:
                    AuxSchema().load({"n": 1})
                except SerdesAdmissionError:
                    results["deny"] = "rejected"
                else:
                    results["deny"] = "accepted"

        def one_more_worker():
            with serdes_scope(serdes_admit=ONE_MORE):
                barrier.wait()
                first = AuxSchema().load({"n": 1})
                try:
                    AuxSchema().load({"n": 2})
                except SerdesAdmissionError:
                    second = "rejected"
                else:
                    second = "accepted"
                results["one_more"] = (first, second)

        threads = [
            threading.Thread(target=deny_worker),
            threading.Thread(target=one_more_worker),
        ]
        for thread in threads:
            thread.start()
        # Main thread participates while both frames are active: no frame
        # here, so the original API (including INCLUDE) must keep working.
        barrier.wait()
        results["plain"] = AuxSchema().load({"n": 1, "x": 2}, unknown=INCLUDE)
        for thread in threads:
            thread.join()

        assert results["deny"] == "rejected"
        assert results["one_more"] == ({"n": 1}, "rejected")
        assert results["plain"] == {"n": 1, "x": 2}

    def test_asyncio_tasks_do_not_share_stacks(self):
        async def deny_task():
            with serdes_scope(serdes_admit=DENY_NEW):
                await asyncio.sleep(0)
                with pytest.raises(SerdesAdmissionError):
                    AuxSchema().load({"n": 1})

        async def one_more_task():
            with serdes_scope(serdes_admit=ONE_MORE):
                await asyncio.sleep(0)
                assert AuxSchema().load({"n": 1}) == {"n": 1}
                with pytest.raises(SerdesAdmissionError):
                    AuxSchema().load({"n": 2})

        async def main():
            await asyncio.gather(deny_task(), one_more_task())

        asyncio.run(main())

    def test_one_more_quota_not_shared_into_child_contexts(self):
        # Frames visible to copied contexts (asyncio tasks) must not share the
        # mutable one_more quota: each child admits its own conversion.
        async def child():
            assert AuxSchema().load({"n": 1}) == {"n": 1}
            with pytest.raises(SerdesAdmissionError):
                AuxSchema().load({"n": 2})

        async def main():
            with serdes_scope(serdes_admit=ONE_MORE):
                await asyncio.gather(child(), child())
                # Children's admissions did not consume the parent quota.
                assert AuxSchema().load({"n": 3}) == {"n": 3}
                with pytest.raises(SerdesAdmissionError):
                    AuxSchema().load({"n": 4})

        asyncio.run(main())
        assert frame_depth() == 0
