"""Public Topic 28 interfaces exercised on the minimum supported Python."""

import pytest


class TestTargetFeaturesCompatibility:
    @pytest.mark.parametrize(
        "isa,abi,expected",
        [
            ("RV32IM", "ILP32", frozenset({"I", "M"})),
            ("RV32IMF", "ILP32F", frozenset({"I", "M", "F"})),
            ("RV32IMFD", "ILP32D", frozenset({"I", "M", "F", "D"})),
        ],
    )
    def test_normalized_targets_expose_their_extensions(self, isa, abi, expected):
        from scratchv.backend.target_features import TargetFeatures

        features = TargetFeatures(isa, abi)
        assert features.extensions == expected


class TestSchedulerCompatibility:
    def test_keyword_only_target_preserves_immutable_instruction_behavior(self):
        from dataclasses import FrozenInstanceError, replace

        from scratchv.backend.schedule_semantics import SchedInst

        instruction = SchedInst(0, "BNEZ", ["t0", ".next"], target=".next")
        assert instruction.opcode == "bnez"
        assert instruction.operands == ("t0", ".next")
        assert instruction.uses == frozenset({"x5"})
        assert instruction.target == ".next"
        assert replace(instruction, id=1).target == ".next"
        with pytest.raises(FrozenInstanceError):
            instruction.target = ".other"
        with pytest.raises(TypeError):
            SchedInst(0, ".label", (), frozenset(), frozenset(), "", 0, ".next")


class TestTypedInitializerCompatibility:
    @pytest.mark.parametrize(
        "dtype,value,expected_bits",
        [(1, -0.0, 0x80000000), (11, 1.0, 0x3FF0000000000000)],
    )
    def test_reads_scalar_typed_encoding(self, dtype, value, expected_bits, tmp_path):
        import onnx
        from onnx import helper

        from scratchv.frontend.onnx_bits import typed_initializer_bits

        graph = helper.make_graph(
            [],
            "main",
            [],
            [helper.make_tensor_value_info("value", dtype, [])],
            [helper.make_tensor("value", dtype, [], [value])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.checker.check_model(model)
        path = tmp_path / "constant.onnx"
        onnx.save(model, path)

        assert typed_initializer_bits(str(path), onnx) == {
            ("value", dtype): expected_bits,
        }
