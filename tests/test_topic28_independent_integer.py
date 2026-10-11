"""N1/N3/N4: typed integer pipelines must preserve values and valid IR."""

import numpy as np
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T, OpCode
from tests.topic28_runtime import execute_i32


LEVELS = ("none", "basic", "all")


def integer_model(tmp_path, operator, final):
    onnx = pytest.importorskip("onnx")
    values = {"a": 2147483647, "b": 1 if operator == "Add" else 2,
              "c": 2 if final == "Div" else 0}
    graph = onnx.helper.make_graph([
        onnx.helper.make_node(operator, ["a", "b"], ["intermediate"]),
        onnx.helper.make_node(final, ["intermediate", "c"], ["out"]),
    ], "main", [], [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.INT32, [])],
        [onnx.numpy_helper.from_array(np.asarray(v, dtype=np.int32), name=k)
         for k, v in values.items()])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = tmp_path / "integer.onnx"
    onnx.save(model, path)
    return path, model


@pytest.mark.parametrize("operator", ["Add", "Mul"])
def test_n1_parser_leaves_integer_arithmetic_to_typed_folder(operator, tmp_path):
    path, _ = integer_model(tmp_path, operator, "Div")
    program = ONNXParser().parse(str(path))
    assert verify_ir(program) == (True, [])
    opcode = OpCode.ADD if operator == "Add" else OpCode.MUL
    assert any(i.opcode is opcode for f in program.functions for b in f.blocks for i in b.instructions)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("operator,final,expected", [
    ("Add", "Div", -1073741824), ("Add", "Min", -2147483648),
    ("Mul", "Div", -1), ("Mul", "Min", -2),
])
def test_n1_overflow_chain_matches_onnx_and_qemu(level, operator, final, expected, tmp_path):
    from onnx.reference import ReferenceEvaluator

    path, model = integer_model(tmp_path, operator, final)
    reference = ReferenceEvaluator(model).run(None, {})[0]
    assert reference.dtype == np.int32 and int(reference) == expected
    result = CompilerDriver(CompilerConfig(
        use_extended_isel=True, reg_alloc="greedy", optimize_level=level, verify_ir=True,
    )).compile(str(path), str(tmp_path / "out.s"))
    assert result.success, result.errors
    assert execute_i32(result.output_text, tmp_path) == expected


def memory_program(case, input_value):
    b = IRBuilder()
    b.new_function("main")
    b.new_block("entry")
    p = b.alloca(1, T.INT32)
    b.store(p, b.load_const(input_value, T.INT32))
    r = b.rem(b.load(p, T.INT32), b.load_const(5, T.INT32))
    if case == "n3":
        y = b.add(r, b.load_const(0, T.INT32))
    else:
        two = b.load_const(2, T.INT32)
        three = b.load_const(3, T.INT32)
        y = b.add(b.mul(r, two), three)
    b.ret(y)
    assert verify_ir(b.program) == (True, [])
    return b.program


def compile_memory(tmp_path, case, input_value, mode, verify, level="none", passes=None):
    program = memory_program(case, input_value)
    driver = CompilerDriver(CompilerConfig(
        use_extended_isel=True, reg_alloc=mode, optimize_level=level,
        verify_ir=verify, passes=passes,
    ))
    driver._parse = lambda *args: program
    result = driver.compile("", str(tmp_path / "out.s"), dsl_source="")
    assert result.success, result.errors
    assert verify_ir(program) == (True, [])
    assert all(len(i.operands) == 2 for f in program.functions for b in f.blocks
               for i in b.instructions if i.opcode in {OpCode.ADD, OpCode.MUL})
    return execute_i32(result.output_text, tmp_path)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("input_value", [7, -7])
def test_n3_integer_identity_keeps_result_definition(level, input_value, tmp_path):
    assert compile_memory(tmp_path, "n3", input_value, "greedy", True, level) == (2 if input_value > 0 else -2)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("input_value", [7, -7])
def test_n4_integer_mul_add_keeps_both_operations(level, input_value, tmp_path):
    assert compile_memory(tmp_path, "n4", input_value, "greedy", True, level) == (7 if input_value > 0 else -1)


@pytest.mark.parametrize("passes", [("ir-peephole",), ("ir-peephole", "ir-peephole")])
def test_n3_explicit_identity_passes_are_safe(passes, tmp_path):
    assert compile_memory(tmp_path, "n3", 7, "greedy", True, passes=passes) == 2


@pytest.mark.parametrize("passes", [("muladd-fusion",), ("muladd-fusion", "muladd-fusion")])
def test_n4_explicit_fusion_passes_are_safe(passes, tmp_path):
    assert compile_memory(tmp_path, "n4", 7, "greedy", True, passes=passes) == 7
