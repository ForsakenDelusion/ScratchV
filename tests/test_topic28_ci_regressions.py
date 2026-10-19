"""CI regressions from combining scalar selection with the tensor frontend."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper, numpy_helper

from benchmarks.bench_runner import BenchmarkRunner
from scratchv.backend.abi_frame import apply_abi_frames
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.machine_types import (
    MachineInstr as MI, MachineOp as O, MachineOperand as P,
)
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.simulator.rv32_emulator import REG_ID, RV32Emulator
from scratchv.verification.ir_interpreter import IRInterpreter
from tests.topic28_runtime import execute_i32


ROOT = Path(__file__).resolve().parents[1]


def reshape_model(tmp_path, dimensions, *, intermediate=False, empty=False):
    result = "middle" if intermediate else "y"
    nodes = [helper.make_node("Cast", ["raw"], ["shape"], to=T.INT64),
             helper.make_node("Reshape", ["x", "shape"], [result])]
    if intermediate:
        nodes.append(helper.make_node("Identity", ["middle"], ["y"]))
    source_shape, target_shape = ([0, 3], [0, 3]) if empty else ([2, 3], [3, 2])
    graph = helper.make_graph(
        nodes, "main", [helper.make_tensor_value_info("x", T.FLOAT, source_shape)],
        [helper.make_tensor_value_info("y", T.FLOAT, target_shape if intermediate else dimensions)],
        [numpy_helper.from_array(np.array(target_shape, np.int64), "raw")],
        value_info=[helper.make_tensor_value_info("middle", T.FLOAT, dimensions)] if intermediate else [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    path = tmp_path / "shape.onnx"
    onnx.save(model, path)
    return path, tuple(source_shape), tuple(target_shape)


class TestONNXShapeAnnotations:
    @pytest.mark.parametrize("dimensions", [["N", "M"], [None, None]])
    @pytest.mark.parametrize("intermediate", [False, True])
    def test_unknown_dimensions_do_not_become_zero_sized_tensors(
        self, tmp_path, dimensions, intermediate,
    ):
        path, source_shape, expected_shape = reshape_model(
            tmp_path, dimensions, intermediate=intermediate,
        )
        parser = ONNXParser()
        program = parser.parse(str(path))
        source = np.arange(6, dtype=np.float32).reshape(source_shape)
        result = IRInterpreter(program).run(
            {"x": source}, initializers=parser.initializers,
        ).return_value
        np.testing.assert_array_equal(result, source.reshape(expected_shape))

    def test_concrete_zero_dimension_is_preserved(self, tmp_path):
        path, source_shape, expected_shape = reshape_model(tmp_path, [0, 3], empty=True)
        parser = ONNXParser()
        program = parser.parse(str(path))
        reshape = program.functions[0].blocks[0].instructions[-2]
        assert reshape.dest.shape == expected_shape
        result = IRInterpreter(program).run(
            {"x": np.empty(source_shape, np.float32)}, initializers=parser.initializers,
        ).return_value
        assert result.shape == expected_shape and result.dtype == np.float32


class TestBenchmarkFailureReporting:
    def test_undefined_input_fails_even_without_expected_output(self, tmp_path):
        source = tmp_path / "undefined.dsl"
        source.write_text("return missing\n")
        runner = BenchmarkRunner(str(tmp_path), verbose=False)
        result = runner.run_case({
            "name": "undefined", "dsl_path": str(source),
            "expected_path": "", "desc_path": "",
        })
        assert not result.passed
        assert "Undefined DSL value: missing" in result.error

    def test_existing_dsl_corpus_has_no_implicit_accumulators(self):
        report = BenchmarkRunner(str(ROOT / "benchmarks/cases"), verbose=False).run_all()
        assert report.fail_count == 0, [
            (row.name, row.error, row.output) for row in report.results if not row.passed
        ]
        assert all("SIM ERROR" not in row.output for row in report.results)


class TestLoopCostGate:
    def test_tinyfive_loop_results_and_existing_cost_limits_pass(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/run_topic06_benchmarks.py"),
            "--category", "loop", "--fail-on-test-failure",
        ], cwd=ROOT, env=environment, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Passed: 3" in result.stdout and "Failed: 0" in result.stdout


class TestSignedImmediateLoopComparison:
    @pytest.mark.parametrize("value,bound,expected", [
        (-2147483648, -2048, 1), (-2048, -2048, 0), (-1, 0, 1),
        (0, -1, 0), (2046, 2047, 1), (2047, 2047, 0), (2147483647, 2047, 0),
    ])
    def test_encoder_and_emulator_match_external_riscv_execution(
        self, tmp_path, value, bound, expected,
    ):
        assembly = f".text\n.globl main\nmain:\nli t0,{value}\nslti a0,t0,{bound}\nret\n"
        assert execute_i32(assembly, tmp_path) == expected
        emulator = RV32Emulator(mem_size=4096)
        emulator.load_code(bytes(RISCVAEncoder().assemble(assembly)))
        emulator.run(max_instr=16)
        assert emulator.regs[REG_ID["a0"]] == expected


class TestGreedyFallthroughAllocation:
    @pytest.mark.parametrize("argument,expected", [(-1, 12), (0, 21), (1, 12)])
    def test_single_predecessor_mappings_do_not_escape_into_a_join(
        self, tmp_path, argument, expected,
    ):
        value, delta = P.vreg("value"), P.vreg("delta")
        stream = [
            MI(O.LABEL, target="main", label_role="function"),
            MI(O.LI, value, P.immediate(9)),
            MI(O.LI, delta, P.immediate(1)),
            MI(O.LABEL, target=".step"),
            MI(O.ADD, value, value, delta),
            MI(O.BEQ, P.reg("a0"), P.reg("zero"), target=".alternate"),
            MI(O.LABEL, target=".body"),
            MI(O.ADD, value, value, delta),
            MI(O.J, target=".join"),
            MI(O.LABEL, target=".alternate"),
            MI(O.LI, P.vreg("scratch"), P.immediate(99)),
            MI(O.LI, value, P.immediate(20)),
            MI(O.LABEL, target=".join"),
            MI(O.ADD, P.reg("a0"), value, delta),
            MI(O.JALR, P.reg("zero"), P.reg("ra")),
        ]
        allocated = RegisterAllocator(stream, "greedy").allocate_result()
        assembly = apply_abi_frames(AsmEmitter(allocated.instructions).emit(), allocated)
        assert execute_i32(assembly, tmp_path, argument=argument) == expected
