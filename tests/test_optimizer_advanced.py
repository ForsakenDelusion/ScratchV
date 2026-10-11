"""Tests for advanced optimizer passes: peephole, muladd_fusion, LICM."""

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode
from scratchv.optimizer.peephole import IRPeepholeOptimizer
from scratchv.optimizer.muladd_fusion import MulAddFusion
from scratchv.optimizer.licm import LICM
from scratchv.analysis.ir_verifier import verify_ir


class TestIRPeepholeOptimizer:
    def test_keep_add_zero_result_definition(self):
        builder = IRBuilder()
        function = builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        function.params.append(a)
        c = builder.make_value(name="c", dtype=DataType.INT32)
        # ADD defines c even though its value equals a.
        zero = builder.make_value(name="_z", dtype=DataType.INT32,
                                  is_constant=True, const_value=0)
        builder._emit(OpCode.ADD, c, [a, zero])
        builder.ret(c)

        opt = IRPeepholeOptimizer()
        assert opt.optimize(builder.program) == 0
        block = builder.program.functions[0].blocks[0]
        assert block.instructions[0].opcode is OpCode.ADD
        assert block.instructions[-1].operands[0] is c
        assert verify_ir(builder.program) == (True, [])

    def test_eliminate_mul_one(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        c = builder.make_value(name="c")
        one = builder.make_value(name="_o", is_constant=True, const_value=1)
        builder._emit(OpCode.MUL, c, [a, one])
        builder.ret(c)

        opt = IRPeepholeOptimizer()
        count = opt.optimize(builder.program)
        assert count >= 1
        # MUL with 1 should be replaced
        block = builder.program.functions[0].blocks[0]
        has_mul = any(i.opcode == OpCode.MUL for i in block.instructions)
        assert not has_mul

    def test_eliminate_mul_zero(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        c = builder.make_value(name="c")
        zero = builder.make_value(name="_z", is_constant=True, const_value=0)
        builder._emit(OpCode.MUL, c, [a, zero])
        builder.ret(c)

        opt = IRPeepholeOptimizer()
        opt.optimize(builder.program)
        block = builder.program.functions[0].blocks[0]
        mul_instrs = [i for i in block.instructions if i.opcode == OpCode.MUL]
        assert len(mul_instrs) == 0
        # Should be replaced with load_const 0
        lc_instrs = [i for i in block.instructions
                     if i.opcode == OpCode.LOAD_CONST]
        assert any(i.attrs.get("value") == 0 for i in lc_instrs)


class TestMulAddFusion:
    def test_keep_mul_add_without_valid_fused_opcode(self):
        builder = IRBuilder()
        function = builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)
        acc = builder.make_value(name="acc", dtype=DataType.INT32)
        function.params.extend([a, b, acc])

        tmp = builder.mul(a, b)
        result = builder.add(tmp, acc)
        builder.ret(result)

        opt = MulAddFusion()
        count = opt.optimize(builder.program)
        assert count == 0

        block = builder.program.functions[0].blocks[0]
        # A valid MUL and binary ADD stay separate; no multiplier is dropped.
        adds = [i for i in block.instructions if i.opcode == OpCode.ADD]
        muls = [i for i in block.instructions if i.opcode == OpCode.MUL]
        assert len(adds) == 1
        assert len(muls) == 1
        assert adds[0].operands == [tmp, acc]
        assert verify_ir(builder.program) == (True, [])

    def test_no_fuse_without_mul(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")
        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.add(a, b)
        builder.ret(c)

        opt = MulAddFusion()
        count = opt.optimize(builder.program)
        assert count == 0


class TestLICM:
    def test_hoist_invariant(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        a = builder.make_value(name="a", dtype=DataType.INT32)
        b = builder.make_value(name="b", dtype=DataType.INT32)

        # Loop with invariant mul inside
        iv = builder.for_loop(0, 10)  # FOR
        # This mul depends on a, b defined outside the loop -> invariant
        c = builder.mul(a, b)
        builder.add(c, iv)  # this depends on iv → variant, keep
        builder.endfor()
        builder.ret()

        opt = LICM()
        count = opt.optimize(builder.program)
        assert count == 1  # mul should be hoisted

        block = builder.program.functions[0].blocks[0]
        # Find the FOR and check if mul is before it
        for_idx = next(i for i, instr in enumerate(block.instructions)
                       if instr.opcode == OpCode.FOR)
        mul_idx = next(i for i, instr in enumerate(block.instructions)
                       if instr.opcode == OpCode.MUL)
        assert mul_idx < for_idx, "MUL should be hoisted before FOR"

    def test_no_hoist_variant(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        iv = builder.for_loop(0, 10)
        # This add depends on iv (loop variant) → should not be hoisted
        builder.add(iv, builder.load_const(1, DataType.INT32))
        builder.endfor()
        builder.ret()

        opt = LICM()
        opt.optimize(builder.program)
        # The load_const IS invariant and gets hoisted.
        # But the add depending on iv stays in the loop.
        # Verify the add remains after the FOR.
        block = builder.program.functions[0].blocks[0]
        for_idx = next(i for i, instr in enumerate(block.instructions)
                       if instr.opcode == OpCode.FOR)
        add_instrs = [i for i in block.instructions if i.opcode == OpCode.ADD]
        # The ADD (variant) should still be inside the loop (after FOR)
        assert all(block.instructions.index(i) > for_idx for i in add_instrs)
