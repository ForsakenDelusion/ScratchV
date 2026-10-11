"""Constant folding optimization pass.

Evaluates arithmetic operations with constant operands at compile time,
replacing them with load_const instructions.
"""

from __future__ import annotations

from scratchv.ir.types import (
    DataType, OpCode, Instruction, BasicBlock, Function, Program,
)
from scratchv.pass_interface import OptimizationPass


class ConstantFolder(OptimizationPass):
    """Fold constant expressions in an IR Program.

    ``preserve_float_semantics=True`` is required for real typed hard-float
    targets: Python arithmetic does not model binary32 operand quantization,
    target rounding modes, or fflags. The default retains the legacy stub
    pipeline's historical behavior; CompilerDriver enables the safe policy
    for extended instruction selection, including explicitly selected passes.
    """

    name = "constant-folding"

    def __init__(self, *, preserve_float_semantics: bool = False):
        self.preserve_float_semantics = preserve_float_semantics

    def optimize(self, program: Program) -> int:
        """Run constant folding on all functions. Returns number of folds."""
        return sum(self._fold_function(func) for func in program.functions)

    def _fold_function(self, func: Function) -> int:
        return sum(self._fold_block(block) for block in func.blocks)

    def _fold_block(self, block: BasicBlock) -> int:
        changes = 0
        new_instrs: list[Instruction] = []
        for instr in block.instructions:
            folded = self._try_fold(instr)
            if folded is not None:
                new_instrs.append(folded)
                changes += 1
            else:
                new_instrs.append(instr)
        block.instructions = new_instrs
        return changes

    def _try_fold(self, instr: Instruction) -> Instruction | None:
        """Try to fold an instruction. Returns a replacement or None."""
        if instr.opcode not in (
                OpCode.ADD, OpCode.SUB,
                OpCode.MUL, OpCode.DIV, OpCode.REM,
                OpCode.MIN, OpCode.MAX, OpCode.ABS):
            return None
        expected_operands = 1 if instr.opcode is OpCode.ABS else 2
        if len(instr.operands) != expected_operands:
            return None

        if any(not operand.is_constant for operand in instr.operands):
            return None
        if any(operand.const_value is None for operand in instr.operands):
            return None

        lhs = instr.operands[0]
        dtype = instr.dest.dtype if instr.dest is not None else lhs.dtype
        if self.preserve_float_semantics and any(
            operand.dtype in {DataType.FLOAT32, DataType.FLOAT64}
            for operand in [*instr.operands, *([instr.dest] if instr.dest else [])]
        ):
            # Keep the actual target operation. Even an exactly representable
            # host result does not prove that operand conversion or fflags agree.
            return None
        if dtype is DataType.INT32:
            values = [operand.const_value for operand in instr.operands]
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                return None
            if instr.opcode is OpCode.ABS:
                result = self._wrap_i32(abs(values[0]))
            else:
                result = self._compute_i32(
                    instr.opcode,
                    values[0],
                    values[1],
                )
        else:
            if instr.opcode in {
                OpCode.REM, OpCode.MIN, OpCode.MAX, OpCode.ABS,
            }:
                return None
            rhs = instr.operands[1]
            a, b = float(lhs.const_value), float(rhs.const_value)
            result = self._compute_float(instr.opcode, a, b)
        if result is None:
            return None

        dest = instr.dest
        if dest is not None:
            dest.is_constant = True
            dest.const_value = result
            dest.bit_pattern = None

        return Instruction(
            opcode=OpCode.LOAD_CONST,
            dest=dest,
            attrs={"value": result},
        )

    @staticmethod
    def _compute_float(
        opcode: OpCode, a: float, b: float,
    ) -> float | None:
        mapping = {
            OpCode.ADD: a + b,
            OpCode.SUB: a - b,
            OpCode.MUL: a * b,
            OpCode.DIV: a / b if b != 0 else None,
        }
        return mapping.get(opcode)

    @staticmethod
    def _compute_i32(opcode: OpCode, a: int, b: int) -> int | None:
        """Evaluate an i32 operation with RV32 two's-complement semantics."""
        if opcode in (OpCode.DIV, OpCode.REM):
            if b == 0:
                return None
            quotient = abs(a) // abs(b)
            if (a < 0) != (b < 0):
                quotient = -quotient
            result = quotient if opcode is OpCode.DIV else a - quotient * b
        else:
            mapping = {
                OpCode.ADD: a + b,
                OpCode.SUB: a - b,
                OpCode.MUL: a * b,
                OpCode.MIN: min(a, b),
                OpCode.MAX: max(a, b),
            }
            result = mapping.get(opcode)
            if result is None:
                return None

        # RV32 integer arithmetic wraps modulo 2**32.
        return ConstantFolder._wrap_i32(result)

    @staticmethod
    def _wrap_i32(value: int) -> int:
        value &= 0xFFFF_FFFF
        return value if value < 0x8000_0000 else value - 0x1_0000_0000
