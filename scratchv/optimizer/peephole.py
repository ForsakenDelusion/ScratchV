"""Peephole optimization: eliminates redundant instruction patterns.

Scans basic blocks for common redundant patterns:
  - mul rd, rs, 1   →  mv rd, rs
  - mul rd, rs, 0   →  li rd, 0
  - j L immediately followed by L:  →  delete jump
"""

from __future__ import annotations

from scratchv.ir.types import DataType, OpCode, Instruction, BasicBlock, Program, Value
from scratchv.pass_interface import OptimizationPass


class IRPeepholeOptimizer(OptimizationPass):
    """Eliminate redundant instruction patterns in IR.

    Enable ``preserve_float_semantics`` for typed hard-float code. Integer
    identity rewrites are not valid IEEE transformations: signed zeros, NaNs
    and exception flags are observable. Legacy callers retain the old policy.
    """

    name = "ir-peephole"

    def __init__(self, *, preserve_float_semantics: bool = False):
        self.preserve_float_semantics = preserve_float_semantics
        self._reserved_names: set[str] = set()
        self._name_counter = 0

    def optimize(self, program: Program) -> int:
        """Run peephole optimization.

        Returns number of eliminated instructions.
        """
        self._reserved_names = {v.name for v in program.global_values}
        self._name_counter = 0
        for func in program.functions:
            self._reserved_names.update(v.name for v in [*func.params, *func.returns, *func.locals])
            for block in func.blocks:
                for instr in [*block.phi_nodes, *block.instructions]:
                    self._reserved_names.update(v.name for v in instr.operands)
                    if instr.dest is not None:
                        self._reserved_names.add(instr.dest.name)
        return sum(
            self._optimize_block(block)
            for func in program.functions
            for block in func.blocks
        )

    def _optimize_block(self, block: BasicBlock) -> int:
        changes = 0
        instrs = block.instructions
        i = 0
        while i < len(instrs):
            instr = instrs[i]

            if self.preserve_float_semantics and self._uses_float(instr):
                # Do not delete a definition, invent an i32 zero, or replace
                # Inf * 0 with a constant that suppresses the invalid flag.
                i += 1
                continue

            # ADD(value, 0) still defines a distinct IR Value. Retain it until
            # a whole-function use-replacement pass can safely remove that
            # definition; deleting it as a machine no-op leaves dangling uses.

            # Pattern 2: mul rd, rs, 1 → mv rd, rs (use add rd, rs, x0)
            if self._is_mul_one(instr):
                dest = instr.dest
                src = instr.operands[0]
                instrs[i] = Instruction(
                    opcode=OpCode.ADD,
                    dest=dest,
                    operands=[src, self._make_zero_operand(instr)],
                )
                changes += 1
                i += 1
                continue

            # Pattern 3: mul rd, rs, 0 → li rd, 0 (load_const 0)
            if self._is_mul_zero(instr):
                dest = instr.dest
                instrs[i] = Instruction(
                    opcode=OpCode.LOAD_CONST,
                    dest=dest,
                    attrs={"value": 0},
                )
                changes += 1
                i += 1
                continue

            # Pattern 4: j L followed immediately by L:
            if self._is_jump_to_next(instrs, i):
                instrs.pop(i)
                changes += 1
                continue

            i += 1

        return changes

    @staticmethod
    def _uses_float(instr: Instruction) -> bool:
        values = [*instr.operands, *([instr.dest] if instr.dest else [])]
        return any(value.dtype in {DataType.FLOAT32, DataType.FLOAT64} for value in values)

    def _is_mul_one(self, instr: Instruction) -> bool:
        if instr.opcode != OpCode.MUL:
            return False
        if len(instr.operands) < 2:
            return False
        rhs = instr.operands[1]
        return rhs.is_constant and rhs.const_value == 1

    def _is_mul_zero(self, instr: Instruction) -> bool:
        if instr.opcode != OpCode.MUL:
            return False
        if len(instr.operands) < 2:
            return False
        rhs = instr.operands[1]
        return rhs.is_constant and rhs.const_value == 0

    def _is_jump_to_next(self, instrs: list[Instruction], i: int) -> bool:
        """Check for: br L  followed by label L."""
        if i + 1 >= len(instrs):
            return False
        instr = instrs[i]
        if instr.opcode != OpCode.BR:
            return False
        next_instr = instrs[i + 1]
        return (next_instr.opcode == OpCode.LABEL
                and next_instr.target == instr.target)

    def _make_zero_operand(self, instr: Instruction):
        """Create a zero constant value."""
        while True:
            self._name_counter += 1
            name = f"__scratchv_peephole_zero_{self._name_counter}"
            if name not in self._reserved_names:
                self._reserved_names.add(name)
                return Value(name=name, dtype=instr.operands[0].dtype,
                             is_constant=True, const_value=0)
