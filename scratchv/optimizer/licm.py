"""Loop Invariant Code Motion (LICM).

Moves instructions that produce the same result every iteration
from inside a loop to before the loop header.

This pass works on the IR FOR/ENDFOR structure. An instruction
is loop-invariant if all its operands are:
  1. Constants, or
  2. Defined outside the loop, or
  3. Already hoisted invariants.
"""

from __future__ import annotations

from scratchv.ir.types import (
    DataType, OpCode, Instruction, BasicBlock, Function, Program,
)
from scratchv.pass_interface import OptimizationPass


class LICM(OptimizationPass):
    """Hoist loop-invariant code out of loops."""

    name = "licm"

    def __init__(self, *, preserve_float_semantics: bool = False):
        self.preserve_float_semantics = preserve_float_semantics

    def optimize(self, program: Program) -> int:
        """Run LICM on all functions.

        Returns number of hoisted instructions.
        """
        return sum(self._process_function(func) for func in program.functions)

    def _process_function(self, func: Function) -> int:
        return sum(self._process_block(block) for block in func.blocks)

    def _process_block(self, block: BasicBlock) -> int:
        """Find FOR/ENDFOR pairs in a block and hoist invariants."""
        changes = 0
        instrs = block.instructions
        i = 0
        while i < len(instrs):
            if instrs[i].opcode != OpCode.FOR:
                i += 1
                continue

            # Find matching ENDFOR
            loop_start = i
            loop_end = self._find_matching_endfor(instrs, loop_start)
            if loop_end is None:
                i += 1
                continue

            # Collect loop-variant names (loop variable induction var)
            dest = instrs[loop_start].dest
            iv_name = dest.name if dest else ""
            variant_names = {iv_name}

            # Find instructions defined within the loop (excluding FOR itself)
            loop_defs = set()
            for j in range(loop_start + 1, loop_end):
                instr = instrs[j]
                if instr.dest:
                    loop_defs.add(instr.dest.name)

            # Scan for invariant instructions
            hoisted = []
            j = loop_start + 1
            while j < loop_end:
                instr = instrs[j]
                if self._is_invariant(instr, variant_names, loop_defs):
                    hoisted.append((j, instr))
                    j += 1
                else:
                    # Add this instruction's dest to variant set
                    if instr.dest:
                        variant_names.add(instr.dest.name)
                    j += 1

            # Remove all candidates before inserting any: insertion would
            # otherwise shift the indices of candidates still in the loop.
            for idx, _ in reversed(hoisted):
                instrs.pop(idx)
            instrs[loop_start:loop_start] = [instr for _, instr in hoisted]
            # The total length and ENDFOR's index are unchanged. Keep the
            # original order and do not skip a following loop.
            changes += len(hoisted)

            i = loop_end + 1

        return changes

    def _find_matching_endfor(self, instrs: list[Instruction], start: int):
        """Find matching ENDFOR for a FOR at given index."""
        depth = 0
        for i in range(start, len(instrs)):
            if instrs[i].opcode == OpCode.FOR:
                depth += 1
            elif instrs[i].opcode == OpCode.ENDFOR:
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _is_invariant(self, instr: Instruction, variant_names: set[str],
                      loop_defs: set[str]) -> bool:
        """Check if an instruction is loop-invariant."""
        # Only pure numeric operations and literal loads are candidates.
        # An invariant pointer does not prove invariant memory contents, or
        # that a LOAD/ALLOCA is safe to speculate out of a zero-trip loop.
        if not (instr.opcode.is_arith() or instr.opcode is OpCode.LOAD_CONST):
            return False
        if self.preserve_float_semantics and instr.opcode.is_arith() and any(
            value.dtype in (DataType.FLOAT32, DataType.FLOAT64)
            for value in [*instr.operands, *([instr.dest] if instr.dest else [])]
        ):
            # An unused result may still set fflags. Moving the operation
            # before the loop changes whether it executes at all.
            return False
        # An instruction is invariant if all its operands are:
        # - constants, or
        # - defined outside the loop (not in loop_defs and not variant)
        if not instr.operands:
            return True
        for op in instr.operands:
            if op.is_constant:
                continue
            if op.name in variant_names:
                return False
            if op.name in loop_defs:
                return False
        return True
