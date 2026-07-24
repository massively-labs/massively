# Formal proof status

This ledger states exactly what the Lean development proves and what remains
outside its theorem boundary.

## Proved central theorems

`InstructionNormalization.compileToCore_run_decode_correct` proves that, for
every word width, finite processor count, finite nonempty shared memory,
finite instruction table, well-typed instruction program, initial
configuration, and finite step count, executing the compiled Massively Core
program and decoding its barrier state gives exactly the instruction-level
priority-CRCW PRAM execution.

Equality includes shared memory, every processor's program counter (including
the halted state), and every processor's recursive-product register value.
The same `steps` argument occurs on both sides: one source instruction clock
is represented by exactly one normalized PRAM round and one Core bulk round.

`InstructionNormalization.compileToPublicAPI_run_decode_correct` strengthens
the target side to an independent denotational model of a fixed public
Massively API program.  For every finite instruction-machine run, the compiled
program built from `map`, `gather`, `copy_where`, stable `sort`, `unique`,
`copy`, and collision-free `scatter` decodes to exactly the source execution.
The public program is fixed by the source syntax and is not specialized to an
input configuration or requested step count.

The checked proof chain is:

```text
InstructionPRAM.Program
  -- InstructionNormalization.normalize --> PriorityCRCW.Program
  -- Compilation.compile                --> MassivelyCore.Program
  -- PublicAPICompilation.compile       --> PublicAPI.Program
```

### Instruction normalization

- `InstructionPRAM.Instruction` independently defines local, read, write,
  branch, and halt instructions with per-processor control flow.
- `InstructionNormalization.evaluate_liftLocalRead`,
  `evaluate_liftLocalTransition`, and `evaluate_liftReadTransition` prove the
  typed substitution used to encode user registers behind a program counter.
- `normalized_readAddressAt`, `normalized_nextRegistersAt`,
  `normalized_writeEnabledAt`, `normalized_writeAddressAt`, and
  `normalized_writeValueAt` preserve every local instruction observation.
- `normalized_priorityOwnerAt` preserves priority write conflicts.
- `normalize_step_correct`, `normalize_run_correct`, and
  `normalize_run_decode_correct` prove one-step and every-finite-run
  normalization correctness.

### Normalized PRAM to Core

- `Compilation.winner_compile_correct`: Core conflict resolution chooses the
  same least processor identifier as priority-CRCW.
- `Compilation.scheduled_winner_compile_correct`: every permutation of the
  compacted proposal schedule chooses the same owner.
- `Compilation.memory_compile_correct` and `compile_step_correct`: controlled
  scatter and one complete Core round preserve the normalized PRAM state.
- `Compilation.compile_run_decode_correct`: the result holds for every finite
  normalized run.
- `InstructionNormalization.compileToCore_run_decode_correct`: composition of
  both layers, without treating either transition function as the other one's
  definition.

### Core to public API basis

- `PublicAPI.Program.step` independently specifies the routing pipeline at
  public API boundaries.  It does not call `MassivelyCore.Program.winnerAt`.
- `winningWriteRows_compile` proves that stable ordering by
  `(destination, owner)` followed by first-per-destination selection returns
  the least enabled owner for each destination.
- `firstOwnerAt_eq_winner` proves that this public first-row observation equals
  Core's commutative priority reduction.
- `PublicAPICompilation.compile_step_correct` and `compile_run_correct` prove
  one round and every finite run.
- `InstructionNormalization.compileToPublicAPI_run_decode_correct` composes the
  instruction normalizer, Core compiler, and public-API compiler.

The Lean names `Basis.uniqueByDestination` and `Program.winningWriteRows` model
the contract implemented by Rust's public `vector::unique`: after a stable
`vector::sort` on `(destination, owner)`, consecutive equal destinations retain
their first row.  They do not claim a separately exported Rust function named
`unique_by_key`.

`Example.instructionProgram_two_steps` is a concrete read-then-write
instruction program compiled through both layers. The earlier collision
example remains as a direct test of priority conflict resolution.

## Checked cost facts

- `normalizedRoundCount_factor_one`: instruction clocks and normalized/Core
  bulk rounds have a factor-one count.
- `Compilation.compile_nodeCount` and `compileToCore_nodeCount`: the second
  compiler does not duplicate normalized scalar syntax.
- `compiled_proposalCount_le_processors`: at most one proposal is emitted per
  logical processor per round.
- `MassivelyCore.Program.runCost_*` and
  `InstructionNormalization.compileToCore_runCost_*`: under the documented
  general sort/group/scatter schedule, `s` steps have exactly `5s` launches,
  `4s` barriers, and zero atomics. Scalar work retains its explicit
  state-dependent routing term.

These are symbolic schedule counts, not calibrated GPU elapsed-time claims.
There is not yet a machine-independent asymptotic work/span refinement theorem
connecting this schedule to physical GPU execution.

## Current scope

- finite synchronous executions and finite instruction tables;
- a separate program counter for every processor;
- one local, read, write, branch, or halt instruction per processor clock;
- at most one shared-memory access per conventional instruction;
- pre-step snapshot reads and priority-CRCW writes, where the least processor
  identifier wins;
- finite words, bounded indices, booleans, and recursive product registers;
- typed pure scalar terms with constants, product construction/projection,
  conditionals, finite-index cases, explicit sharing, and signature-supplied
  pure primitives;
- no hidden global-memory, atomic, barrier, allocation, or launch effect
  inside scalar terms.

The explicit idle address carried by an instruction program witnesses that
shared memory is nonempty. Processor and instruction counts may otherwise be
zero.

## Deliberate boundary

This establishes semantic completeness of the mathematical Core calculus
relative to the formal conventional priority-CRCW PRAM model. It does not
formalize the informal set called "all parallel programs" independently of
that external model.

The new public-API theorem removes the earlier semantic gap between the Core
winner fold and a composition of public sequence-algorithm contracts.  It is
still not a mechanized refinement proof of the Rust implementation, CubeCL IR,
a device compiler, or GPU hardware.  In particular, the following remain
outside Lean:

- Rust `MIter`/`MVec` row layout, lifetimes, aliasing, allocation, and `Result`
  behavior;
- the correspondence between Lean finite indices and Rust `usize`/stored
  `u32` indices, including size and bounds failures;
- that the concrete sort kernel is stable for every backend and that concrete
  `unique` retains the first row of every equal adjacent run;
- that final `scatter` is invoked only after destination uniqueness, and its
  concrete kernel implements the collision-free contract;
- CubeCL lowering, synchronization, device memory, compiler, driver, and
  hardware refinement;
- a host-side loop or generated orchestration artifact that materializes the
  theorem's fixed public call sequence with checked error propagation.

These are now implementation-refinement obligations for named public
operations rather than an uninstantiated claim about the abstract Core.

Docker pins and checks the Lean artifact; it does not turn implementation
conformance tests into formal premises.
