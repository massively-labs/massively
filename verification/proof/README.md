# What is proved about Massively Core

Massively Core is a mathematical model of the bulk operations underneath the
library. This verification answers a specific engineering question: can that
model reproduce a conventional parallel machine with per-processor control
flow and shared-memory conflicts without changing the program's result?

Within the model described below, the Lean proof says yes.

## The two machines

The source is a finite-control priority-CRCW PRAM. Each logical processor has
its own program counter and typed registers. An instruction can perform local
work, read shared memory, write shared memory, branch, or halt. Reads observe
the state at the start of the clock. If several processors write the same
address, the processor with the smallest identifier wins.

The target is Massively Core, a bulk-synchronous machine built from the same
kind of stages used by a GPU implementation:

| Source-machine operation | Massively Core stage |
| --- | --- |
| Read the pre-clock shared state | Pull |
| Evaluate local control and register updates | Map |
| Select processors that want to write | Proposal compaction |
| Resolve writes to the same address | Deterministic destination reduction |
| Commit the winning writes | Controlled scatter |

The machines and their transition functions are defined independently. The
proof does not obtain equivalence by giving both machines the same evaluator.

## Central results

The verified compiler has two explicit steps:

```text
instruction PRAM -> normalized PRAM -> Massively Core
```

For every word width, finite processor set, finite nonempty shared memory,
finite instruction table, well-typed program, initial configuration, and
finite step count, running the compiled Core program and decoding its state
produces exactly the same result as running the instruction PRAM.

The equality covers all observable machine state:

- every shared-memory cell;
- every processor's program counter, including whether it has halted;
- every processor's complete register value, including recursively nested
  product values used for multi-column data.

One source instruction clock becomes exactly one normalized round and one Core
bulk round. The theorem is therefore not merely an eventual-result simulation:
the two executions agree after every requested finite number of clocks.

The development now also proves a second lowering:

```text
instruction PRAM -> normalized PRAM -> Massively Core -> public API basis
```

The final target has an independent transition semantics consisting of the
contracts of public `map`, `gather`, `copy_where`, stable `sort`,
`unique`, `copy`, and collision-free `scatter`.  In particular, conflicting
writes are implemented by sorting write rows by `(destination, processor id)`,
retaining the first row of each destination run, and scattering the resulting
distinct destinations.  Lean proves that this is exactly the least-processor
priority rule and that the resulting public program agrees after every finite
number of source clocks.

Here “public API basis” is a denotational model of those exported operation
contracts.  The Rust API calls the adjacent-duplicate operation `unique`, not
`unique_by_key`; equality on the destination field gives the required keyed
behavior after sorting.

The proof also checks the difficult parts of this translation separately.
Encoding the program counter into the processor state preserves typed local
expressions and control flow. Priority-write normalization preserves the
winning processor. Core conflict resolution chooses the same winner for every
permutation of the compacted proposal schedule, so the result does not depend
on how proposals happen to be ordered.

## Checked cost properties

The proof keeps semantic correctness separate from a symbolic schedule model.
For the documented general sort/group/scatter schedule, it establishes:

- a factor-one correspondence between instruction clocks and Core rounds;
- no duplication of normalized scalar syntax by the Core compiler;
- at most one write proposal per logical processor per round;
- exactly five kernel launches, four barriers, and zero atomics per round;
- explicit accounting for the state-dependent routing work.

These are exact statements about the formal schedule. They do not predict GPU
elapsed time, memory-system behavior, or the quality of a particular backend.

## Why this matters to a software engineer

The result shows that Massively Core is not limited to straight-line maps or a
single hard-coded algorithm. Its pull, local computation, compaction,
deterministic reduction, and scatter stages are sufficient to encode a
standard synchronous parallel machine with data-dependent control flow and
conflicting shared writes. Compiler transformations within this formal model
can be judged against a precise state-preservation theorem rather than an
informal claim of equivalent behavior.

The result is deliberately relative to a named source model. “Complete” here
means complete for the formal priority-CRCW PRAM above; it does not mean that
every possible concurrent or distributed system has been modeled.

## Boundary of the proof

The theorem covers finite synchronous runs, finite instruction tables, and at
most one shared-memory access per conventional instruction. Scalar operations
are total, typed, and free of hidden global-memory, allocation, launch,
barrier, or atomic effects.

It does not prove that:

- the Rust implementation, generated CubeCL IR, compiler, driver, or GPU
  implements the modeled public-operation contracts;
- Lean finite indices refine Rust `usize`/stored `u32` indices in all error and
  overflow cases, or that concrete buffers satisfy the required layout,
  length, ownership, and non-aliasing obligations;
- the concrete backend sort is stable, concrete `unique` keeps the first row,
  or concrete scatter receives distinct in-bounds destinations (these are now
  explicit refinement obligations rather than assumptions hidden in Core);
- arbitrary asynchronous programs or memory models reduce to this PRAM;
- the symbolic launch and barrier counts imply a wall-clock speedup;
- every public Massively algorithm is functionally correct.

This directory contains the universal mathematical result. The independent
[`../oracle/`](../oracle/) tests provide separate implementation
evidence by comparing public Rust operations with CPU reference behavior; they
are not premises of the Lean theorem. The exact theorem names, assumptions,
and remaining refinement work are recorded in
[`STATUS.md`](STATUS.md).
