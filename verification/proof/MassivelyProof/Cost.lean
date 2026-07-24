import MassivelyProof.Compiler

namespace MassivelyProof

namespace ValueType

/-- Number of machine words occupied by a logical row. Recursive products make
the measure independent of any fixed public tuple arity. -/
def words : ValueType → Nat
  | .boolean => 1
  | .word => 1
  | .index _ => 1
  | .product left right => left.words + right.words

end ValueType

namespace Term

/-- Sharing-aware scalar syntax size. `let1` counts its input once. -/
def nodeCount
    {wordWidth : Nat}
    {signature : Signature wordWidth}
    {context : List ValueType} :
    {output : ValueType} → Term signature context output → Nat
  | _, .read _ => 1
  | _, .literal _ => 1
  | _, .constant _ => 1
  | _, .pair left right => left.nodeCount + right.nodeCount + 1
  | _, .first productTerm => productTerm.nodeCount + 1
  | _, .second productTerm => productTerm.nodeCount + 1
  | _, .apply _ argument => argument.nodeCount + 1
  | _, .ite condition whenTrue whenFalse =>
      condition.nodeCount + whenTrue.nodeCount + whenFalse.nodeCount + 1
  | _, .caseIndex index branches =>
      (List.finRange _).foldl
        (fun total branch => total + (branches branch).nodeCount)
        (index.nodeCount + 1)
  | _, .let1 input body => input.nodeCount + body.nodeCount + 1

/-- Abstract dependence depth of local scalar syntax. Pair branches may run in
parallel; explicit sharing and primitive application are sequential edges. -/
def depth
    {wordWidth : Nat}
    {signature : Signature wordWidth}
    {context : List ValueType} :
    {output : ValueType} → Term signature context output → Nat
  | _, .read _ => 1
  | _, .literal _ => 1
  | _, .constant _ => 1
  | _, .pair left right => Nat.max left.depth right.depth + 1
  | _, .first productTerm => productTerm.depth + 1
  | _, .second productTerm => productTerm.depth + 1
  | _, .apply _ argument => argument.depth + 1
  | _, .ite condition whenTrue whenFalse =>
      Nat.max condition.depth (Nat.max whenTrue.depth whenFalse.depth) + 1
  | _, .caseIndex index branches =>
      (List.finRange _).foldl
        (fun maximum branch => Nat.max maximum (branches branch).depth)
        index.depth + 1
  | _, .let1 input body => input.depth + body.depth + 1

end Term

namespace PriorityCRCW.Program

/-- Source syntax size; this is a compile-time measure, not runtime work. -/
def nodeCount
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType) : Nat :=
  program.readAddress.nodeCount +
  program.nextRegisters.nodeCount +
  program.writeEnabled.nodeCount +
  program.writeAddress.nodeCount +
  program.writeValue.nodeCount

end PriorityCRCW.Program

namespace MassivelyCore.Program

/-- Target syntax size in the same sharing-aware metric. -/
def nodeCount
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType) : Nat :=
  program.pullAddress.nodeCount +
  program.mapRegisters.nodeCount +
  program.selectWrite.nodeCount +
  program.pushAddress.nodeCount +
  program.pushValue.nodeCount

/-- Per-processor scalar work before proposal routing. -/
def localWork
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType) : Nat :=
  program.nodeCount

/-- Critical path through the five independent output expressions after the
shared pull environment is available. -/
def localDepth
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType) : Nat :=
  Nat.max program.pullAddress.depth
    (Nat.max program.mapRegisters.depth
      (Nat.max program.selectWrite.depth
        (Nat.max program.pushAddress.depth program.pushValue.depth)))

/-- Number of compacted write proposals in this state. -/
def proposalCount
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) : Nat :=
  (program.proposals configuration).length

/-- Stable compaction never creates a proposal. -/
theorem proposalCount_le_processors
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    program.proposalCount configuration ≤ processors := by
  unfold proposalCount MassivelyCore.Program.proposals
    MassivelyCore.Program.proposalsFrom
  simpa using List.length_filterMap_le
    (program.proposalAt configuration) (List.finRange processors)

end MassivelyCore.Program

/-- Abstract balanced routing/reduction depth. -/
def reductionDepth (items : Nat) : Nat :=
  if items = 0 then 0 else Nat.log2 items + 1

/-- Symbolic resource vector for one abstract GPU plan. Counters remain
unweighted: no elapsed-time claim is hidden in this record. -/
structure Cost where
  logicalItems : Nat := 0
  scalarWork : Nat := 0
  span : Nat := 0
  globalLoads : Nat := 0
  globalStores : Nat := 0
  atomics : Nat := 0
  barriers : Nat := 0
  launches : Nat := 0
  temporaryWords : Nat := 0
  allocationWords : Nat := 0
  materializations : Nat := 0
deriving DecidableEq, Repr

namespace Cost

def zero : Cost := {}

def sequential (first second : Cost) : Cost where
  logicalItems := first.logicalItems + second.logicalItems
  scalarWork := first.scalarWork + second.scalarWork
  span := first.span + second.span
  globalLoads := first.globalLoads + second.globalLoads
  globalStores := first.globalStores + second.globalStores
  atomics := first.atomics + second.atomics
  barriers := first.barriers + second.barriers
  launches := first.launches + second.launches
  temporaryWords := Nat.max first.temporaryWords second.temporaryWords
  allocationWords := first.allocationWords + second.allocationWords
  materializations := first.materializations + second.materializations

end Cost

namespace MassivelyCore.Program

/-- General deterministic sort/group/scatter resource contract for one Core
round. It exposes the proposal-dependent routing term rather than pretending
that arbitrary priority conflicts are free atomics.

The constants describe the normalized abstract schedule, not a calibrated GPU:
pull/map, compaction, route/reduce, scatter, and barrier-visible register
materialization. -/
def generalRoundCost
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) : Cost :=
  let proposals := program.proposalCount configuration
  let rowWords := registerType.words
  {
    logicalItems := processors + proposals
    scalarWork :=
      processors * (program.localWork + 1) +
      proposals * (reductionDepth proposals + 2)
    span := program.localDepth + reductionDepth proposals + 4
    globalLoads := processors * (rowWords + 1)
    globalStores := processors * rowWords + proposals
    atomics := 0
    barriers := 4
    launches := 5
    temporaryWords := processors * (rowWords + 3) + proposals * 2
    allocationWords := processors * rowWords + proposals * 3
    materializations := 3
  }

@[simp]
theorem generalRoundCost_atomics
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.generalRoundCost configuration).atomics = 0 := rfl

@[simp]
theorem generalRoundCost_materializations
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.generalRoundCost configuration).materializations = 3 := rfl

@[simp]
theorem generalRoundCost_barriers
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.generalRoundCost configuration).barriers = 4 := rfl

@[simp]
theorem generalRoundCost_launches
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.generalRoundCost configuration).launches = 5 := rfl

/-- Exact work equation, retaining the explicit conflict-routing term. -/
theorem generalRoundCost_scalarWork
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.generalRoundCost configuration).scalarWork =
      processors * (program.localWork + 1) +
      program.proposalCount configuration *
        (reductionDepth (program.proposalCount configuration) + 2) := rfl

/-- Symbolic cost of a finite Core execution, following the same state order
as `run`. -/
def runCost
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType) :
    Nat → MassivelyCore.Config wordWidth processors memorySize registerType → Cost
  | 0, _ => Cost.zero
  | rounds + 1, configuration =>
      Cost.sequential (program.generalRoundCost configuration)
        (runCost program rounds (program.step configuration))

@[simp]
theorem runCost_atomics
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (rounds : Nat)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.runCost rounds configuration).atomics = 0 := by
  induction rounds generalizing configuration with
  | zero => rfl
  | succ rounds induction =>
      simp [runCost, Cost.sequential, induction]

@[simp]
theorem runCost_barriers
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (rounds : Nat)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.runCost rounds configuration).barriers = rounds * 4 := by
  induction rounds generalizing configuration with
  | zero => rfl
  | succ rounds induction =>
      simp [runCost, Cost.sequential, induction, Nat.succ_mul, Nat.add_comm]

@[simp]
theorem runCost_launches
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (rounds : Nat)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.runCost rounds configuration).launches = rounds * 5 := by
  induction rounds generalizing configuration with
  | zero => rfl
  | succ rounds induction =>
      simp [runCost, Cost.sequential, induction, Nat.succ_mul, Nat.add_comm]

@[simp]
theorem runCost_materializations
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : MassivelyCore.Program signature processors memorySize registerType)
    (rounds : Nat)
    (configuration : MassivelyCore.Config
      wordWidth processors memorySize registerType) :
    (program.runCost rounds configuration).materializations = rounds * 3 := by
  induction rounds generalizing configuration with
  | zero => rfl
  | succ rounds induction =>
      simp [runCost, Cost.sequential, induction, Nat.succ_mul, Nat.add_comm]

end MassivelyCore.Program

namespace Compilation

/-- Compilation copies each typed scalar term once; there is no syntax or
compile-time arity blow-up. -/
theorem compile_nodeCount
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType) :
    (compile program).nodeCount = program.nodeCount := rfl

/-- A compiled priority round emits at most one proposal per logical
processor. -/
theorem compiled_proposalCount_le_processors
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config
      wordWidth processors memorySize registerType) :
    (compile program).proposalCount (encode configuration) ≤ processors :=
  MassivelyCore.Program.proposalCount_le_processors _ _

end Compilation

end MassivelyProof
