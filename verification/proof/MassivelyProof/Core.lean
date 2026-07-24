import MassivelyProof.Algebra

namespace MassivelyProof.MassivelyCore

/-- Core-local context used to compute one bulk pull address.  It is defined
here rather than borrowed from the external PRAM model, so the two machine
semantics can be read independently. -/
abbrev PullContext (items : Nat) (registerType : ValueType) : List ValueType :=
  [.index items, registerType]

/-- Core-local context available to map/select/push expressions after the
old materialized memory value has been pulled. -/
abbrev MapContext (items : Nat) (registerType : ValueType) : List ValueType :=
  [.index items, registerType, .word]

/-- A normalized Massively Core round.

The terms are deliberately named by their bulk role rather than by PRAM
instructions. Pure pull/map expressions observe the old materialized state;
`selectWrite`, destination reduction, and scatter form the controlled bulk
effect committed at the next barrier. -/
structure Program
    {wordWidth : Nat}
    (signature : Signature wordWidth)
    (processors memorySize : Nat)
    (registerType : ValueType) where
  pullAddress :
    Term signature
      (PullContext processors registerType)
      (.index memorySize)
  mapRegisters :
    Term signature
      (MapContext processors registerType)
      registerType
  selectWrite :
    Term signature
      (MapContext processors registerType)
      .boolean
  pushAddress :
    Term signature
      (MapContext processors registerType)
      (.index memorySize)
  pushValue :
    Term signature
      (MapContext processors registerType)
      .word

/-- State observable at a Core materialization/barrier boundary. -/
structure Config
    (wordWidth processors memorySize : Nat)
    (registerType : ValueType) where
  memory : Fin memorySize → Word wordWidth
  registers : Fin processors → registerType.denote wordWidth

namespace Config

/-- Extensional equality of barrier-visible Core states. -/
@[ext]
theorem ext
    {wordWidth processors memorySize : Nat}
    {registerType : ValueType}
    (left right : Config wordWidth processors memorySize registerType)
    (memory : left.memory = right.memory)
    (registers : left.registers = right.registers) : left = right := by
  cases left
  cases right
  cases memory
  cases registers
  rfl

end Config

/-- A compacted write proposal. Values remain in the per-processor mapped
column and are gathered only after a winning owner has been selected. -/
structure Proposal (processors memorySize : Nat) where
  destination : Fin memorySize
  owner : Fin processors
deriving DecidableEq, Repr

namespace Program

/-- Environment for a lazy pull-address expression. -/
def pullEnvironment
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (_program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    Environment wordWidth
      (PullContext processors registerType) :=
  .cons processor (.cons (configuration.registers processor) .nil)

def pulledAddressAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Fin memorySize :=
  program.pullAddress.evaluate
    (program.pullEnvironment configuration processor)

/-- Pull from the old materialized shared-memory column. -/
def pulledValueAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Word wordWidth :=
  configuration.memory (program.pulledAddressAt configuration processor)

/-- Environment evaluated independently for each logical processor row. -/
def mapEnvironment
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    Environment wordWidth
      (MapContext processors registerType) :=
  .cons processor
    (.cons (configuration.registers processor)
      (.cons (program.pulledValueAt configuration processor) .nil))

def mappedRegistersAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : registerType.denote wordWidth :=
  program.mapRegisters.evaluate
    (program.mapEnvironment configuration processor)

def selectedAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Bool :=
  program.selectWrite.evaluate
    (program.mapEnvironment configuration processor)

def pushedAddressAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Fin memorySize :=
  program.pushAddress.evaluate
    (program.mapEnvironment configuration processor)

def pushedValueAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Word wordWidth :=
  program.pushValue.evaluate
    (program.mapEnvironment configuration processor)

/-- Pointwise map/select result before stream compaction. -/
def proposalAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Option (Proposal processors memorySize) :=
  if program.selectedAt configuration processor then
    some {
      destination := program.pushedAddressAt configuration processor
      owner := processor
    }
  else
    none

/-- Stable compact over the logical processor rows. -/
def proposalsFrom
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processorOrder : List (Fin processors)) :
    List (Proposal processors memorySize) :=
  processorOrder.filterMap (program.proposalAt configuration)

/-- Canonical compacted proposal stream. -/
def proposals
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    List (Proposal processors memorySize) :=
  program.proposalsFrom configuration (List.finRange processors)

/-- One destination-filtered owner-reduction action. -/
def foldProposalAt
    {processors memorySize : Nat}
    (destination : Fin memorySize)
    (winner : Winner processors)
    (proposal : Proposal processors memorySize) : Winner processors :=
  if proposal.destination = destination then
    Winner.combine winner (Winner.owner proposal.owner)
  else
    winner

/-- Reduce a supplied proposal schedule for one destination. -/
def winnerAtFrom
    {processors memorySize : Nat}
    (destination : Fin memorySize)
    (scheduled : List (Proposal processors memorySize)) : Winner processors :=
  scheduled.foldl (foldProposalAt destination) (Winner.none processors)

/-- Destination owner after stable compact and schedule-independent reduction. -/
def winnerAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) : Winner processors :=
  winnerAtFrom destination (program.proposals configuration)

/-- Controlled scatter: untouched destinations retain their old value, while
a touched destination gathers the mapped value of its unique winning owner. -/
def scatteredMemoryAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) : Word wordWidth :=
  match (program.winnerAt configuration destination).toOption with
  | none => configuration.memory destination
  | some processor => program.pushedValueAt configuration processor

/-- One bulk-synchronous Core round. Every pull is evaluated against
`configuration`; the returned state becomes visible only at the barrier. -/
def step
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    Config wordWidth processors memorySize registerType where
  memory := program.scatteredMemoryAt configuration
  registers := program.mappedRegistersAt configuration

/-- Finite bulk-synchronous execution. -/
def run
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType) :
    Nat → Config wordWidth processors memorySize registerType →
      Config wordWidth processors memorySize registerType
  | 0, configuration => configuration
  | rounds + 1, configuration =>
      run program rounds (program.step configuration)

/-- Destination-filtered owner updates commute pairwise. -/
theorem foldProposalAt_commutes
    {processors memorySize : Nat}
    (destination : Fin memorySize)
    (winner : Winner processors)
    (left right : Proposal processors memorySize) :
    foldProposalAt destination
        (foldProposalAt destination winner left) right =
      foldProposalAt destination
        (foldProposalAt destination winner right) left := by
  unfold foldProposalAt
  by_cases leftMatches : left.destination = destination
  · by_cases rightMatches : right.destination = destination
    · simp only [leftMatches, rightMatches, if_true]
      calc
        Winner.combine
            (Winner.combine winner (Winner.owner left.owner))
            (Winner.owner right.owner) =
          Winner.combine winner
            (Winner.combine (Winner.owner left.owner)
              (Winner.owner right.owner)) :=
          (Winner.reduction_lawful processors).associative _ _ _
        _ = Winner.combine winner
            (Winner.combine (Winner.owner right.owner)
              (Winner.owner left.owner)) :=
          congrArg (Winner.combine winner)
            ((Winner.reduction_lawful processors).commutative _ _)
        _ = Winner.combine
            (Winner.combine winner (Winner.owner right.owner))
            (Winner.owner left.owner) :=
          ((Winner.reduction_lawful processors).associative _ _ _).symm
    · simp [leftMatches, rightMatches]
  · by_cases rightMatches : right.destination = destination
    · simp [leftMatches, rightMatches]
    · simp [leftMatches, rightMatches]

/-- Any permutation of the compacted proposal stream selects exactly the same
owner at every destination. -/
theorem winnerAtFrom_schedule_independent
    {processors memorySize : Nat}
    (destination : Fin memorySize)
    {logical scheduled : List (Proposal processors memorySize)}
    (permutation : logical.Perm scheduled) :
    winnerAtFrom destination logical = winnerAtFrom destination scheduled := by
  unfold winnerAtFrom
  exact foldl_eq_of_perm (foldProposalAt destination)
    (foldProposalAt_commutes destination) permutation _

end Program

end MassivelyProof.MassivelyCore
