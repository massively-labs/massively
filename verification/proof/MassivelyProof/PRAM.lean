import MassivelyProof.Algebra

namespace MassivelyProof.PriorityCRCW

/-- Context used to compute the shared-memory read address for one processor. -/
abbrev ReadContext (processors : Nat) (registerType : ValueType) :
    List ValueType :=
  [.index processors, registerType]

/-- Context available after the old shared-memory snapshot has been read. -/
abbrev TransitionContext (processors : Nat) (registerType : ValueType) :
    List ValueType :=
  [.index processors, registerType, .word]

/-- A finite, typed priority-CRCW PRAM round.

Every processor performs one shared read and proposes at most one shared
write. A program counter and additional local registers can be represented by
`registerType`; recursive products avoid any fixed register arity. -/
structure Program
    {wordWidth : Nat}
    (signature : Signature wordWidth)
    (processors memorySize : Nat)
    (registerType : ValueType) where
  readAddress :
    Term signature (ReadContext processors registerType) (.index memorySize)
  nextRegisters :
    Term signature (TransitionContext processors registerType) registerType
  writeEnabled :
    Term signature (TransitionContext processors registerType) .boolean
  writeAddress :
    Term signature (TransitionContext processors registerType) (.index memorySize)
  writeValue :
    Term signature (TransitionContext processors registerType) .word

/-- Barrier-visible PRAM state. -/
structure Config
    (wordWidth processors memorySize : Nat)
    (registerType : ValueType) where
  memory : Fin memorySize → Word wordWidth
  registers : Fin processors → registerType.denote wordWidth

namespace Config

/-- Extensional equality of source PRAM states. -/
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

namespace Program

/-- Environment for the address calculation of one processor. -/
def readEnvironment
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (_program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    Environment wordWidth (ReadContext processors registerType) :=
  .cons processor (.cons (configuration.registers processor) .nil)

/-- Address read by a processor in the current source round. -/
def readAddressAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Fin memorySize :=
  program.readAddress.evaluate
    (program.readEnvironment configuration processor)

/-- Every source read observes the pre-round shared-memory snapshot. -/
def readValueAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Word wordWidth :=
  configuration.memory (program.readAddressAt configuration processor)

/-- Environment for local transition and write-proposal terms. -/
def transitionEnvironment
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    Environment wordWidth (TransitionContext processors registerType) :=
  .cons processor
    (.cons (configuration.registers processor)
      (.cons (program.readValueAt configuration processor) .nil))

def nextRegistersAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : registerType.denote wordWidth :=
  program.nextRegisters.evaluate
    (program.transitionEnvironment configuration processor)

def writeEnabledAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Bool :=
  program.writeEnabled.evaluate
    (program.transitionEnvironment configuration processor)

def writeAddressAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Fin memorySize :=
  program.writeAddress.evaluate
    (program.transitionEnvironment configuration processor)

def writeValueAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Word wordWidth :=
  program.writeValue.evaluate
    (program.transitionEnvironment configuration processor)

/-- One source fold action for a fixed shared-memory destination.

This definition works directly over logical processors and does not construct
the target's compacted proposal stream. -/
def foldProcessorAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize)
    (winner : Winner processors)
    (processor : Fin processors) : Winner processors :=
  if program.writeEnabledAt configuration processor &&
      program.writeAddressAt configuration processor == destination then
    Winner.combine winner (Winner.owner processor)
  else
    winner

/-- Priority owner selected by the independently defined source machine. -/
def priorityOwnerAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) : Winner processors :=
  (List.finRange processors).foldl
    (program.foldProcessorAt configuration destination)
    (Winner.none processors)

/-- Pointwise priority-CRCW shared-memory update. -/
def nextMemoryAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) : Word wordWidth :=
  match (program.priorityOwnerAt configuration destination).toOption with
  | none => configuration.memory destination
  | some processor => program.writeValueAt configuration processor

/-- One synchronous source-machine round. -/
def step
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    Config wordWidth processors memorySize registerType where
  memory := program.nextMemoryAt configuration
  registers := program.nextRegistersAt configuration

/-- Finite execution avoids assuming termination of an arbitrary parallel
program. -/
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

end Program

end MassivelyProof.PriorityCRCW
