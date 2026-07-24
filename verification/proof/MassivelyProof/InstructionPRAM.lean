import MassivelyProof.Algebra

namespace MassivelyProof.InstructionPRAM

/-- Pure local context of one conventional PRAM processor instruction. -/
abbrev LocalContext (processors : Nat) (registerType : ValueType) :
    List ValueType :=
  [.index processors, registerType]

/-- Context of a read continuation after the old shared-memory snapshot has
supplied one word. -/
abbrev ReadContext (processors : Nat) (registerType : ValueType) :
    List ValueType :=
  [.index processors, registerType, .word]

/-- A finite instruction set for a conventional synchronous priority-CRCW
PRAM.  Every active processor executes one instruction per clock step.

Control flow is explicit: local, read, and write instructions have a static
successor; `branch` chooses between two successors; `halt` removes the
processor from subsequent steps. -/
inductive Instruction
    {wordWidth : Nat}
    (signature : Signature wordWidth)
    (processors memorySize labels : Nat)
    (registerType : ValueType) where
  | halt
  | local
      (nextRegisters :
        Term signature (LocalContext processors registerType) registerType)
      (nextPC : Option (Fin labels))
  | read
      (address :
        Term signature (LocalContext processors registerType)
          (.index memorySize))
      (receive :
        Term signature (ReadContext processors registerType) registerType)
      (nextPC : Option (Fin labels))
  | write
      (address :
        Term signature (LocalContext processors registerType)
          (.index memorySize))
      (value :
        Term signature (LocalContext processors registerType) .word)
      (nextRegisters :
        Term signature (LocalContext processors registerType) registerType)
      (nextPC : Option (Fin labels))
  | branch
      (condition :
        Term signature (LocalContext processors registerType) .boolean)
      (truePC falsePC : Option (Fin labels))

/-- A finite instruction table.  The two idle constants are semantically
unobservable for inactive memory operations, but make normalization total
without assuming an implicit address or word.  In particular, construction
of a program witnesses that shared memory is nonempty. -/
structure Program
    {wordWidth : Nat}
    (signature : Signature wordWidth)
    (processors memorySize labels : Nat)
    (registerType : ValueType) where
  idleAddress : Fin memorySize
  idleWord : Word wordWidth
  instructions :
    Fin labels →
      Instruction signature processors memorySize labels registerType

/-- Independent instruction-machine state.  `none` is a halted processor. -/
structure Config
    (wordWidth processors memorySize labels : Nat)
    (registerType : ValueType) where
  memory : Fin memorySize → Word wordWidth
  programCounters : Fin processors → Option (Fin labels)
  registers : Fin processors → registerType.denote wordWidth

namespace Config

@[ext]
theorem ext
    {wordWidth processors memorySize labels : Nat}
    {registerType : ValueType}
    (left right :
      Config wordWidth processors memorySize labels registerType)
    (memory : left.memory = right.memory)
    (programCounters : left.programCounters = right.programCounters)
    (registers : left.registers = right.registers) :
    left = right := by
  cases left
  cases right
  cases memory
  cases programCounters
  cases registers
  rfl

end Config

namespace Program

def localEnvironment
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (_program :
      Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors) :
    Environment wordWidth (LocalContext processors registerType) :=
  .cons processor (.cons (configuration.registers processor) .nil)

def readEnvironment
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (_program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors)
    (readValue : Word wordWidth) :
    Environment wordWidth (ReadContext processors registerType) :=
  .cons processor
    (.cons (configuration.registers processor)
      (.cons readValue .nil))

/-- Address observed by the instruction machine. Non-read instructions use
the explicit idle address, whose loaded value is ignored. -/
def readAddressAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors) : Fin memorySize :=
  match configuration.programCounters processor with
  | none => program.idleAddress
  | some pc =>
      match program.instructions pc with
      | .read address _ _ =>
          address.evaluate (program.localEnvironment configuration processor)
      | _ => program.idleAddress

/-- Reads use the complete pre-step snapshot. -/
def readValueAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors) : Word wordWidth :=
  configuration.memory (program.readAddressAt configuration processor)

def nextProgramCounterAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors) : Option (Fin labels) :=
  match configuration.programCounters processor with
  | none => none
  | some pc =>
      match program.instructions pc with
      | .halt => none
      | .local _ nextPC => nextPC
      | .read _ _ nextPC => nextPC
      | .write _ _ _ nextPC => nextPC
      | .branch condition truePC falsePC =>
          match condition.evaluate
              (program.localEnvironment configuration processor) with
          | true => truePC
          | false => falsePC

def nextRegistersAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors) : registerType.denote wordWidth :=
  match configuration.programCounters processor with
  | none => configuration.registers processor
  | some pc =>
      match program.instructions pc with
      | .halt => configuration.registers processor
      | .local nextRegisters _ =>
          nextRegisters.evaluate
            (program.localEnvironment configuration processor)
      | .read _ receive _ =>
          receive.evaluate
            (program.readEnvironment configuration processor
              (program.readValueAt configuration processor))
      | .write _ _ nextRegisters _ =>
          nextRegisters.evaluate
            (program.localEnvironment configuration processor)
      | .branch _ _ _ => configuration.registers processor

def writeEnabledAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors) : Bool :=
  match configuration.programCounters processor with
  | none => false
  | some pc =>
      match program.instructions pc with
      | .write _ _ _ _ => true
      | _ => false

def writeAddressAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors) : Fin memorySize :=
  match configuration.programCounters processor with
  | none => program.idleAddress
  | some pc =>
      match program.instructions pc with
      | .write address _ _ _ =>
          address.evaluate (program.localEnvironment configuration processor)
      | _ => program.idleAddress

def writeValueAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (processor : Fin processors) : Word wordWidth :=
  match configuration.programCounters processor with
  | none => program.idleWord
  | some pc =>
      match program.instructions pc with
      | .write _ value _ _ =>
          value.evaluate (program.localEnvironment configuration processor)
      | _ => program.idleWord

/-- Direct source-side priority fold. It mentions no normalized PRAM or Core
proposal type. -/
def foldProcessorAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (destination : Fin memorySize)
    (winner : Winner processors)
    (processor : Fin processors) : Winner processors :=
  if program.writeEnabledAt configuration processor &&
      program.writeAddressAt configuration processor == destination then
    Winner.combine winner (Winner.owner processor)
  else
    winner

def priorityOwnerAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (destination : Fin memorySize) : Winner processors :=
  (List.finRange processors).foldl
    (program.foldProcessorAt configuration destination)
    (Winner.none processors)

def nextMemoryAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType)
    (destination : Fin memorySize) : Word wordWidth :=
  match (program.priorityOwnerAt configuration destination).toOption with
  | none => configuration.memory destination
  | some processor => program.writeValueAt configuration processor

/-- One conventional synchronous PRAM clock step. -/
def step
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType)
    (configuration :
      Config wordWidth processors memorySize labels registerType) :
    Config wordWidth processors memorySize labels registerType where
  memory := program.nextMemoryAt configuration
  programCounters := program.nextProgramCounterAt configuration
  registers := program.nextRegistersAt configuration

def run
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : Program signature processors memorySize labels registerType) :
    Nat → Config wordWidth processors memorySize labels registerType →
      Config wordWidth processors memorySize labels registerType
  | 0, configuration => configuration
  | steps + 1, configuration =>
      run program steps (program.step configuration)

end Program

end MassivelyProof.InstructionPRAM
