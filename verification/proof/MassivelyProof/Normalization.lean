import MassivelyProof.Cost
import MassivelyProof.InstructionPRAM

namespace MassivelyProof.InstructionNormalization

/-- Program counter plus user registers carried by the normalized machine.
The final bounded-index value is the halted sentinel. -/
abbrev EncodedRegisterType (labels : Nat) (registerType : ValueType) :
    ValueType :=
  .product (.index (labels + 1)) registerType

def encodePC {labels : Nat} : Option (Fin labels) → Fin (labels + 1)
  | none => Fin.last labels
  | some pc => pc.castSucc

def decodePC {labels : Nat} (encoded : Fin (labels + 1)) :
    Option (Fin labels) :=
  if bounded : encoded.val < labels then
    some ⟨encoded.val, bounded⟩
  else
    none

@[simp]
theorem decode_encodePC_none (labels : Nat) :
    decodePC (encodePC (labels := labels) none) = none := by
  simp [decodePC, encodePC]

@[simp]
theorem decode_encodePC_some {labels : Nat} (pc : Fin labels) :
    decodePC (encodePC (some pc)) = some pc := by
  simp [decodePC, encodePC]

@[simp]
theorem decode_encodePC {labels : Nat} (pc : Option (Fin labels)) :
    decodePC (encodePC pc) = pc := by
  cases pc <;> simp

@[simp]
theorem encode_decodePC {labels : Nat} (encoded : Fin (labels + 1)) :
    encodePC (decodePC encoded) = encoded := by
  unfold decodePC
  split
  · apply Fin.ext
    simp [encodePC]
  · apply Fin.ext
    have atSentinel : encoded.val = labels := by omega
    simp [encodePC, atSentinel, Fin.last]

@[simp]
theorem encodePC_match_bool
    {labels : Nat}
    (condition : Bool)
    (whenTrue whenFalse : Option (Fin labels)) :
    (match condition with
      | true => encodePC whenTrue
      | false => encodePC whenFalse) =
      encodePC
        (match condition with
        | true => whenTrue
        | false => whenFalse) := by
  cases condition <;> rfl

namespace Terms

def readState
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType} :
    Term signature
      (PriorityCRCW.ReadContext processors
        (EncodedRegisterType labels registerType))
      (EncodedRegisterType labels registerType) :=
  .read (.there .here)

def transitionState
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType} :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType))
      (EncodedRegisterType labels registerType) :=
  .read (.there .here)

def readPC
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType} :
    Term signature
      (PriorityCRCW.ReadContext processors
        (EncodedRegisterType labels registerType))
      (.index (labels + 1)) :=
  .first readState

def transitionPC
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType} :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType))
      (.index (labels + 1)) :=
  .first transitionState

def transitionRegisters
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType} :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType))
      registerType :=
  .second transitionState

def localReadReplacement
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType} :
    ∀ {output},
      Variable (InstructionPRAM.LocalContext processors registerType) output →
        Term signature
          (PriorityCRCW.ReadContext processors
            (EncodedRegisterType labels registerType)) output
  | _, .here => .read .here
  | _, .there .here => .second readState

def localTransitionReplacement
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType} :
    ∀ {output},
      Variable (InstructionPRAM.LocalContext processors registerType) output →
        Term signature
          (PriorityCRCW.TransitionContext processors
            (EncodedRegisterType labels registerType)) output
  | _, .here => .read .here
  | _, .there .here => transitionRegisters

def readTransitionReplacement
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType} :
    ∀ {output},
      Variable (InstructionPRAM.ReadContext processors registerType) output →
        Term signature
          (PriorityCRCW.TransitionContext processors
            (EncodedRegisterType labels registerType)) output
  | _, .here => .read .here
  | _, .there .here => transitionRegisters
  | _, .there (.there .here) => .read (.there (.there .here))

def liftLocalRead
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType output : ValueType}
    (term :
      Term signature
        (InstructionPRAM.LocalContext processors registerType) output) :
    Term signature
      (PriorityCRCW.ReadContext processors
        (EncodedRegisterType labels registerType)) output :=
  term.substitute localReadReplacement

def liftLocalTransition
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType output : ValueType}
    (term :
      Term signature
        (InstructionPRAM.LocalContext processors registerType) output) :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType)) output :=
  term.substitute localTransitionReplacement

def liftReadTransition
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType output : ValueType}
    (term :
      Term signature
        (InstructionPRAM.ReadContext processors registerType) output) :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType)) output :=
  term.substitute readTransitionReplacement

def encodedReadEnvironment
    {wordWidth processors labels : Nat}
    {registerType : ValueType}
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth) :
    Environment wordWidth
      (PriorityCRCW.ReadContext processors
        (EncodedRegisterType labels registerType)) :=
  .cons processor (.cons (encodePC pc, registers) .nil)

def encodedTransitionEnvironment
    {wordWidth processors labels : Nat}
    {registerType : ValueType}
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth)
    (readValue : Word wordWidth) :
    Environment wordWidth
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType)) :=
  .cons processor
    (.cons (encodePC pc, registers) (.cons readValue .nil))

def instructionLocalEnvironment
    {wordWidth processors : Nat}
    {registerType : ValueType}
    (processor : Fin processors)
    (registers : registerType.denote wordWidth) :
    Environment wordWidth
      (InstructionPRAM.LocalContext processors registerType) :=
  .cons processor (.cons registers .nil)

def instructionReadEnvironment
    {wordWidth processors : Nat}
    {registerType : ValueType}
    (processor : Fin processors)
    (registers : registerType.denote wordWidth)
    (readValue : Word wordWidth) :
    Environment wordWidth
      (InstructionPRAM.ReadContext processors registerType) :=
  .cons processor (.cons registers (.cons readValue .nil))

@[simp]
theorem evaluate_readPC
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth) :
    (@readPC wordWidth processors labels signature registerType).evaluate
        (encodedReadEnvironment processor pc registers) =
      encodePC pc := rfl

@[simp]
theorem evaluate_transitionPC
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth)
    (readValue : Word wordWidth) :
    (@transitionPC wordWidth processors labels signature registerType).evaluate
        (encodedTransitionEnvironment processor pc registers readValue) =
      encodePC pc := rfl

@[simp]
theorem evaluate_transitionRegisters
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth)
    (readValue : Word wordWidth) :
    (@transitionRegisters wordWidth processors labels signature
      registerType).evaluate
        (encodedTransitionEnvironment processor pc registers readValue) =
      registers := rfl

@[simp]
theorem evaluate_transitionState
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth)
    (readValue : Word wordWidth) :
    (@transitionState wordWidth processors labels signature
      registerType).evaluate
        (encodedTransitionEnvironment processor pc registers readValue) =
      (encodePC pc, registers) := rfl

@[simp]
theorem evaluate_liftLocalRead
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType output : ValueType}
    (term :
      Term signature
        (InstructionPRAM.LocalContext processors registerType) output)
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth) :
    (liftLocalRead term).evaluate
        (encodedReadEnvironment processor pc registers) =
      term.evaluate (instructionLocalEnvironment processor registers) := by
  apply Term.evaluate_substitute
  intro variableType reference
  cases reference with
  | here =>
      simp [localReadReplacement, encodedReadEnvironment,
        instructionLocalEnvironment, Term.evaluate, Variable.lookup]
  | there rest => cases rest with
    | here =>
        simp [localReadReplacement, encodedReadEnvironment,
          instructionLocalEnvironment, readState,
          Term.evaluate, Variable.lookup]
    | there impossible => exact nomatch impossible

@[simp]
theorem evaluate_liftLocalTransition
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType output : ValueType}
    (term :
      Term signature
        (InstructionPRAM.LocalContext processors registerType) output)
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth)
    (readValue : Word wordWidth) :
    (liftLocalTransition term).evaluate
        (encodedTransitionEnvironment processor pc registers readValue) =
      term.evaluate (instructionLocalEnvironment processor registers) := by
  apply Term.evaluate_substitute
  intro variableType reference
  cases reference with
  | here =>
      simp [localTransitionReplacement, encodedTransitionEnvironment,
        instructionLocalEnvironment, Term.evaluate, Variable.lookup]
  | there rest => cases rest with
    | here =>
        simp [localTransitionReplacement, encodedTransitionEnvironment,
          instructionLocalEnvironment, transitionRegisters,
          transitionState, Term.evaluate, Variable.lookup]
    | there impossible => exact nomatch impossible

@[simp]
theorem evaluate_liftReadTransition
    {wordWidth processors labels : Nat}
    {signature : Signature wordWidth}
    {registerType output : ValueType}
    (term :
      Term signature
        (InstructionPRAM.ReadContext processors registerType) output)
    (processor : Fin processors)
    (pc : Option (Fin labels))
    (registers : registerType.denote wordWidth)
    (readValue : Word wordWidth) :
    (liftReadTransition term).evaluate
        (encodedTransitionEnvironment processor pc registers readValue) =
      term.evaluate
        (instructionReadEnvironment processor registers readValue) := by
  apply Term.evaluate_substitute
  intro variableType reference
  cases reference with
  | here =>
      simp [readTransitionReplacement, encodedTransitionEnvironment,
        instructionReadEnvironment, Term.evaluate, Variable.lookup]
  | there rest => cases rest with
    | here =>
        simp [readTransitionReplacement, encodedTransitionEnvironment,
          instructionReadEnvironment, transitionRegisters,
          transitionState, Term.evaluate, Variable.lookup]
    | there rest => cases rest with
      | here =>
          simp [readTransitionReplacement, encodedTransitionEnvironment,
            instructionReadEnvironment, Term.evaluate, Variable.lookup]
      | there impossible => exact nomatch impossible

end Terms

open Terms

def readAddressBranch
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (encodedPC : Fin (labels + 1)) :
    Term signature
      (PriorityCRCW.ReadContext processors
        (EncodedRegisterType labels registerType))
      (.index memorySize) :=
  match decodePC encodedPC with
  | none => .constant program.idleAddress
  | some pc =>
      match program.instructions pc with
      | .read address _ _ => liftLocalRead address
      | _ => .constant program.idleAddress

def nextStateBranch
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (encodedPC : Fin (labels + 1)) :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType))
      (EncodedRegisterType labels registerType) :=
  match decodePC encodedPC with
  | none => transitionState
  | some pc =>
      match program.instructions pc with
      | .halt =>
          .pair (.constant (encodePC none)) transitionRegisters
      | .local nextRegisters nextPC =>
          .pair (.constant (encodePC nextPC))
            (liftLocalTransition nextRegisters)
      | .read _ receive nextPC =>
          .pair (.constant (encodePC nextPC))
            (liftReadTransition receive)
      | .write _ _ nextRegisters nextPC =>
          .pair (.constant (encodePC nextPC))
            (liftLocalTransition nextRegisters)
      | .branch condition truePC falsePC =>
          .pair
            (.ite (liftLocalTransition condition)
              (.constant (encodePC truePC))
              (.constant (encodePC falsePC)))
            transitionRegisters

def writeEnabledBranch
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (encodedPC : Fin (labels + 1)) :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType)) .boolean :=
  match decodePC encodedPC with
  | some pc =>
      match program.instructions pc with
      | .write _ _ _ _ => .constant true
      | _ => .constant false
  | none => .constant false

def writeAddressBranch
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (encodedPC : Fin (labels + 1)) :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType))
      (.index memorySize) :=
  match decodePC encodedPC with
  | some pc =>
      match program.instructions pc with
      | .write address _ _ _ => liftLocalTransition address
      | _ => .constant program.idleAddress
  | none => .constant program.idleAddress

def writeValueBranch
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (encodedPC : Fin (labels + 1)) :
    Term signature
      (PriorityCRCW.TransitionContext processors
        (EncodedRegisterType labels registerType)) .word :=
  match decodePC encodedPC with
  | some pc =>
      match program.instructions pc with
      | .write _ value _ _ => liftLocalTransition value
      | _ => .constant program.idleWord
  | none => .constant program.idleWord

/-- Syntax-directed normalization. One instruction-machine clock becomes one
round of the existing one-read/at-most-one-write PRAM. -/
def normalize
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType) :
    PriorityCRCW.Program signature processors memorySize
      (EncodedRegisterType labels registerType) where
  readAddress := .caseIndex readPC (readAddressBranch program)
  nextRegisters := .caseIndex transitionPC (nextStateBranch program)
  writeEnabled := .caseIndex transitionPC (writeEnabledBranch program)
  writeAddress := .caseIndex transitionPC (writeAddressBranch program)
  writeValue := .caseIndex transitionPC (writeValueBranch program)

def encode
    {wordWidth processors memorySize labels : Nat}
    {registerType : ValueType}
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    PriorityCRCW.Config wordWidth processors memorySize
      (EncodedRegisterType labels registerType) where
  memory := configuration.memory
  registers := fun processor =>
    (encodePC (configuration.programCounters processor),
      configuration.registers processor)

def decode
    {wordWidth processors memorySize labels : Nat}
    {registerType : ValueType}
    (configuration :
      PriorityCRCW.Config wordWidth processors memorySize
        (EncodedRegisterType labels registerType)) :
    InstructionPRAM.Config wordWidth processors memorySize labels
      registerType where
  memory := configuration.memory
  programCounters := fun processor =>
    decodePC (configuration.registers processor).1
  registers := fun processor => (configuration.registers processor).2

@[simp]
theorem decode_encode
    {wordWidth processors memorySize labels : Nat}
    {registerType : ValueType}
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    decode (encode configuration) = configuration := by
  apply InstructionPRAM.Config.ext
  · rfl
  · funext processor
    exact decode_encodePC _
  · rfl

@[simp]
theorem encode_decode
    {wordWidth processors memorySize labels : Nat}
    {registerType : ValueType}
    (configuration :
      PriorityCRCW.Config wordWidth processors memorySize
        (EncodedRegisterType labels registerType)) :
    encode (decode configuration) = configuration := by
  apply PriorityCRCW.Config.ext
  · rfl
  · funext processor
    apply Prod.ext
    · exact encode_decodePC _
    · rfl

@[simp]
theorem normalized_readEnvironment
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    (normalize program).readEnvironment (encode configuration) processor =
      encodedReadEnvironment processor
        (configuration.programCounters processor)
        (configuration.registers processor) := rfl

@[simp]
theorem instruction_localEnvironment
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    program.localEnvironment configuration processor =
      instructionLocalEnvironment processor
        (configuration.registers processor) := rfl

@[simp]
theorem normalized_readAddressAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    (normalize program).readAddressAt (encode configuration) processor =
      program.readAddressAt configuration processor := by
  unfold PriorityCRCW.Program.readAddressAt
  rw [normalized_readEnvironment]
  cases pcEquation : configuration.programCounters processor with
  | none =>
      simp [normalize, readAddressBranch,
        InstructionPRAM.Program.readAddressAt, pcEquation, Term.evaluate]
  | some pc =>
      cases instructionEquation : program.instructions pc <;>
        simp [normalize, readAddressBranch,
          InstructionPRAM.Program.readAddressAt, pcEquation,
          instructionEquation, Term.evaluate]

@[simp]
theorem normalized_readValueAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    (normalize program).readValueAt (encode configuration) processor =
      program.readValueAt configuration processor := by
  unfold PriorityCRCW.Program.readValueAt
    InstructionPRAM.Program.readValueAt
  rw [normalized_readAddressAt]
  rfl

@[simp]
theorem normalized_transitionEnvironment
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    (normalize program).transitionEnvironment (encode configuration) processor =
      encodedTransitionEnvironment processor
        (configuration.programCounters processor)
        (configuration.registers processor)
        (program.readValueAt configuration processor) := by
  unfold PriorityCRCW.Program.transitionEnvironment
  rw [normalized_readValueAt]
  rfl

@[simp]
theorem instruction_readEnvironment
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors)
    (readValue : Word wordWidth) :
    program.readEnvironment configuration processor readValue =
      instructionReadEnvironment processor
        (configuration.registers processor) readValue := rfl

@[simp]
theorem normalized_nextRegistersAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    (normalize program).nextRegistersAt (encode configuration) processor =
      (encodePC (program.nextProgramCounterAt configuration processor),
        program.nextRegistersAt configuration processor) := by
  unfold PriorityCRCW.Program.nextRegistersAt
  rw [normalized_transitionEnvironment]
  cases pcEquation : configuration.programCounters processor with
  | none =>
      simp [normalize, nextStateBranch,
        InstructionPRAM.Program.nextProgramCounterAt,
        InstructionPRAM.Program.nextRegistersAt, pcEquation, Term.evaluate]
  | some pc =>
      cases instructionEquation : program.instructions pc with
      | halt =>
        simp [normalize, nextStateBranch,
          InstructionPRAM.Program.nextProgramCounterAt,
          InstructionPRAM.Program.nextRegistersAt, pcEquation,
          instructionEquation, Term.evaluate]
        rfl
      | «local» nextRegisters nextPC =>
        simp [normalize, nextStateBranch,
          InstructionPRAM.Program.nextProgramCounterAt,
          InstructionPRAM.Program.nextRegistersAt, pcEquation,
          instructionEquation, Term.evaluate]
        rfl
      | read address receive nextPC =>
        simp [normalize, nextStateBranch,
          InstructionPRAM.Program.nextProgramCounterAt,
          InstructionPRAM.Program.nextRegistersAt, pcEquation,
          instructionEquation, Term.evaluate]
        rfl
      | write address value nextRegisters nextPC =>
        simp [normalize, nextStateBranch,
          InstructionPRAM.Program.nextProgramCounterAt,
          InstructionPRAM.Program.nextRegistersAt, pcEquation,
          instructionEquation, Term.evaluate]
        rfl
      | branch condition truePC falsePC =>
        simp [normalize, nextStateBranch,
          InstructionPRAM.Program.nextProgramCounterAt,
          InstructionPRAM.Program.nextRegistersAt, pcEquation,
          instructionEquation, Term.evaluate]
        cases condition.evaluate
            (instructionLocalEnvironment processor
              (configuration.registers processor)) <;> rfl

@[simp]
theorem normalized_writeEnabledAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    (normalize program).writeEnabledAt (encode configuration) processor =
      program.writeEnabledAt configuration processor := by
  unfold PriorityCRCW.Program.writeEnabledAt
  rw [normalized_transitionEnvironment]
  cases pcEquation : configuration.programCounters processor with
  | none =>
      simp [normalize, writeEnabledBranch,
        InstructionPRAM.Program.writeEnabledAt, pcEquation, Term.evaluate]
  | some pc =>
      cases instructionEquation : program.instructions pc <;>
        simp [normalize, writeEnabledBranch,
          InstructionPRAM.Program.writeEnabledAt, pcEquation,
          instructionEquation, Term.evaluate]

@[simp]
theorem normalized_writeAddressAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    (normalize program).writeAddressAt (encode configuration) processor =
      program.writeAddressAt configuration processor := by
  unfold PriorityCRCW.Program.writeAddressAt
  rw [normalized_transitionEnvironment]
  cases pcEquation : configuration.programCounters processor with
  | none =>
      simp [normalize, writeAddressBranch,
        InstructionPRAM.Program.writeAddressAt, pcEquation, Term.evaluate]
  | some pc =>
      cases instructionEquation : program.instructions pc <;>
        simp [normalize, writeAddressBranch,
          InstructionPRAM.Program.writeAddressAt, pcEquation,
          instructionEquation, Term.evaluate]

@[simp]
theorem normalized_writeValueAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (processor : Fin processors) :
    (normalize program).writeValueAt (encode configuration) processor =
      program.writeValueAt configuration processor := by
  unfold PriorityCRCW.Program.writeValueAt
  rw [normalized_transitionEnvironment]
  cases pcEquation : configuration.programCounters processor with
  | none =>
      simp [normalize, writeValueBranch,
        InstructionPRAM.Program.writeValueAt, pcEquation, Term.evaluate]
  | some pc =>
      cases instructionEquation : program.instructions pc <;>
        simp [normalize, writeValueBranch,
          InstructionPRAM.Program.writeValueAt, pcEquation,
          instructionEquation, Term.evaluate]

theorem normalized_foldProcessorAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (destination : Fin memorySize)
    (winner : Winner processors)
    (processor : Fin processors) :
    (normalize program).foldProcessorAt (encode configuration) destination
        winner processor =
      program.foldProcessorAt configuration destination winner processor := by
  unfold PriorityCRCW.Program.foldProcessorAt
    InstructionPRAM.Program.foldProcessorAt
  rw [normalized_writeEnabledAt, normalized_writeAddressAt]

@[simp]
theorem normalized_priorityOwnerAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (destination : Fin memorySize) :
    (normalize program).priorityOwnerAt (encode configuration) destination =
      program.priorityOwnerAt configuration destination := by
  have foldEquality :
      (normalize program).foldProcessorAt (encode configuration) destination =
        program.foldProcessorAt configuration destination := by
    funext winner processor
    exact normalized_foldProcessorAt program configuration destination
      winner processor
  unfold PriorityCRCW.Program.priorityOwnerAt
    InstructionPRAM.Program.priorityOwnerAt
  rw [foldEquality]

@[simp]
theorem normalized_nextMemoryAt
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType)
    (destination : Fin memorySize) :
    (normalize program).nextMemoryAt (encode configuration) destination =
      program.nextMemoryAt configuration destination := by
  unfold PriorityCRCW.Program.nextMemoryAt
    InstructionPRAM.Program.nextMemoryAt
  rw [normalized_priorityOwnerAt]
  cases (program.priorityOwnerAt configuration destination).toOption with
  | none => rfl
  | some processor => exact normalized_writeValueAt program configuration _

/-- One conventional instruction clock is simulated by exactly one normalized
PRAM round. -/
theorem normalize_step_correct
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    (normalize program).step (encode configuration) =
      encode (program.step configuration) := by
  apply PriorityCRCW.Config.ext
  · funext destination
    exact normalized_nextMemoryAt program configuration destination
  · funext processor
    exact normalized_nextRegistersAt program configuration processor

/-- Every finite instruction execution is preserved. The same `steps` value
is used on both sides: normalization has factor-one global-round overhead. -/
theorem normalize_run_correct
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (steps : Nat)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    (normalize program).run steps (encode configuration) =
      encode (program.run steps configuration) := by
  induction steps generalizing configuration with
  | zero => rfl
  | succ steps induction =>
      simp only [PriorityCRCW.Program.run, InstructionPRAM.Program.run]
      rw [normalize_step_correct]
      exact induction _

/-- Source-shaped observation theorem for normalization. -/
theorem normalize_run_decode_correct
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (steps : Nat)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    decode ((normalize program).run steps (encode configuration)) =
      program.run steps configuration := by
  rw [normalize_run_correct]
  exact decode_encode _

/-- Abstract global-round cost of normalization. -/
def normalizedRoundCount (instructionSteps : Nat) : Nat := instructionSteps

@[simp]
theorem normalizedRoundCount_factor_one (instructionSteps : Nat) :
    normalizedRoundCount instructionSteps = instructionSteps := rfl

/-- End-to-end compiler from the conventional instruction machine to the
bulk-synchronous Core calculus. -/
def compileToCore
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType) :
    MassivelyCore.Program signature processors memorySize
      (EncodedRegisterType labels registerType) :=
  Compilation.compile (normalize program)

def encodeCore
    {wordWidth processors memorySize labels : Nat}
    {registerType : ValueType}
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    MassivelyCore.Config wordWidth processors memorySize
      (EncodedRegisterType labels registerType) :=
  Compilation.encode (encode configuration)

def decodeCore
    {wordWidth processors memorySize labels : Nat}
    {registerType : ValueType}
    (configuration :
      MassivelyCore.Config wordWidth processors memorySize
        (EncodedRegisterType labels registerType)) :
    InstructionPRAM.Config wordWidth processors memorySize labels
      registerType :=
  decode (Compilation.decode configuration)

/-- Strong target-shaped end-to-end simulation theorem. -/
theorem compileToCore_run_correct
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (steps : Nat)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    (compileToCore program).run steps (encodeCore configuration) =
      Compilation.encode (encode (program.run steps configuration)) := by
  unfold compileToCore encodeCore
  rw [Compilation.compile_run_correct, normalize_run_correct]

/-- Main source-shaped completeness theorem for the conventional
instruction-level priority-CRCW PRAM.  It composes instruction normalization
with the independent Core simulation and preserves every finite run. -/
theorem compileToCore_run_decode_correct
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (steps : Nat)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    decodeCore
        ((compileToCore program).run steps (encodeCore configuration)) =
      program.run steps configuration := by
  unfold decodeCore
  rw [compileToCore_run_correct]
  exact decode_encode _

/-- Core compilation adds no second copy of the normalized scalar syntax. -/
theorem compileToCore_nodeCount
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType) :
    (compileToCore program).nodeCount = (normalize program).nodeCount :=
  Compilation.compile_nodeCount _

/-- Exact resource totals of the normalized general Core schedule.  These are
symbolic schedule counts, not elapsed-time claims about a particular GPU. -/
@[simp]
theorem compileToCore_runCost_launches
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (steps : Nat)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    ((compileToCore program).runCost steps
      (encodeCore configuration)).launches = steps * 5 :=
  MassivelyCore.Program.runCost_launches _ _ _

@[simp]
theorem compileToCore_runCost_barriers
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (steps : Nat)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    ((compileToCore program).runCost steps
      (encodeCore configuration)).barriers = steps * 4 :=
  MassivelyCore.Program.runCost_barriers _ _ _

@[simp]
theorem compileToCore_runCost_atomics
    {wordWidth processors memorySize labels : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : InstructionPRAM.Program signature processors memorySize labels
      registerType)
    (steps : Nat)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels
        registerType) :
    ((compileToCore program).runCost steps
      (encodeCore configuration)).atomics = 0 :=
  MassivelyCore.Program.runCost_atomics _ _ _

end MassivelyProof.InstructionNormalization
