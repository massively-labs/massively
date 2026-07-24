import MassivelyProof.PRAM
import MassivelyProof.Core

namespace MassivelyProof.Compilation

/-- Syntax-directed compilation from the external PRAM syntax to a normalized
Massively Core round. It does not inspect input data or unroll execution
rounds. -/
def compile
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType) :
    MassivelyCore.Program signature processors memorySize registerType where
  pullAddress := program.readAddress
  mapRegisters := program.nextRegisters
  selectWrite := program.writeEnabled
  pushAddress := program.writeAddress
  pushValue := program.writeValue

/-- Encode a source barrier state as a Core materialized state. -/
def encode
    {wordWidth processors memorySize : Nat}
    {registerType : ValueType}
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType) :
    MassivelyCore.Config wordWidth processors memorySize registerType where
  memory := configuration.memory
  registers := configuration.registers

/-- Decode a Core materialized state back to the source observation. -/
def decode
    {wordWidth processors memorySize : Nat}
    {registerType : ValueType}
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType) :
    PriorityCRCW.Config wordWidth processors memorySize registerType where
  memory := configuration.memory
  registers := configuration.registers

@[simp]
theorem decode_encode
    {wordWidth processors memorySize : Nat}
    {registerType : ValueType}
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType) :
    decode (encode configuration) = configuration := rfl

@[simp]
theorem encode_decode
    {wordWidth processors memorySize : Nat}
    {registerType : ValueType}
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType) :
    encode (decode configuration) = configuration := rfl

@[simp]
theorem compiled_pulledAddressAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).pulledAddressAt (encode configuration) processor =
      program.readAddressAt configuration processor := rfl

@[simp]
theorem compiled_pulledValueAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).pulledValueAt (encode configuration) processor =
      program.readValueAt configuration processor := rfl

@[simp]
theorem compiled_mappedRegistersAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).mappedRegistersAt (encode configuration) processor =
      program.nextRegistersAt configuration processor := rfl

@[simp]
theorem compiled_selectedAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).selectedAt (encode configuration) processor =
      program.writeEnabledAt configuration processor := rfl

@[simp]
theorem compiled_pushedAddressAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).pushedAddressAt (encode configuration) processor =
      program.writeAddressAt configuration processor := rfl

@[simp]
theorem compiled_pushedValueAt
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).pushedValueAt (encode configuration) processor =
      program.writeValueAt configuration processor := rfl

/-- Compacting target proposals and then folding a destination is equal to the
source machine's direct conditional fold over the same logical processors.
This is the central bridge between the independently defined machines. -/
theorem fold_proposalsFrom_compile
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize)
    (processorOrder : List (Fin processors))
    (initial : Winner processors) :
    ((compile program).proposalsFrom (encode configuration) processorOrder).foldl
        (MassivelyCore.Program.foldProposalAt destination) initial =
      processorOrder.foldl
        (program.foldProcessorAt configuration destination) initial := by
  induction processorOrder generalizing initial with
  | nil => rfl
  | cons processor processors induction =>
      cases enabled : program.writeEnabledAt configuration processor with
      | false =>
          simpa [MassivelyCore.Program.proposalsFrom,
            MassivelyCore.Program.proposalAt,
            PriorityCRCW.Program.foldProcessorAt,
            enabled] using induction initial
      | true =>
          by_cases addressMatches :
            program.writeAddressAt configuration processor = destination
          · simpa [MassivelyCore.Program.proposalsFrom,
              MassivelyCore.Program.proposalAt,
              MassivelyCore.Program.foldProposalAt,
              PriorityCRCW.Program.foldProcessorAt,
              enabled, addressMatches] using
                induction
                  (Winner.combine initial (Winner.owner processor))
          · simpa [MassivelyCore.Program.proposalsFrom,
              MassivelyCore.Program.proposalAt,
              MassivelyCore.Program.foldProposalAt,
              PriorityCRCW.Program.foldProcessorAt,
              enabled, addressMatches] using induction initial

/-- The target destination reduction chooses exactly the source priority
owner. -/
theorem winner_compile_correct
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) :
    (compile program).winnerAt (encode configuration) destination =
      program.priorityOwnerAt configuration destination := by
  unfold MassivelyCore.Program.winnerAt
    MassivelyCore.Program.winnerAtFrom
    MassivelyCore.Program.proposals
    PriorityCRCW.Program.priorityOwnerAt
  exact fold_proposalsFrom_compile program configuration destination
    (List.finRange processors) (Winner.none processors)

/-- The compiled Core result remains correct under every permutation of the
logical compacted proposal stream. -/
theorem scheduled_winner_compile_correct
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize)
    (scheduled : List (MassivelyCore.Proposal processors memorySize))
    (permutation :
      ((compile program).proposals (encode configuration)).Perm scheduled) :
    MassivelyCore.Program.winnerAtFrom destination scheduled =
      program.priorityOwnerAt configuration destination := by
  rw [← MassivelyCore.Program.winnerAtFrom_schedule_independent
    destination permutation]
  exact winner_compile_correct program configuration destination

/-- Controlled scatter produces the same pointwise memory update as the
source priority-CRCW rule. -/
theorem memory_compile_correct
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) :
    (compile program).scatteredMemoryAt (encode configuration) destination =
      program.nextMemoryAt configuration destination := by
  unfold MassivelyCore.Program.scatteredMemoryAt
    PriorityCRCW.Program.nextMemoryAt
  rw [winner_compile_correct]
  cases (program.priorityOwnerAt configuration destination).toOption <;> rfl

/-- One complete Core bulk round simulates one source PRAM round. -/
theorem compile_step_correct
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType) :
    (compile program).step (encode configuration) =
      encode (program.step configuration) := by
  apply MassivelyCore.Config.ext
  · funext destination
    exact memory_compile_correct program configuration destination
  · funext processor
    exact compiled_mappedRegistersAt program configuration processor

/-- Every finite Core execution simulates the corresponding source execution.
The compiled syntax is fixed; the theorem does not unroll `rounds`. -/
theorem compile_run_correct
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (rounds : Nat)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType) :
    (compile program).run rounds (encode configuration) =
      encode (program.run rounds configuration) := by
  induction rounds generalizing configuration with
  | zero => rfl
  | succ rounds induction =>
      simp only [MassivelyCore.Program.run, PriorityCRCW.Program.run]
      rw [compile_step_correct]
      exact induction _

/-- Public source-shaped finite-run observation theorem. -/
theorem compile_run_decode_correct
    {wordWidth processors memorySize : Nat}
    {signature : Signature wordWidth}
    {registerType : ValueType}
    (program : PriorityCRCW.Program signature processors memorySize registerType)
    (rounds : Nat)
    (configuration : PriorityCRCW.Config wordWidth processors memorySize registerType) :
    decode ((compile program).run rounds (encode configuration)) =
      program.run rounds configuration := by
  rw [compile_run_correct]
  rfl

end MassivelyProof.Compilation
