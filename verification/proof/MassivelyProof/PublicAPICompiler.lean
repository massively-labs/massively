import MassivelyProof.PublicAPI
import MassivelyProof.Normalization

namespace MassivelyProof.PublicAPICompilation

open MassivelyProof

/-- Syntax-directed lowering of a Core round to the public vector API basis. -/
def compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType) :
    PublicAPI.Program (wordWidth := wordWidth) signature processors memorySize registerType where
  pullAddress := program.pullAddress
  mapRegisters := program.mapRegisters
  selectWrite := program.selectWrite
  pushAddress := program.pushAddress
  pushValue := program.pushValue

@[simp] theorem pulledAddress_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).pulledAddressAt configuration processor =
      program.pulledAddressAt configuration processor := rfl

@[simp] theorem pulledValue_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).pulledValueAt configuration processor =
      program.pulledValueAt configuration processor := rfl

@[simp] theorem mappedRegisters_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).mappedRegistersAt configuration processor =
      program.mappedRegistersAt configuration processor := rfl

@[simp] theorem selected_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    (compile program).selectedAt configuration processor =
      program.selectedAt configuration processor := rfl

@[simp] theorem writeRow_destination_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    ((compile program).writeRowAt configuration processor).destination =
      program.pushedAddressAt configuration processor := rfl

@[simp] theorem writeRow_owner_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    ((compile program).writeRowAt configuration processor).owner = processor := rfl

@[simp] theorem writeRow_value_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    ((compile program).writeRowAt configuration processor).value =
      program.pushedValueAt configuration processor := rfl

/-- The API sort stage has one destination block, inside which owners occur in
increasing processor order. -/
theorem sortedWriteRows_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType) :
    (compile program).sortedWriteRows configuration = fun destination =>
      ((List.finRange processors).filter fun processor =>
        program.selectedAt configuration processor &&
          program.pushedAddressAt configuration processor = destination).map
            ((compile program).writeRowAt configuration) := by
  funext destination
  rfl

/-- First enabled owner for one destination, in priority order. -/
def firstOwnerAt
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) : Option (Fin processors) :=
  (List.finRange processors).find? fun processor =>
    program.selectedAt configuration processor &&
      program.pushedAddressAt configuration processor = destination

/-- Direct destination fold over processor identifiers.  This is the same
priority reduction as Core, without passing through the proposal row type. -/
def foldOwnerAt
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize)
    (winner : Winner processors)
    (processor : Fin processors) : Winner processors :=
  if program.selectedAt configuration processor &&
      program.pushedAddressAt configuration processor = destination then
    Winner.combine winner (Winner.owner processor)
  else winner

theorem foldOwnerAt_preserves_least
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize)
    (least : Fin processors)
    (processorList : List (Fin processors))
    (leastBefore : ∀ processor ∈ processorList, least.val < processor.val) :
    processorList.foldl (foldOwnerAt program configuration destination)
        (Winner.owner least) = Winner.owner least := by
  induction processorList with
  | nil => rfl
  | cons processor rest induction =>
      have less : least.val < processor.val := leastBefore processor (by simp)
      have leastTail : ∀ other ∈ rest, least.val < other.val := by
        intro other member
        exact leastBefore other (by simp [member])
      cases rowMatches :
          program.selectedAt configuration processor &&
            program.pushedAddressAt configuration processor = destination
      · simpa [foldOwnerAt, rowMatches] using induction leastTail
      · have combineLeast :
            Winner.combine (Winner.owner least) (Winner.owner processor) =
              Winner.owner least := by
          unfold Winner.combine Winner.owner
          apply Fin.ext
          simp only [Fin.val_castSucc]
          exact Nat.min_eq_left (Nat.le_of_lt less)
        rw [List.foldl_cons]
        rw [show foldOwnerAt program configuration destination
              (Winner.owner least) processor =
              Winner.combine (Winner.owner least) (Winner.owner processor) by
          simp [foldOwnerAt, rowMatches]]
        rw [combineLeast]
        exact induction leastTail

theorem foldOwnerAt_eq_foldProposal
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize)
    (processorsList : List (Fin processors))
    (initial : Winner processors) :
    processorsList.foldl (foldOwnerAt program configuration destination) initial =
      (processorsList.filterMap (program.proposalAt configuration)).foldl
        (MassivelyCore.Program.foldProposalAt destination) initial := by
  induction processorsList generalizing initial with
  | nil => rfl
  | cons processor rest induction =>
      cases enabled : program.selectedAt configuration processor with
      | false =>
          simpa [foldOwnerAt, MassivelyCore.Program.proposalAt, enabled] using
            induction initial
      | true =>
          by_cases address : program.pushedAddressAt configuration processor = destination
          · simpa [foldOwnerAt, MassivelyCore.Program.proposalAt, enabled, address,
              MassivelyCore.Program.foldProposalAt] using
              induction (Winner.combine initial (Winner.owner processor))
          · simpa [foldOwnerAt, MassivelyCore.Program.proposalAt, enabled, address,
              MassivelyCore.Program.foldProposalAt] using induction initial

/-- Folding an increasing processor list from its sentinel selects its first
matching processor.  This is the algebraic fact behind the API
`sort(destination, owner) → unique_by_key(destination)` construction. -/
theorem find_eq_priorityFold
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize)
    (processorList : List (Fin processors))
    (increasing : processorList.Pairwise fun left right => left.val < right.val) :
    processorList.find? (fun processor =>
        program.selectedAt configuration processor &&
          program.pushedAddressAt configuration processor = destination) =
      (processorList.foldl (foldOwnerAt program configuration destination)
        (Winner.none processors)).toOption := by
  induction processorList with
  | nil => simp [Winner.toOption, Winner.none]
  | cons processor rest induction =>
      have restIncreasing := increasing.tail
      have processorLeast : ∀ other ∈ rest, processor.val < other.val := by
        intro other member
        exact (List.pairwise_cons.mp increasing).1 other member
      by_cases rowMatches :
        program.selectedAt configuration processor &&
          program.pushedAddressAt configuration processor = destination
      · have tailCannotBeat :
          rest.foldl (foldOwnerAt program configuration destination)
              (Winner.owner processor) = Winner.owner processor := by
          exact foldOwnerAt_preserves_least program configuration destination
            processor rest processorLeast
        simp only [List.find?_cons, rowMatches]
        simp only [List.foldl_cons, foldOwnerAt, rowMatches, ↓reduceIte]
        have noneCombine :
            Winner.combine (Winner.none processors) (Winner.owner processor) =
              Winner.owner processor := by
          unfold Winner.combine Winner.none Winner.owner
          apply Fin.ext
          simp only [Fin.val_last, Fin.val_castSucc]
          exact Nat.min_eq_right (Nat.le_of_lt processor.isLt)
        rw [noneCombine, tailCannotBeat]
        exact (Winner.toOption_owner processor).symm
      · simp [rowMatches, foldOwnerAt, induction restIncreasing]

/-- API sorting followed by `unique_by_key` returns exactly the write row of
the first (least-id) enabled processor at each destination. -/
theorem winningWriteRows_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType) :
    (compile program).winningWriteRows configuration = fun destination =>
      (firstOwnerAt program configuration destination).map fun processor =>
        (compile program).writeRowAt configuration processor := by
  funext destination
  unfold PublicAPI.Program.winningWriteRows PublicAPI.Basis.uniqueByDestination
  rw [sortedWriteRows_compile]
  unfold firstOwnerAt
  induction List.finRange processors with
  | nil => simp
  | cons processor rest induction =>
      by_cases rowMatches :
        program.selectedAt configuration processor &&
          program.pushedAddressAt configuration processor = destination
      · simp [rowMatches]
      · simp [rowMatches, induction]

/-- The canonical finite processor enumeration is strictly increasing. -/
theorem finRange_pairwise_lt (count : Nat) :
    (List.finRange count).Pairwise fun left right => left.val < right.val := by
  induction count with
  | zero => simp
  | succ count induction =>
      rw [List.finRange_succ, List.pairwise_cons]
      constructor
      · intro processor member
        simp only [List.mem_map] at member
        obtain ⟨original, _, rfl⟩ := member
        simp
      · simpa [List.pairwise_map] using induction

/-- The public first-row observation and Core winner fold choose the same
priority owner. -/
theorem firstOwnerAt_eq_winner
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) :
    firstOwnerAt program configuration destination =
      (program.winnerAt configuration destination).toOption := by
  unfold firstOwnerAt MassivelyCore.Program.winnerAt
    MassivelyCore.Program.proposals MassivelyCore.Program.proposalsFrom
    MassivelyCore.Program.winnerAtFrom
  rw [find_eq_priorityFold program configuration destination
    (List.finRange processors) (finRange_pairwise_lt processors)]
  rw [foldOwnerAt_eq_foldProposal]

/-- Pointwise memory result of the public API basis equals Core controlled
scatter. -/
theorem nextMemory_compile
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) :
    (compile program).nextMemory configuration destination =
      program.scatteredMemoryAt configuration destination := by
  unfold PublicAPI.Program.nextMemory PublicAPI.Basis.scatterDistinct
    PublicAPI.Basis.copy MassivelyCore.Program.scatteredMemoryAt
  rw [winningWriteRows_compile]
  dsimp only
  rw [firstOwnerAt_eq_winner]
  cases (program.winnerAt configuration destination).toOption <;> rfl

/-- One public API call sequence preserves one Core barrier transition. -/
theorem compile_step_correct
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType) :
    (compile program).step configuration = program.step configuration := by
  apply MassivelyCore.Config.ext
  · funext destination
    exact nextMemory_compile program configuration destination
  · funext processor
    exact mappedRegisters_compile program configuration processor

/-- Every finite Core run is preserved by the fixed public API program. -/
theorem compile_run_correct
    (program : MassivelyCore.Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (rounds : Nat)
    (configuration : MassivelyCore.Config wordWidth processors memorySize registerType) :
    (compile program).run rounds configuration = program.run rounds configuration := by
  induction rounds generalizing configuration with
  | zero => rfl
  | succ rounds induction =>
      simp only [PublicAPI.Program.run, MassivelyCore.Program.run]
      rw [compile_step_correct]
      exact induction _

end MassivelyProof.PublicAPICompilation

namespace MassivelyProof.InstructionNormalization

/-- End-to-end compiler from the conventional instruction machine to a fixed
program over the public Massively vector API basis. -/
def compileToPublicAPI
    (program : InstructionPRAM.Program (wordWidth := wordWidth) signature processors memorySize labels registerType) :
    PublicAPI.Program (wordWidth := wordWidth) signature processors memorySize
      (EncodedRegisterType labels registerType) :=
  PublicAPICompilation.compile (compileToCore program)

/-- Main API-level completeness theorem.  The witness program consists only of
the public basis modeled in `PublicAPI`: map, gather, copy_where, sort,
unique_by_key, copy, and collision-free scatter. -/
theorem compileToPublicAPI_run_decode_correct
    (program : InstructionPRAM.Program (wordWidth := wordWidth) signature processors memorySize labels registerType)
    (steps : Nat)
    (configuration :
      InstructionPRAM.Config wordWidth processors memorySize labels registerType) :
    decodeCore
        ((compileToPublicAPI program).run steps (encodeCore configuration)) =
      program.run steps configuration := by
  unfold compileToPublicAPI
  rw [PublicAPICompilation.compile_run_correct]
  exact compileToCore_run_decode_correct program steps configuration

end MassivelyProof.InstructionNormalization
