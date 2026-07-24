import MassivelyProof.Normalization

namespace MassivelyProof.Example

/-- Small literal vocabulary witnessing that the typed source fragment is
inhabited without admitting arbitrary scalar callbacks. -/
inductive Literal : ValueType → Type
  | truth : Literal .boolean
  | zeroIndex (bound : Nat) : Literal (.index (bound + 1))

def denoteLiteral
    {wordWidth : Nat} {output : ValueType}
    (literal : Literal output) : output.denote wordWidth :=
  match literal with
  | .truth => true
  | .zeroIndex bound => ⟨0, Nat.succ_pos bound⟩

/-- The example needs no primitive operations beyond structural reads and
literals. -/
def signature (wordWidth : Nat) : Signature wordWidth where
  Literal := Literal
  denoteLiteral := denoteLiteral
  Primitive := fun _ _ => Empty
  denotePrimitive := fun primitive => nomatch primitive

/-- Two processors read their corresponding cells and both write the value to
cell zero. Priority therefore selects processor zero. -/
def collisionProgram :
    PriorityCRCW.Program (signature 2) 2 2 .word where
  readAddress := .read .here
  nextRegisters := .read (.there (.there .here))
  writeEnabled := .literal .truth
  writeAddress := .literal (.zeroIndex 1)
  writeValue := .read (.there (.there .here))

def initial : PriorityCRCW.Config 2 2 2 .word where
  memory := fun address =>
    if address.val = 0 then ⟨1, by decide⟩ else ⟨2, by decide⟩
  registers := fun _ => ⟨0, by decide⟩

/-- Both processors target cell zero, and the source semantics selects the
least processor identifier. -/
theorem collision_owner_is_processor_zero :
    collisionProgram.priorityOwnerAt initial (0 : Fin 2) =
      Winner.owner (0 : Fin 2) := by
  native_decide

/-- The compiled target makes the same priority choice before scattering. -/
theorem compiled_collision_owner_is_processor_zero :
    (Compilation.compile collisionProgram).winnerAt
        (Compilation.encode initial) (0 : Fin 2) =
      Winner.owner (0 : Fin 2) := by
  rw [Compilation.winner_compile_correct]
  exact collision_owner_is_processor_zero

/-- Concrete instantiation of the universal finite-run theorem. -/
theorem collisionProgram_three_rounds :
    Compilation.decode
        ((Compilation.compile collisionProgram).run 3
          (Compilation.encode initial)) =
      collisionProgram.run 3 initial :=
  Compilation.compile_run_decode_correct collisionProgram 3 initial

/-- The example compilation has exactly the same scalar syntax size. -/
theorem collisionProgram_no_syntax_blowup :
    (Compilation.compile collisionProgram).nodeCount =
      collisionProgram.nodeCount :=
  Compilation.compile_nodeCount collisionProgram

/-- A conventional two-instruction PRAM program. Every processor first reads
its own cell and then writes the received word to cell zero before halting. -/
def instructionProgram :
    InstructionPRAM.Program (signature 2) 2 2 2 .word where
  idleAddress := ⟨0, by decide⟩
  idleWord := ⟨0, by decide⟩
  instructions := fun pc =>
    if pc.val = 0 then
      .read
        (.read .here)
        (.read (.there (.there .here)))
        (some ⟨1, by decide⟩)
    else
      .write
        (.constant ⟨0, by decide⟩)
        (.read (.there .here))
        (.read (.there .here))
        none

def instructionInitial : InstructionPRAM.Config 2 2 2 2 .word where
  memory := initial.memory
  programCounters := fun _ => some ⟨0, by decide⟩
  registers := fun _ => ⟨0, by decide⟩

/-- Concrete end-to-end instance: two instruction clocks become exactly two
Core bulk rounds. -/
theorem instructionProgram_two_steps :
    InstructionNormalization.decodeCore
        ((InstructionNormalization.compileToCore instructionProgram).run 2
          (InstructionNormalization.encodeCore instructionInitial)) =
      instructionProgram.run 2 instructionInitial :=
  InstructionNormalization.compileToCore_run_decode_correct
    instructionProgram 2 instructionInitial

end MassivelyProof.Example
