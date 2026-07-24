import MassivelyProof.Scalar

namespace MassivelyProof

/-- Executable reduction data. Algebraic laws are separate so execution does
not carry proof fields. -/
structure Reduction (α : Type u) where
  identity : α
  combine : α → α → α

/-- Laws required by a sequential or tree-shaped reduction. -/
structure LawfulReduction {α : Type u} (reduction : Reduction α) : Prop where
  associative : ∀ a b c,
    reduction.combine (reduction.combine a b) c =
      reduction.combine a (reduction.combine b c)
  leftIdentity : ∀ a, reduction.combine reduction.identity a = a
  rightIdentity : ∀ a, reduction.combine a reduction.identity = a

/-- Commutativity makes a parallel reduction independent of proposal order. -/
structure LawfulCommutativeReduction {α : Type u}
    (reduction : Reduction α) : Prop extends LawfulReduction reduction where
  commutative : ∀ a b, reduction.combine a b = reduction.combine b a

/-- A processor owner plus one sentinel value meaning "no writer".

The sentinel is `processors`, while a real owner has value strictly below
`processors`. Keeping only the owner identifier in the reduction allows
arbitrary word values to be fetched after conflict resolution. -/
abbrev Winner (processors : Nat) := Fin (processors + 1)

namespace Winner

/-- No processor proposed a write. -/
def none (processors : Nat) : Winner processors := Fin.last processors

/-- Embed a real processor identifier below the sentinel. -/
def owner {processors : Nat} (processor : Fin processors) : Winner processors :=
  processor.castSucc

/-- Priority conflict resolution: the least processor identifier wins. -/
def combine {processors : Nat}
    (left right : Winner processors) : Winner processors :=
  ⟨Nat.min left.val right.val,
    Nat.lt_of_le_of_lt (Nat.min_le_left _ _) left.isLt⟩

/-- Executable winner reduction. -/
def reduction (processors : Nat) : Reduction (Winner processors) where
  identity := none processors
  combine := combine

/-- Decode the sentinel representation. -/
def toOption {processors : Nat}
    (winner : Winner processors) : Option (Fin processors) :=
  if bounded : winner.val < processors then
    some ⟨winner.val, bounded⟩
  else
    Option.none

@[simp]
theorem toOption_none (processors : Nat) :
    toOption (none processors) = Option.none := by
  simp [toOption, none]

@[simp]
theorem toOption_owner {processors : Nat} (processor : Fin processors) :
    toOption (owner processor) = some processor := by
  simp [toOption, owner]

/-- Minimum with the sentinel leaves every valid winner unchanged. -/
theorem combine_none_right {processors : Nat} (winner : Winner processors) :
    combine winner (none processors) = winner := by
  apply Fin.ext
  simp [combine, none, Nat.min_eq_left (Nat.le_of_lt_succ winner.isLt)]

/-- The winner reduction is lawful for every processor count, including zero.
This is the algebraic reason a backend may reorder colliding proposals. -/
theorem reduction_lawful (processors : Nat) :
    LawfulCommutativeReduction (reduction processors) where
  associative := by
    intro left middle right
    apply Fin.ext
    simp [reduction, combine, Nat.min_assoc]
  leftIdentity := by
    intro winner
    apply Fin.ext
    simp [reduction, combine, none,
      Nat.min_eq_right (Nat.le_of_lt_succ winner.isLt)]
  rightIdentity := combine_none_right
  commutative := by
    intro left right
    apply Fin.ext
    simp [reduction, combine, Nat.min_comm]

end Winner

/-- A fold whose element actions commute is invariant under list permutation.
This lemma is independent of PRAM and Core and is reused by the scheduling
theorem. -/
theorem foldl_eq_of_perm
    {Item : Type u} {Accumulator : Type v}
    (step : Accumulator → Item → Accumulator)
    (commute : ∀ accumulator left right,
      step (step accumulator left) right =
        step (step accumulator right) left)
    {left right : List Item}
    (permutation : left.Perm right)
    (initial : Accumulator) :
    left.foldl step initial = right.foldl step initial := by
  induction permutation generalizing initial with
  | nil => rfl
  | cons item permutation induction =>
      simp only [List.foldl]
      exact induction _
  | swap left right tail =>
      simp only [List.foldl]
      rw [commute]
  | trans first second firstInduction secondInduction =>
      exact (firstInduction initial).trans (secondInduction initial)

end MassivelyProof
