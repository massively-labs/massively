import Std

namespace MassivelyProof

/-- Object-language values available to local scalar computation.

`product` is recursive rather than arity-indexed, so the proof describes
arbitrary multi-column rows without generating one language per column count.
Memory words are finite by construction. -/
inductive ValueType where
  | boolean
  | word
  | index (bound : Nat)
  | product (left right : ValueType)
deriving DecidableEq, Repr

namespace ValueType

/-- Mathematical denotation of a scalar type at a fixed word width. -/
def denote (wordWidth : Nat) : ValueType → Type
  | .boolean => Bool
  | .word => Fin (2 ^ wordWidth)
  | .index bound => Fin bound
  | .product left right => left.denote wordWidth × right.denote wordWidth

end ValueType

/-- Finite machine word used by the reference PRAM. -/
abbrev Word (wordWidth : Nat) := ValueType.denote wordWidth .word

/-- Intrinsically bounded bulk index. -/
abbrev Index (bound : Nat) := ValueType.denote 0 (.index bound)

/-- A fixed first-order scalar vocabulary.

Programs contain symbols from the signature, not arbitrary semantic callback
functions. The denotations are used only by the mathematical interpreter. -/
structure Signature (wordWidth : Nat) where
  Literal : ValueType → Type
  denoteLiteral : {output : ValueType} →
    Literal output → output.denote wordWidth
  Primitive : ValueType → ValueType → Type
  denotePrimitive : {input output : ValueType} →
    Primitive input output →
      input.denote wordWidth → output.denote wordWidth

/-- A well-typed de Bruijn reference into a heterogeneous context. -/
inductive Variable : List ValueType → ValueType → Type
  | here : Variable (output :: context) output
  | there : Variable context output → Variable (head :: context) output

/-- A heterogeneous environment matching an object-language context. -/
inductive Environment (wordWidth : Nat) : List ValueType → Type
  | nil : Environment wordWidth []
  | cons : head.denote wordWidth → Environment wordWidth tail →
      Environment wordWidth (head :: tail)

namespace Variable

/-- Total typed lookup. An invalid variable/environment combination cannot be
constructed. -/
def lookup
    {wordWidth : Nat}
    {context : List ValueType}
    {output : ValueType}
    (reference : Variable context output)
    (environment : Environment wordWidth context) :
    output.denote wordWidth :=
  match reference, environment with
  | .here, .cons value _ => value
  | .there rest, .cons _ tail => rest.lookup tail

end Variable

/-- Intrinsically typed, first-order scalar terms.

The language deliberately has no array read, global write, atomic, barrier, or
kernel-launch constructor. Such effects must remain visible in Massively Core.
`let1` represents explicit sharing and prevents normalization from duplicating
its input syntax. -/
inductive Term
    {wordWidth : Nat}
    (signature : Signature wordWidth) :
    List ValueType → ValueType → Type
  | read : Variable context output → Term signature context output
  | literal : signature.Literal output → Term signature context output
  | constant : output.denote wordWidth → Term signature context output
  | pair : Term signature context left → Term signature context right →
      Term signature context (.product left right)
  | first : Term signature context (.product left right) →
      Term signature context left
  | second : Term signature context (.product left right) →
      Term signature context right
  | apply : signature.Primitive input output →
      Term signature context input → Term signature context output
  | ite : Term signature context .boolean →
      Term signature context output → Term signature context output →
      Term signature context output
  | caseIndex : Term signature context (.index bound) →
      (Fin bound → Term signature context output) →
      Term signature context output
  | let1 : Term signature context input → Term signature [input] output →
      Term signature context output

namespace Term

/-- Denotation of a typed scalar term. -/
def evaluate
    {wordWidth : Nat}
    {signature : Signature wordWidth}
    {context : List ValueType}
    (environment : Environment wordWidth context) :
    {output : ValueType} →
      Term signature context output → output.denote wordWidth
  | _, .read reference => reference.lookup environment
  | _, .literal symbol => signature.denoteLiteral symbol
  | _, .constant value => value
  | _, .pair left right =>
      (left.evaluate environment, right.evaluate environment)
  | _, .first productTerm => (productTerm.evaluate environment).1
  | _, .second productTerm => (productTerm.evaluate environment).2
  | _, .apply primitive argument =>
      signature.denotePrimitive primitive (argument.evaluate environment)
  | _, .ite condition whenTrue whenFalse =>
      match condition.evaluate environment with
      | true => whenTrue.evaluate environment
      | false => whenFalse.evaluate environment
  | _, .caseIndex index branches =>
      (branches (index.evaluate environment)).evaluate environment
  | _, .let1 input body =>
      body.evaluate (.cons (input.evaluate environment) .nil)

/-- Capture-avoiding replacement of the free variables of a term.  The body
of `let1` is closed over its singleton bound-variable context, so it needs no
outer-context lifting. -/
def substitute
    {wordWidth : Nat}
    {signature : Signature wordWidth}
    {source target : List ValueType}
    (replacement : ∀ {output},
      Variable source output → Term signature target output) :
    {output : ValueType} →
      Term signature source output → Term signature target output
  | _, .read reference => replacement reference
  | _, .literal symbol => .literal symbol
  | _, .constant value => .constant value
  | _, .pair left right =>
      .pair (left.substitute replacement) (right.substitute replacement)
  | _, .first productTerm => .first (productTerm.substitute replacement)
  | _, .second productTerm => .second (productTerm.substitute replacement)
  | _, .apply primitive argument =>
      .apply primitive (argument.substitute replacement)
  | _, .ite condition whenTrue whenFalse =>
      .ite (condition.substitute replacement)
        (whenTrue.substitute replacement)
        (whenFalse.substitute replacement)
  | _, .caseIndex index branches =>
      .caseIndex (index.substitute replacement)
        (fun branch => (branches branch).substitute replacement)
  | _, .let1 input body =>
      .let1 (input.substitute replacement) body

/-- Substitution preserves evaluation whenever every replacement term denotes
the source variable it replaces. -/
theorem evaluate_substitute
    {wordWidth : Nat}
    {signature : Signature wordWidth}
    {source target : List ValueType}
    (replacement : ∀ {output},
      Variable source output → Term signature target output)
    (sourceEnvironment : Environment wordWidth source)
    (targetEnvironment : Environment wordWidth target)
    (replacementCorrect : ∀ {output}
        (reference : Variable source output),
      (replacement reference).evaluate targetEnvironment =
        reference.lookup sourceEnvironment) :
    ∀ {output} (term : Term signature source output),
      (term.substitute replacement).evaluate targetEnvironment =
        term.evaluate sourceEnvironment := by
  intro output term
  induction term with
  | read reference =>
      simpa [substitute, evaluate] using replacementCorrect reference
  | literal => rfl
  | constant => rfl
  | pair left right leftInduction rightInduction =>
      simp only [substitute, evaluate]
      rw [leftInduction replacement sourceEnvironment replacementCorrect,
        rightInduction replacement sourceEnvironment replacementCorrect]
  | first pair induction =>
      simp only [substitute, evaluate]
      rw [induction replacement sourceEnvironment replacementCorrect]
  | second pair induction =>
      simp only [substitute, evaluate]
      rw [induction replacement sourceEnvironment replacementCorrect]
  | apply primitive argument induction =>
      simp only [substitute, evaluate]
      rw [induction replacement sourceEnvironment replacementCorrect]
  | ite condition whenTrue whenFalse conditionInduction trueInduction falseInduction =>
      simp only [substitute, evaluate]
      rw [conditionInduction replacement sourceEnvironment replacementCorrect]
      cases condition.evaluate sourceEnvironment
      · exact falseInduction replacement sourceEnvironment replacementCorrect
      · exact trueInduction replacement sourceEnvironment replacementCorrect
  | caseIndex index branches indexInduction branchesInduction =>
      simp only [substitute, evaluate]
      rw [indexInduction replacement sourceEnvironment replacementCorrect]
      exact branchesInduction _ replacement sourceEnvironment replacementCorrect
  | let1 input body inputInduction bodyInduction =>
      simp only [substitute, evaluate]
      rw [inputInduction replacement sourceEnvironment replacementCorrect]

/-- Explicit sharing has the denotation of ordinary composition. -/
theorem evaluate_let1
    {wordWidth : Nat}
    {signature : Signature wordWidth}
    {context : List ValueType}
    {input output : ValueType}
    (bound : Term signature context input)
    (body : Term signature [input] output)
    (environment : Environment wordWidth context) :
    (Term.let1 bound body).evaluate environment =
      body.evaluate (.cons (bound.evaluate environment) .nil) := rfl

end Term

end MassivelyProof
