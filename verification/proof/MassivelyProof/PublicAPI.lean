import MassivelyProof.Core

namespace MassivelyProof.PublicAPI

open MassivelyProof

/-!
This file models the small public-API basis used by the completeness compiler:

* `map` computes pull addresses, mapped registers, and write rows;
* `gather` reads the old materialized memory;
* `copy_where` compacts enabled write rows;
* `sort` orders rows by `(destination, owner)`;
* `unique_by_key` retains the first row for each destination;
* `copy` preserves untouched memory;
* `scatter` commits rows whose destinations are distinct.

The definitions below are an independent denotational specification of those
public operations.  They are not definitions of the existing Core winner fold.
-/

/-- A materialized API-boundary state. -/
abbrev Config := MassivelyCore.Config

/-- A write row carried between public sequence algorithms.  Unlike the Core
proposal, this row contains the value that the final public scatter consumes. -/
structure WriteRow (wordWidth processors memorySize : Nat) where
  destination : Fin memorySize
  owner : Fin processors
  value : Word wordWidth

namespace Basis

/-- Denotation of public `map`. -/
def map {ι : Type} (indices : List ι) (mapper : ι → α) : List α :=
  indices.map mapper

/-- Denotation of public `gather`. -/
def gather (memory : Fin memorySize → α) (indices : List (Fin memorySize)) : List α :=
  indices.map memory

/-- Denotation of public `copy_where`; selected rows retain their input order. -/
def copyWhere (rows : List α) (selected : α → Bool) : List α :=
  rows.filter selected

/-- Public sort specification.  It orders the selected processor rows by
`(destination, owner)`.  Iterating finite destination and owner domains is an
extensional specification of that order, not an implementation algorithm. -/
def sortWriteRows
    (rowAt : Fin processors → WriteRow wordWidth processors memorySize)
    (selected : Fin processors → Bool) :
    Fin memorySize → List (WriteRow wordWidth processors memorySize) :=
  fun destination =>
    ((List.finRange processors).filter fun owner =>
      selected owner && (rowAt owner).destination = destination).map rowAt

/-- Public `unique_by_key` specification for destination-grouped rows.  It
retains the first row for every destination.  The pointwise definition exposes
exactly the first-occurrence contract used by the correctness proof. -/
def uniqueByDestination
    (runs : Fin memorySize → List (WriteRow wordWidth processors memorySize)) :
    Fin memorySize → Option (WriteRow wordWidth processors memorySize) :=
  fun destination => (runs destination).head?

/-- Denotation of public `copy`. -/
def copy (memory : Fin memorySize → α) : Fin memorySize → α := memory

/-- A schedule-independent specification of collision-free public `scatter`.
`unique_by_key` establishes the distinct-destination precondition of the actual
parallel scatter. -/
def scatterDistinct
    (memory : Fin memorySize → Word wordWidth)
    (rows : Fin memorySize → Option (WriteRow wordWidth processors memorySize)) :
    Fin memorySize → Word wordWidth := fun destination =>
  match rows destination with
  | some row => row.value
  | none => memory destination

end Basis

/-- Public-API program obtained from one normalized Core round.  The syntax
fields are scalar callbacks used by public `map`; the routing skeleton is
fixed by `step` below. -/
structure Program
    {wordWidth : Nat}
    (signature : Signature wordWidth)
    (processors memorySize : Nat)
    (registerType : ValueType) where
  pullAddress :
    Term signature (MassivelyCore.PullContext processors registerType)
      (.index memorySize)
  mapRegisters :
    Term signature (MassivelyCore.MapContext processors registerType)
      registerType
  selectWrite :
    Term signature (MassivelyCore.MapContext processors registerType) .boolean
  pushAddress :
    Term signature (MassivelyCore.MapContext processors registerType)
      (.index memorySize)
  pushValue :
    Term signature (MassivelyCore.MapContext processors registerType) .word

namespace Program

def processorRows (processors : Nat) : List (Fin processors) :=
  List.finRange processors

def pullEnvironment
    (_program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    Environment wordWidth
      (MassivelyCore.PullContext processors registerType) :=
  .cons processor (.cons (configuration.registers processor) .nil)

def pulledAddressAt
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Fin memorySize :=
  Term.evaluate (signature := signature)
    (program.pullEnvironment configuration processor) program.pullAddress

def pulledAddresses
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    List (Fin memorySize) :=
  Basis.map (processorRows processors)
    (program.pulledAddressAt configuration)

def pulledValues
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    List (Word wordWidth) :=
  Basis.gather configuration.memory (program.pulledAddresses configuration)

def pulledValueAt
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Word wordWidth :=
  configuration.memory (program.pulledAddressAt configuration processor)

def mapEnvironment
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) :
    Environment wordWidth
      (MassivelyCore.MapContext processors registerType) :=
  .cons processor
    (.cons (configuration.registers processor)
      (.cons (program.pulledValueAt configuration processor) .nil))

def mappedRegistersAt
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : registerType.denote wordWidth :=
  Term.evaluate (signature := signature)
    (program.mapEnvironment configuration processor) program.mapRegisters

def selectedAt
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : Bool :=
  Term.evaluate (signature := signature)
    (program.mapEnvironment configuration processor) program.selectWrite

def writeRowAt
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (processor : Fin processors) : WriteRow wordWidth processors memorySize where
  destination :=
    Term.evaluate (signature := signature)
      (program.mapEnvironment configuration processor) program.pushAddress
  owner := processor
  value := Term.evaluate (signature := signature)
    (program.mapEnvironment configuration processor) program.pushValue

def mappedWriteRows
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    List (WriteRow wordWidth processors memorySize) :=
  Basis.map (processorRows processors) (program.writeRowAt configuration)

def compactedWriteRows
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    List (WriteRow wordWidth processors memorySize) :=
  Basis.copyWhere (program.mappedWriteRows configuration) fun row =>
    program.selectedAt configuration row.owner

def sortedWriteRows
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    Fin memorySize → List (WriteRow wordWidth processors memorySize) :=
  Basis.sortWriteRows (program.writeRowAt configuration)
    (program.selectedAt configuration)

def winningWriteRows
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    Fin memorySize → Option (WriteRow wordWidth processors memorySize) :=
  Basis.uniqueByDestination (program.sortedWriteRows configuration)

def nextMemory
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    Fin memorySize → Word wordWidth :=
  Basis.scatterDistinct (Basis.copy configuration.memory)
    (program.winningWriteRows configuration)

/-- The pointwise public routing pipeline exposed before the final barrier.
This observation is useful both for proofs and for implementations that lower
the basis as several public API calls. -/
def winningOwnerAt
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType)
    (destination : Fin memorySize) : Option (Fin processors) :=
  ((List.finRange processors).filter fun processor =>
    program.selectedAt configuration processor &&
      (program.writeRowAt configuration processor).destination = destination).head?

/-- One execution of the public basis.  Every stage is named after a public
sequence operation; no Core winner reduction is used in this definition. -/
def step
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType)
    (configuration : Config wordWidth processors memorySize registerType) :
    Config wordWidth processors memorySize registerType where
  memory := program.nextMemory configuration
  registers := program.mappedRegistersAt configuration

def run
    (program : Program (wordWidth := wordWidth) signature processors memorySize registerType) :
    Nat → Config wordWidth processors memorySize registerType →
      Config wordWidth processors memorySize registerType
  | 0, configuration => configuration
  | rounds + 1, configuration => run program rounds (program.step configuration)

end Program

end MassivelyProof.PublicAPI
