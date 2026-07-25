use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MAlloc, MFlag, MIndex, MStorage, Scalar, op::BinaryPredicateOp, op::PredicateOp,
    op::ReductionOp, op::UnaryOp,
};
use oracle::op;
use proptest::prelude::*;

pub const CASES: u32 = 6;

const ORACLE_LENGTHS: [usize; 18] = [
    0, 1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1_023, 1_024, 1_025,
];

pub fn oracle_vec<S>(element: S) -> impl Strategy<Value = Vec<S::Value>>
where
    S: Strategy + Clone,
    S::Value: std::fmt::Debug,
{
    prop::sample::select(&ORACLE_LENGTHS)
        .prop_flat_map(move |len| prop::collection::vec(element.clone(), len))
}

pub struct AddOne;
pub struct Sum;
pub struct Even;
pub struct Equal;
pub struct Less;

#[cubecl::cube]
impl UnaryOp<u32> for AddOne {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input + 1u32
    }
}

impl op::UnaryOp<u32> for AddOne {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input + 1
    }
}

#[cubecl::cube]
impl ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

impl op::ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for Even {
    fn apply(input: u32) -> MFlag {
        massively::flag::from_bool(input % 2u32 == 0u32)
    }
}

impl op::PredicateOp<u32> for Even {
    fn apply(input: u32) -> bool {
        input % 2 == 0
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

impl op::BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub fn as_stencil<Input>(input: Input) -> Input {
    input
}

pub fn as_indices<Input>(input: Input) -> Input {
    input
}

pub fn expected_flag(value: bool) -> MFlag {
    massively::flag::from_bool(value)
}

pub fn expected_flags(values: Vec<bool>) -> Vec<MFlag> {
    values.into_iter().map(expected_flag).collect()
}

pub fn exec() -> Executor<WgpuRuntime> {
    Executor::new(WgpuDevice::DefaultDevice)
}

pub fn exact<Storage>(_exec: &Executor<WgpuRuntime>, storage: Storage) -> Storage
where
    Storage: MStorage<WgpuRuntime>,
{
    storage.len().unwrap();
    storage
}

pub fn exact_pair<Left, Right>(
    _exec: &Executor<WgpuRuntime>,
    (left, right): (Left, Right),
) -> (Left, Right)
where
    Left: MStorage<WgpuRuntime>,
    Right: MStorage<WgpuRuntime>,
{
    let len = left.len().unwrap();
    assert_eq!(right.len().unwrap(), len);
    (left, right)
}

pub fn read_value<T>(exec: &Executor<WgpuRuntime>, value: Scalar<WgpuRuntime, T>) -> T
where
    T: MAlloc<WgpuRuntime>,
{
    value.read(exec).unwrap()
}

pub fn read_optional_index(
    exec: &Executor<WgpuRuntime>,
    value: Scalar<WgpuRuntime, Option<MIndex>>,
) -> Option<usize> {
    value.read(exec).unwrap().map(|index| index as usize)
}

pub fn read_optional_index_pair(
    exec: &Executor<WgpuRuntime>,
    value: Scalar<WgpuRuntime, Option<(MIndex, MIndex)>>,
) -> Option<(usize, usize)> {
    value
        .read(exec)
        .unwrap()
        .map(|(first, second)| (first as usize, second as usize))
}

pub fn lazify<Input>(input: Input) -> Input
where
    Input: massively::MIter<WgpuRuntime>,
{
    input
}

pub fn flags_for(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| u32::from((index as u32 + value) % 3 != 0))
        .collect()
}

pub fn indices_for(len: usize) -> Vec<u32> {
    (0..len).rev().map(|index| index as u32).collect()
}
