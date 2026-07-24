use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op::*, vector::*, *};

type Triple = (u32, u32, u32);

fn exec() -> Executor<WgpuRuntime> {
    Executor::new(WgpuDevice::DefaultDevice)
}

macro_rules! nested_input {
    ($a:expr, $b:expr, $c:expr) => {
        zip2($a.slice(..), zip2($b.slice(..), $c.slice(..)))
    };
}

struct SumTriple;

#[cubecl::cube]
impl ReductionOp<Triple> for SumTriple {
    fn apply(lhs: Triple, rhs: Triple) -> Triple {
        (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    }
}

struct LessTriple;

#[cubecl::cube]
impl BinaryPredicateOp<Triple> for LessTriple {
    fn apply(lhs: Triple, rhs: Triple) -> MFlag {
        massively::flag::from_bool(lhs.0 < rhs.0)
    }
}

struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

struct AddOne;

#[cubecl::cube]
impl UnaryOp<Triple> for AddOne {
    type Output = Triple;

    fn apply(input: Triple) -> Triple {
        (input.0 + 1, input.1 + 1, input.2 + 1)
    }
}

fn stencil<Input>(input: Input) -> Input {
    input
}

#[test]
fn inclusive_scan_treats_nested_zip_calls_as_flat_rows() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let output = inclusive_scan(&exec, nested_input!(a, b, c), SumTriple).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 3, 6]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 30, 60]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 300, 600]);
}

#[test]
fn sort_returns_flat_owned_columns() {
    let exec = exec();
    let a = exec.to_device(&[3_u32, 1, 2]);
    let b = exec.to_device(&[30_u32, 10, 20]);
    let c = exec.to_device(&[300_u32, 100, 200]);
    let output = sort(&exec, nested_input!(a, b, c), LessTriple).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 200, 300]);
}

#[test]
fn copy_where_returns_flat_owned_columns() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3, 4]);
    let b = exec.to_device(&[10_u32, 20, 30, 40]);
    let c = exec.to_device(&[100_u32, 200, 300, 400]);
    let flags = exec.to_device(&[0_u32, 1, 1, 0]);
    let output = copy_where(&exec, nested_input!(a, b, c), stencil(flags.slice(..))).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![2, 3]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![200, 300]);
}

#[test]
fn gather_returns_flat_owned_columns() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let indices = exec.to_device(&[2_u32, 0]);
    let output = gather(&exec, nested_input!(a, b, c), indices.slice(..)).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![3, 1]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![30, 10]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![300, 100]);
}

#[test]
fn merge_returns_flat_owned_columns() {
    let exec = exec();
    let la = exec.to_device(&[1_u32, 3]);
    let lb = exec.to_device(&[10_u32, 30]);
    let lc = exec.to_device(&[100_u32, 300]);
    let ra = exec.to_device(&[2_u32, 4]);
    let rb = exec.to_device(&[20_u32, 40]);
    let rc = exec.to_device(&[200_u32, 400]);
    let output = merge(
        &exec,
        nested_input!(la, lb, lc),
        nested_input!(ra, rb, rc),
        LessTriple,
    )
    .unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30, 40]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 200, 300, 400]);
}

#[test]
fn set_operations_preserve_flat_row_payloads() {
    let exec = exec();
    let la = exec.to_device(&[1_u32, 2, 2, 4]);
    let lb = exec.to_device(&[10_u32, 20, 21, 40]);
    let lc = exec.to_device(&[100_u32, 200, 210, 400]);
    let ra = exec.to_device(&[2_u32, 2, 2, 3, 4, 4]);
    let rb = exec.to_device(&[200_u32, 201, 202, 300, 400, 401]);
    let rc = exec.to_device(&[2_000_u32, 2_010, 2_020, 3_000, 4_000, 4_010]);

    let union = set_union(
        &exec,
        nested_input!(la, lb, lc),
        nested_input!(ra, rb, rc),
        LessTriple,
    )
    .unwrap();
    let (a, b, c) = MStorage::into_columns(union);
    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 2, 2, 2, 3, 4, 4]);
    assert_eq!(
        exec.to_host(&b).unwrap(),
        vec![10, 20, 21, 202, 300, 40, 401]
    );
    assert_eq!(
        exec.to_host(&c).unwrap(),
        vec![100, 200, 210, 2_020, 3_000, 400, 4_010]
    );

    let intersection = set_intersection(
        &exec,
        nested_input!(la, lb, lc),
        nested_input!(ra, rb, rc),
        LessTriple,
    )
    .unwrap();
    let (a, b, c) = MStorage::into_columns(intersection);
    assert_eq!(exec.to_host(&a).unwrap(), vec![2, 2, 4]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![20, 21, 40]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![200, 210, 400]);

    let difference = set_difference(
        &exec,
        nested_input!(la, lb, lc),
        nested_input!(ra, rb, rc),
        LessTriple,
    )
    .unwrap();
    let (a, b, c) = MStorage::into_columns(difference);
    assert_eq!(exec.to_host(&a).unwrap(), vec![1]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100]);
}

#[test]
fn sort_by_key_returns_flat_value_columns() {
    let exec = exec();
    let keys = exec.to_device(&[3_u32, 1, 2]);
    let a = exec.to_device(&[30_u32, 10, 20]);
    let b = exec.to_device(&[300_u32, 100, 200]);
    let c = exec.to_device(&[3_000_u32, 1_000, 2_000]);
    let output = sort_by_key(&exec, keys.slice(..), nested_input!(a, b, c), LessU32).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![100, 200, 300]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![1_000, 2_000, 3_000]);
}

#[test]
fn transform_where_writes_a_flat_operation_result() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let flags = exec.to_device(&[1_u32, 0, 1]);
    let oa = exec.to_device(&[90_u32, 90, 90]);
    let ob = exec.to_device(&[80_u32, 80, 80]);
    let oc = exec.to_device(&[70_u32, 70, 70]);
    transform_where(
        &exec,
        nested_input!(a, b, c),
        AddOne,
        stencil(flags.slice(..)),
        zip3(oa.slice_mut(..), ob.slice_mut(..), oc.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&oa).unwrap(), vec![2, 90, 4]);
    assert_eq!(exec.to_host(&ob).unwrap(), vec![11, 80, 31]);
    assert_eq!(exec.to_host(&oc).unwrap(), vec![101, 70, 301]);
}
