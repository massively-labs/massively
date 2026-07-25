//! Boolean sparse matrix multiplication into resident CSR storage.

use cubecl::prelude::*;
use massively::{
    Executor, lazy,
    op::{BinaryPredicateOp, ReductionOp},
    vector,
};

use super::common::{self, DeviceCsr};

struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    lhs: &DeviceCsr<R>,
    rhs: &DeviceCsr<R>,
) -> common::Result<DeviceCsr<R>> {
    assert_eq!(lhs.vertex_count(), rhs.vertex_count());
    let n = lhs.vertex_count();

    let rhs_degrees = common::resident_degrees(exec, rhs)?;
    let capacity = vector::reduce(
        exec,
        lazy::permute(rhs_degrees.slice(..), lhs.destinations().slice(..)),
        0,
        SumU32,
    )?;
    let destinations = exec.alloc::<u32>(capacity);
    let offsets = exec.alloc::<u32>(n.checked_add(1).expect("offset count exceeds MIndex"));
    let mut output_len = 0u32;

    for vertex in 0..n {
        vector::scatter(
            exec,
            lazy::constant(output_len).take(1),
            common::indices(lazy::constant(vertex).take(1)),
            offsets.slice_mut(..),
        )?;

        let bounds = exec.to_host(&lhs.offsets().slice(vertex..vertex + 2))?;
        let frontier = lhs.destinations().slice(bounds[0]..bounds[1]);
        if frontier.len() == 0 {
            continue;
        }

        let edges = common::expand_rows(exec, rhs, frontier)?;
        if edges.destinations().len() == 0 {
            continue;
        }
        let sorted = vector::sort(exec, edges.destinations().slice(..), LessU32)?;
        let row =
            common::materialize_exact(exec, vector::unique(exec, sorted.slice(..), EqualU32)?)?;
        let row_len = row.len();
        vector::scatter(
            exec,
            row.slice(..),
            common::indices(common::counting_u32(output_len as usize, row_len as usize)),
            destinations.slice_mut(..),
        )?;
        output_len += row_len;
    }

    vector::scatter(
        exec,
        lazy::constant(output_len).take(1),
        common::indices(lazy::constant(n).take(1)),
        offsets.slice_mut(..),
    )?;
    let exact_destinations = exec.alloc::<u32>(output_len);
    vector::copy(
        exec,
        destinations.slice(..output_len),
        exact_destinations.slice_mut(..),
    )?;
    DeviceCsr::from_parts(exec, exact_destinations, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_squared_contains_two_hop_pairs() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::path_graph()).unwrap();
        let output = solve(&exec, &graph, &graph).unwrap();
        assert_eq!(
            exec.to_host(&output.offsets()).unwrap(),
            vec![0, 2, 4, 6, 8]
        );
        assert_eq!(
            exec.to_host(output.destinations()).unwrap(),
            vec![0, 2, 1, 3, 0, 2, 1, 3]
        );
    }
}
