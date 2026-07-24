//! Exact k-core decomposition with resident degree and removal state.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, lazy,
    op::{BinaryPredicateOp, UnaryOp},
    vector, zip2,
};

use super::common::{self, DeviceCsr};

struct CandidateLess;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for CandidateLess {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.1 != rhs.1 {
            lhs.1 < rhs.1
        } else {
            lhs.0 < rhs.0
        })
    }
}

struct DecrementActive;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for DecrementActive {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.1 == 0 {
            input.0.saturating_sub(1)
        } else {
            input.0
        }
    }
}

fn read_u32<R: Runtime>(
    exec: &Executor<R>,
    values: &DeviceVec<R, u32>,
    index: u32,
) -> common::Result<u32> {
    Ok(exec.to_host(&values.slice(index..index + 1))?[0])
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.vertex_count() as usize;
    let current_degree = common::resident_degrees(exec, graph)?;
    let removed = common::filled(exec, n, 0u32)?;
    let core = common::filled(exec, n, 0u32)?;
    let mut running_core = 0u32;

    for _ in 0..n {
        let vertex = vector::min_element(
            exec,
            zip2(current_degree.slice(..), removed.slice(..)),
            CandidateLess,
        )?
        .expect("the active vertex set is nonempty");
        let degree = read_u32(exec, &current_degree, vertex)?;
        running_core = running_core.max(degree);

        vector::scatter(
            exec,
            lazy::constant(running_core).take(1),
            lazy::constant(vertex).take(1),
            core.slice_mut(..),
        )?;
        vector::scatter(
            exec,
            lazy::constant(1u32).take(1),
            lazy::constant(vertex).take(1),
            removed.slice_mut(..),
        )?;

        let offsets = exec.to_host(&graph.offsets().slice(vertex..vertex + 2))?;
        let neighbors = graph.destinations().slice(offsets[0]..offsets[1]);
        let neighbor_degree =
            lazy::permute(current_degree.slice(..), common::indices(neighbors.clone()));
        let neighbor_removed = lazy::permute(removed.slice(..), common::indices(neighbors.clone()));
        let updated = vector::map(
            exec,
            zip2(neighbor_degree, neighbor_removed),
            DecrementActive,
        )?;
        vector::scatter(
            exec,
            updated.slice(..),
            common::indices(neighbors),
            current_degree.slice_mut(..),
        )?;
    }

    Ok(core)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn two_triangles_share_a_two_core() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        let output = solve(&exec, &graph).unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![2, 2, 2, 2]);
    }
}
