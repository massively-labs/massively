//! Breadth-first search as destination-state relaxation.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, lazy, op::UnaryOp, vector};

use super::common::{self, DeviceCsr};

struct AddOne;

#[cubecl::cube]
impl UnaryOp<u32> for AddOne {
    type Output = u32;

    fn apply(distance: u32) -> u32 {
        distance + 1u32
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    source: u32,
) -> common::Result<DeviceVec<R, u32>> {
    assert!(source < graph.vertex_count());
    let distance = common::filled(exec, graph.vertex_count(), u32::MAX)?;
    let mut frontier = common::filled(exec, 1, source)?;
    let zero = common::filled(exec, 1, 0u32)?;
    vector::scatter(
        exec,
        zero.slice(..),
        common::indices(frontier.slice(..)),
        distance.slice_mut(..),
    )?;
    let infinity = u32::MAX;

    while frontier.len() != 0 {
        let edges = common::expand_rows(exec, graph, frontier.slice(..))?;
        let proposals = vector::map(
            exec,
            lazy::permute(distance.slice(..), edges.sources().slice(..)),
            AddOne,
        )?;
        frontier = common::relax_min(
            exec,
            edges.destinations().slice(..),
            proposals.slice(..),
            infinity,
            &distance,
        )?;
    }

    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_levels() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::path_graph()).unwrap();
        assert_eq!(
            exec.to_host(&solve(&exec, &graph, 0).unwrap()).unwrap(),
            vec![0, 1, 2, 3]
        );
    }
}
