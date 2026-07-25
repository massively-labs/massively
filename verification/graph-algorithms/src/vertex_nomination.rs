//! Multi-source weighted vertex nomination.
//!
//! Vertices are nominated by their shortest distance from any seed. The
//! returned distance vector can be ranked or thresholded by the caller.

use cubecl::prelude::*;
use massively::{DeviceSlice, DeviceVec, Executor, lazy, op::UnaryOp, vector, zip2};

use super::common::{self, DeviceWeightedCsr};

pub const INF: u32 = 1_000_000_000;

struct AddDistance;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddDistance {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 >= INF {
            INF
        } else {
            u32::min(input.0 + input.1, INF)
        }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
    sources: DeviceSlice<u32>,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.graph().vertex_count();
    let distance = common::filled(exec, n, INF)?;
    if sources.len() == 0 {
        return Ok(distance);
    }

    let mut frontier = vector::sort(exec, sources, common::LessU32)?;
    frontier = common::materialize_exact(
        exec,
        vector::unique(exec, frontier.slice(..), common::EqualU32)?,
    )?;
    vector::scatter(
        exec,
        lazy::constant(0u32).take(frontier.len()),
        common::indices(frontier.slice(..)),
        distance.slice_mut(..),
    )?;

    while frontier.len() != 0 {
        let edges = common::expand_rows(exec, graph.graph(), frontier.slice(..))?;
        let proposals = vector::map(
            exec,
            zip2(
                lazy::permute(distance.slice(..), edges.sources().slice(..)),
                lazy::permute(graph.weights().slice(..), edges.edge_ids().slice(..)),
            ),
            AddDistance,
        )?;
        frontier = common::relax_min(
            exec,
            edges.destinations().slice(..),
            proposals.slice(..),
            INF,
            &distance,
        )?;
    }

    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn chooses_the_nearest_of_multiple_seeds() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceWeightedCsr::<_, u32>::from_host_parts(
            &exec,
            &common::path_graph(),
            &[1, 1, 2, 2, 3, 3],
        )
        .unwrap();
        let sources = exec.to_device(&[0u32, 3]);
        let distance = solve(&exec, &graph, sources.slice(..)).unwrap();
        assert_eq!(exec.to_host(&distance).unwrap(), vec![0, 1, 3, 0]);
    }
}
