//! Frontier SSSP expressed as weighted destination-state relaxation.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, lazy, op::UnaryOp, vector, zip2};

use super::common::{self, DeviceWeightedCsr};

const INF: u32 = 1_000_000_000;

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
    source: u32,
) -> common::Result<DeviceVec<R, u32>> {
    assert!(source < graph.graph().vertex_count());
    let distance = common::filled(exec, graph.graph().vertex_count(), INF)?;
    let mut frontier = common::filled(exec, 1, source)?;
    let zero = common::filled(exec, 1, 0u32)?;
    vector::scatter(
        exec,
        zero.slice(..),
        common::indices(frontier.slice(..)),
        distance.slice_mut(..),
    )?;
    let infinity = INF;

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
    fn weighted_path_distances_accumulate() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceWeightedCsr::<_, u32>::from_host_parts(
            &exec,
            &common::path_graph(),
            &[1, 1, 2, 2, 3, 3],
        )
        .unwrap();
        assert_eq!(
            exec.to_host(&solve(&exec, &graph, 0).unwrap()).unwrap(),
            vec![0, 1, 3, 6]
        );
    }
}
