//! Weighted one-mode graph projection.
//!
//! Every ordered pair of distinct neighbors of an input vertex becomes a
//! directed projected edge. Its weight is the number of input vertices that
//! contain that neighbor pair. Work and output capacity are intrinsically
//! proportional to the sum of squared row degrees.

use cubecl::prelude::*;
use massively::{Executor, op::ExpandOp, seg::Segment, vector};

use super::common::{self, DeviceCsr, DeviceWeightedCsr};

struct NeighborPairs;

#[cubecl::cube]
impl ExpandOp<Segment<u32>> for NeighborPairs {
    type Output = (u32, u32);

    fn count(row: Segment<u32>) -> u32 {
        if row.len() < 2u32 {
            0u32
        } else {
            row.len() * (row.len() - 1u32)
        }
    }

    fn generate(row: Segment<u32>, local_index: u32) -> Self::Output {
        let width = row.len() - 1u32;
        let left = local_index / width;
        let compact_right = local_index % width;
        let right = if compact_right >= left {
            compact_right + 1u32
        } else {
            compact_right
        };
        (row.at(left), row.at(right))
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceWeightedCsr<R, u32>> {
    let pairs = vector::flat_map(
        exec,
        graph
            .segmentation()
            .segments(graph.destinations().slice(..))?,
        NeighborPairs,
    )?;
    let (sources, destinations) = massively::MStorage::into_columns(pairs);
    let weights = common::filled(exec, sources.len(), 1u32)?;
    common::weighted_csr_from_edges(
        exec,
        graph.vertex_count(),
        sources.slice(..),
        destinations.slice(..),
        weights.slice(..),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_endpoints_share_the_middle_vertex() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 1, 3, 4], vec![1, 0, 2, 1]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let projection = solve(&exec, &graph).unwrap();
        assert_eq!(
            exec.to_host(&projection.graph().offsets()).unwrap(),
            vec![0, 1, 1, 2]
        );
        assert_eq!(
            exec.to_host(projection.graph().destinations()).unwrap(),
            vec![2, 0]
        );
        assert_eq!(exec.to_host(projection.weights()).unwrap(), vec![1, 1]);
    }
}
