//! Connected components by monotone minimum-label propagation.
//!
//! The input is interpreted as an undirected graph (each undirected edge is
//! represented by both CSR directions).  The returned label of every vertex
//! is the smallest vertex identifier in its connected component.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, lazy, op::Identity, vector};

use super::common::{self, DeviceCsr};

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.vertex_count();
    let labels = vector::map(exec, common::counting_u32(0, n as usize), Identity)?;
    let mut frontier = vector::map(exec, common::counting_u32(0, n as usize), Identity)?;
    let infinity = u32::MAX;

    while frontier.len() != 0 {
        let edges = common::expand_rows(exec, graph, frontier.slice(..))?;
        let proposals = vector::map(
            exec,
            lazy::permute(labels.slice(..), edges.sources().slice(..)),
            Identity,
        )?;
        frontier = common::relax_min(
            exec,
            edges.destinations().slice(..),
            proposals.slice(..),
            infinity,
            &labels,
        )?;
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn labels_disconnected_components_by_their_minimum_vertex() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 1, 2, 3, 4, 4], vec![1, 0, 3, 2]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let labels = solve(&exec, &graph).unwrap();
        assert_eq!(exec.to_host(&labels).unwrap(), vec![0, 0, 2, 2, 4]);
    }
}
