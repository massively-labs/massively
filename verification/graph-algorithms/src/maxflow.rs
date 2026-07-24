//! Integral maximum flow by GPU-resident residual BFS augmentations.
//!
//! Parallel input arcs are coalesced and every reverse residual arc is made
//! explicit. BFS frontier expansion, eligibility filtering, parent selection,
//! bottleneck reduction, and residual updates use Massively primitives; the
//! host controls augmentation rounds and reconstructs one parent chain.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{BinaryPredicateOp, UnaryOp},
    vector, zip2,
};

use super::common::{self, DeviceCsr, DeviceWeightedCsr};

pub struct Flow<R: Runtime> {
    value: u32,
    residual: DeviceWeightedCsr<R, u32>,
}

impl<R: Runtime> Flow<R> {
    pub const fn value(&self) -> u32 {
        self.value
    }

    pub const fn residual(&self) -> &DeviceWeightedCsr<R, u32> {
        &self.residual
    }
}

struct Eligible;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Eligible {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != 0u32 && input.1 == 0u32 {
            1u32
        } else {
            0u32
        }
    }
}

struct SameDestination;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for SameDestination {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 == rhs.0)
    }
}

struct Subtract;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Subtract {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 - input.1
    }
}

struct Add;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Add {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

fn augmenting_path<R: Runtime>(
    exec: &Executor<R>,
    topology: &DeviceCsr<R>,
    residual: &DeviceVec<R, u32>,
    edge_sources_host: &[u32],
    source: u32,
    sink: u32,
) -> common::Result<Option<Vec<u32>>> {
    let n = topology.vertex_count();
    let visited = common::filled(exec, n as usize, 0u32)?;
    let parent_edge = common::filled(exec, n as usize, u32::MAX)?;
    vector::scatter(
        exec,
        lazy::constant(1u32).take(1),
        common::indices(lazy::constant(source).take(1)),
        visited.slice_mut(..),
    )?;
    let mut frontier = common::filled(exec, 1, source)?;
    let mut reached = false;

    while frontier.len() != 0 {
        let edges = common::expand_rows(exec, topology, frontier.slice(..))?;
        if edges.destinations().len() == 0 {
            break;
        }
        let capacity = vector::gather(
            exec,
            residual.slice(..),
            common::indices(edges.edge_ids().slice(..)),
        )?;
        let destination_visited = vector::gather(
            exec,
            visited.slice(..),
            common::indices(edges.destinations().slice(..)),
        )?;
        let eligible = vector::map(
            exec,
            zip2(capacity.slice(..), destination_visited.slice(..)),
            Eligible,
        )?;
        let candidates = common::materialize_exact(
            exec,
            vector::copy_where(
                exec,
                zip2(edges.destinations().slice(..), edges.edge_ids().slice(..)),
                common::stencil(eligible.slice(..)),
            )?,
        )?;
        if candidates.len()? == 0 {
            break;
        }
        let candidates = vector::sort(exec, candidates.slice(..), common::EdgePairLess)?;
        let candidates = common::materialize_exact(
            exec,
            vector::unique(exec, candidates.slice(..), SameDestination)?,
        )?;
        let (destinations, edge_ids) = MStorage::into_columns(candidates);
        vector::scatter(
            exec,
            edge_ids.slice(..),
            common::indices(destinations.slice(..)),
            parent_edge.slice_mut(..),
        )?;
        vector::scatter(
            exec,
            lazy::constant(1u32).take(destinations.len()),
            common::indices(destinations.slice(..)),
            visited.slice_mut(..),
        )?;
        reached = exec.to_host(&visited.slice(sink..sink + 1))?[0] != 0;
        frontier = destinations;
        if reached {
            break;
        }
    }

    if !reached {
        return Ok(None);
    }
    let parents = exec.to_host(&parent_edge)?;
    let mut path = Vec::new();
    let mut vertex = sink;
    while vertex != source {
        let edge = parents[vertex as usize];
        assert!(edge != u32::MAX, "reached sink has no BFS parent");
        path.push(edge);
        vertex = edge_sources_host[edge as usize];
        assert!(
            path.len() <= n as usize,
            "BFS parent chain contains a cycle"
        );
    }
    Ok(Some(path))
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
    source: u32,
    sink: u32,
) -> common::Result<Flow<R>> {
    let n = graph.graph().vertex_count();
    assert!(source < n);
    assert!(sink < n);
    assert!(source != sink);
    let original_sources = graph.graph().segmentation().segment_ids(exec)?;
    let original_count = original_sources.len();
    let residual_count = original_count
        .checked_mul(2)
        .expect("residual edge capacity exceeds u32");
    let residual_sources = exec.alloc::<u32>(residual_count as usize);
    let residual_destinations = exec.alloc::<u32>(residual_count as usize);
    let residual_capacities = common::filled(exec, residual_count as usize, 0u32)?;
    vector::copy(
        exec,
        original_sources.slice(..),
        residual_sources.slice_mut(..original_count),
    )?;
    vector::copy(
        exec,
        graph.graph().destinations().slice(..),
        residual_destinations.slice_mut(..original_count),
    )?;
    vector::copy(
        exec,
        graph.graph().destinations().slice(..),
        residual_sources.slice_mut(original_count..),
    )?;
    vector::copy(
        exec,
        original_sources.slice(..),
        residual_destinations.slice_mut(original_count..),
    )?;
    vector::copy(
        exec,
        graph.weights().slice(..),
        residual_capacities.slice_mut(..original_count),
    )?;
    let residual_graph = common::weighted_csr_from_edges(
        exec,
        n,
        residual_sources.slice(..),
        residual_destinations.slice(..),
        residual_capacities.slice(..),
    )?;
    let (topology, residual) = residual_graph.into_parts();
    let edge_sources = topology.segmentation().segment_ids(exec)?;
    let reverse_edges = vector::lower_bound(
        exec,
        zip2(edge_sources.slice(..), topology.destinations().slice(..)),
        zip2(topology.destinations().slice(..), edge_sources.slice(..)),
        common::EdgePairLess,
    )?;
    let edge_sources_host = exec.to_host(&edge_sources)?;

    let mut value = 0u32;
    while let Some(path) =
        augmenting_path(exec, &topology, &residual, &edge_sources_host, source, sink)?
    {
        let path = exec.to_device(&path);
        let capacity = vector::gather(exec, residual.slice(..), common::indices(path.slice(..)))?;
        let bottleneck = vector::reduce(exec, capacity.slice(..), u32::MAX, common::MinU32)?;
        assert!(bottleneck != 0);
        let forward = vector::map(
            exec,
            zip2(
                capacity.slice(..),
                lazy::constant(bottleneck).take(capacity.len()),
            ),
            Subtract,
        )?;
        vector::scatter(
            exec,
            forward.slice(..),
            common::indices(path.slice(..)),
            residual.slice_mut(..),
        )?;
        let reverse_path = vector::gather(
            exec,
            reverse_edges.slice(..),
            common::indices(path.slice(..)),
        )?;
        let reverse_capacity = vector::gather(
            exec,
            residual.slice(..),
            common::indices(reverse_path.slice(..)),
        )?;
        let reverse = vector::map(
            exec,
            zip2(
                reverse_capacity.slice(..),
                lazy::constant(bottleneck).take(reverse_capacity.len()),
            ),
            Add,
        )?;
        vector::scatter(
            exec,
            reverse.slice(..),
            common::indices(reverse_path.slice(..)),
            residual.slice_mut(..),
        )?;
        value = value
            .checked_add(bottleneck)
            .expect("maximum flow exceeds u32");
    }

    Ok(Flow {
        value,
        residual: DeviceWeightedCsr::from_parts(topology, residual)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn finds_multiple_augmenting_paths() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 2, 4, 5, 5], vec![1, 2, 2, 3, 3]);
        let graph =
            DeviceWeightedCsr::<_, u32>::from_host_parts(&exec, &host, &[3, 2, 1, 2, 3]).unwrap();
        let flow = solve(&exec, &graph, 0, 3).unwrap();
        assert_eq!(flow.value(), 5);
    }
}
