//! Deterministic multilevel Louvain community detection.
//!
//! Each level performs sequential local moves using the standard modularity-gain
//! ordering for an undirected weighted graph, then contracts communities by a
//! traversal, sort, and reduction.  The public unweighted input is interpreted
//! as a symmetric CSR graph with unit edge weights and no self-loops.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{BinaryPredicateOp, Identity, ReductionOp, UnaryOp},
    vector, zip2,
};

use super::common::{self, DeviceCsr, DeviceWeightedCsr};

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

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

struct Different;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Different {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != input.1 { 1u32 } else { 0u32 }
    }
}

struct GainScore;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32, u32)> for GainScore {
    type Output = f32;

    fn apply(input: (u32, u32, u32, u32)) -> f32 {
        input.0 as f32 - input.1 as f32 * input.2 as f32 / input.3 as f32
    }
}

struct CandidateLess;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for CandidateLess {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.0 != rhs.0 {
            lhs.0 < rhs.0
        } else {
            // For equal gain, the smaller community is the maximum item.
            lhs.1 > rhs.1
        })
    }
}

struct PairLess;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for PairLess {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.0 != rhs.0 {
            lhs.0 < rhs.0
        } else {
            lhs.1 < rhs.1
        })
    }
}

struct PairEqual;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for PairEqual {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 == rhs.0 && lhs.1 == rhs.1)
    }
}

fn read_u32<R: Runtime>(
    exec: &Executor<R>,
    values: &DeviceVec<R, u32>,
    index: u32,
) -> common::Result<u32> {
    Ok(exec.to_host(&values.slice(index..index + 1))?[0])
}

fn read_f32<R: Runtime>(
    exec: &Executor<R>,
    values: &DeviceVec<R, f32>,
    index: u32,
) -> common::Result<f32> {
    Ok(exec.to_host(&values.slice(index..index + 1))?[0])
}

fn strengths<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
) -> common::Result<DeviceVec<R, u32>> {
    common::reduce_rows(exec, graph.graph(), graph.weights().slice(..), 0, SumU32)
}

fn local_move<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
    max_passes: usize,
) -> common::Result<DeviceVec<R, u32>> {
    let topology = graph.graph();
    let n = topology.vertex_count();
    let m2 = vector::reduce(exec, graph.weights().slice(..), 0, SumU32)?;
    let strength = strengths(exec, graph)?;
    let communities = vector::map(exec, common::counting_u32(0, n as usize), Identity)?;
    let totals = vector::map(exec, strength.slice(..), Identity)?;
    if m2 == 0 {
        return Ok(communities);
    }

    for _ in 0..max_passes {
        let mut moved = false;
        for vertex in 0..n {
            let vertex_index = vertex;
            let current = read_u32(exec, &communities, vertex_index)?;
            let vertex_strength = read_u32(exec, &strength, vertex_index)?;
            let current_total = read_u32(exec, &totals, current)?;
            vector::scatter(
                exec,
                lazy::constant(current_total - vertex_strength).take(1),
                common::indices(lazy::constant(current).take(1)),
                totals.slice_mut(..),
            )?;

            let bounds = exec.to_host(&topology.offsets().slice(vertex_index..vertex_index + 2))?;
            let destinations = topology.destinations().slice(bounds[0]..bounds[1]);
            let edge_weights = graph.weights().slice(bounds[0]..bounds[1]);
            let row_len = bounds[1] - bounds[0];
            let neighbor_communities = vector::gather(
                exec,
                communities.slice(..),
                common::indices(destinations.clone()),
            )?;
            let non_self = vector::map(
                exec,
                zip2(destinations, lazy::constant(vertex).take(row_len)),
                Different,
            )?;
            let neighbor_communities = common::materialize_exact(
                exec,
                vector::copy_where(
                    exec,
                    neighbor_communities.slice(..),
                    common::stencil(non_self.slice(..)),
                )?,
            )?;
            let neighbor_weights = common::materialize_exact(
                exec,
                vector::copy_where(exec, edge_weights, common::stencil(non_self.slice(..)))?,
            )?;
            let neighbor_count = neighbor_communities.len();

            // The current community is always an option, even when the vertex
            // currently has no edge back into it.
            let candidate_capacity = neighbor_communities.len() + 1;
            let candidate_communities = exec.alloc::<u32>(candidate_capacity);
            let candidate_weights = exec.alloc::<u32>(candidate_capacity);
            vector::scatter(
                exec,
                neighbor_communities.slice(..),
                common::indices(common::counting_u32(0, neighbor_count as usize)),
                candidate_communities.slice_mut(..),
            )?;
            vector::scatter(
                exec,
                neighbor_weights.slice(..),
                common::indices(common::counting_u32(0, neighbor_count as usize)),
                candidate_weights.slice_mut(..),
            )?;
            vector::scatter(
                exec,
                lazy::constant(current).take(1),
                common::indices(lazy::constant(neighbor_count).take(1)),
                candidate_communities.slice_mut(..),
            )?;
            vector::scatter(
                exec,
                lazy::constant(0u32).take(1),
                common::indices(lazy::constant(neighbor_count).take(1)),
                candidate_weights.slice_mut(..),
            )?;
            let sorted_communities = vector::sort(exec, candidate_communities.slice(..), LessU32)?;
            let sorted_weights = vector::sort_by_key(
                exec,
                candidate_communities.slice(..),
                candidate_weights.slice(..),
                LessU32,
            )?;
            let (unique_communities, incident_weights) = common::materialize_exact_pair(
                exec,
                vector::reduce_by_key(
                    exec,
                    sorted_communities.slice(..),
                    sorted_weights.slice(..),
                    EqualU32,
                    0,
                    SumU32,
                )?,
            )?;
            let candidate_totals = vector::gather(
                exec,
                totals.slice(..),
                common::indices(unique_communities.slice(..)),
            )?;
            let candidate_count = unique_communities.len();
            let scores = vector::map(
                exec,
                massively::zip4(
                    incident_weights.slice(..),
                    lazy::constant(vertex_strength).take(candidate_count),
                    candidate_totals.slice(..),
                    lazy::constant(m2).take(candidate_count),
                ),
                GainScore,
            )?;
            let best_index = vector::max_element(
                exec,
                zip2(scores.slice(..), unique_communities.slice(..)),
                CandidateLess,
            )?
            .expect("the current community is always a candidate");
            let current_location = vector::lower_bound(
                exec,
                unique_communities.slice(..),
                lazy::constant(current).take(1),
                LessU32,
            )?;
            let current_index = exec.to_host(&current_location)?[0];
            let best = read_u32(exec, &unique_communities, best_index)?;
            let best_score = read_f32(exec, &scores, best_index)?;
            let current_score = read_f32(exec, &scores, current_index)?;

            if best != current && best_score > current_score + 1.0e-6 {
                let best_total = read_u32(exec, &totals, best)?;
                vector::scatter(
                    exec,
                    lazy::constant(best_total + vertex_strength).take(1),
                    common::indices(lazy::constant(best).take(1)),
                    totals.slice_mut(..),
                )?;
                vector::scatter(
                    exec,
                    lazy::constant(best).take(1),
                    common::indices(lazy::constant(vertex).take(1)),
                    communities.slice_mut(..),
                )?;
                moved = true;
            } else {
                vector::scatter(
                    exec,
                    lazy::constant(current_total).take(1),
                    common::indices(lazy::constant(current).take(1)),
                    totals.slice_mut(..),
                )?;
            }
        }
        if !moved {
            break;
        }
    }

    Ok(communities)
}

fn compact<R: Runtime>(
    exec: &Executor<R>,
    communities: &DeviceVec<R, u32>,
) -> common::Result<(DeviceVec<R, u32>, u32)> {
    let sorted = vector::sort(exec, communities.slice(..), LessU32)?;
    let unique =
        common::materialize_exact(exec, vector::unique(exec, sorted.slice(..), EqualU32)?)?;
    let count = unique.len();
    let labels = vector::lower_bound(exec, unique.slice(..), communities.slice(..), LessU32)?;
    Ok((labels, count))
}

fn contract<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
    labels: &DeviceVec<R, u32>,
    community_count: u32,
) -> common::Result<DeviceWeightedCsr<R, u32>> {
    let topology = graph.graph();
    let sources = topology.segmentation().segment_ids(exec)?;
    let source_labels = vector::gather(exec, labels.slice(..), common::indices(sources.slice(..)))?;
    let destination_labels = vector::gather(
        exec,
        labels.slice(..),
        common::indices(topology.destinations().slice(..)),
    )?;
    let pair_keys = zip2(source_labels.slice(..), destination_labels.slice(..));
    let sorted_pairs = vector::sort(exec, pair_keys.clone(), PairLess)?;
    let sorted_weights = vector::sort_by_key(exec, pair_keys, graph.weights().slice(..), PairLess)?;
    let (pairs, weights) = common::materialize_exact_pair(
        exec,
        vector::reduce_by_key(
            exec,
            sorted_pairs.slice(..),
            sorted_weights.slice(..),
            PairEqual,
            0,
            SumU32,
        )?,
    )?;
    let (pair_sources, pair_destinations) = MStorage::into_columns(pairs);
    let edge_count = weights.len();
    let (row_ids, row_counts) = common::materialize_exact_pair(
        exec,
        vector::reduce_by_key(
            exec,
            pair_sources.slice(..),
            lazy::constant(1u32).take(edge_count),
            EqualU32,
            0,
            SumU32,
        )?,
    )?;
    let counts = common::filled(exec, community_count as usize, 0u32)?;
    vector::scatter(
        exec,
        row_counts.slice(..),
        common::indices(row_ids.slice(..)),
        counts.slice_mut(..),
    )?;
    let ends = vector::inclusive_scan(exec, counts.slice(..), SumU32)?;
    let offsets = common::filled(exec, community_count as usize + 1, 0u32)?;
    vector::scatter(
        exec,
        ends.slice(..),
        common::indices(common::counting_u32(1, community_count as usize)),
        offsets.slice_mut(..),
    )?;
    let topology = DeviceCsr::from_parts(exec, pair_destinations, offsets)?;
    DeviceWeightedCsr::from_parts(topology, weights)
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    max_passes: usize,
    max_levels: usize,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.vertex_count();
    assert!(n != 0);
    let weights = common::filled(exec, graph.edge_count(), 1u32)?;
    let mut level_graph = DeviceWeightedCsr::from_parts(graph.clone(), weights)?;
    let mut assignment = vector::map(exec, common::counting_u32(0, n as usize), Identity)?;

    for _ in 0..max_levels {
        let communities = local_move(exec, &level_graph, max_passes)?;
        let (labels, community_count) = compact(exec, &communities)?;
        assignment = vector::gather(
            exec,
            labels.slice(..),
            common::indices(assignment.slice(..)),
        )?;
        if community_count == level_graph.graph().vertex_count() {
            break;
        }
        level_graph = contract(exec, &level_graph, &labels, community_count)?;
    }

    Ok(assignment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn separates_two_triangles_joined_by_one_bridge() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(
            vec![0, 2, 4, 7, 10, 12, 14],
            vec![1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4],
        );
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let communities = solve(&exec, &graph, 20, 10).unwrap();
        assert_eq!(exec.to_host(&communities).unwrap(), vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn contracts_sparse_graph_after_host_materialization() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(
            vec![0, 1, 3, 5, 7, 8, 10],
            vec![4, 2, 3, 1, 5, 1, 5, 0, 2, 3],
        );
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let communities = solve(&exec, &graph, 20, 10).unwrap();
        let communities = exec.to_host(&communities).unwrap();
        assert_eq!(communities.len(), 6);
        assert!(communities.iter().all(|&community| community < 6));
    }
}
