//! Deterministic synchronous label propagation.
//!
//! Every iteration selects the most frequent neighbor label. Equal counts are
//! resolved by the smallest label, and isolated vertices keep their label.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{BinaryPredicateOp, Identity, ReductionOp, UnaryOp},
    vector, zip2, zip3,
};

use super::common::{self, DeviceCsr};

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

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct CandidateBefore;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32, u32)> for CandidateBefore {
    fn apply(lhs: (u32, u32, u32), rhs: (u32, u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.0 != rhs.0 {
            lhs.0 < rhs.0
        } else if lhs.1 != rhs.1 {
            lhs.1 > rhs.1
        } else {
            lhs.2 < rhs.2
        })
    }
}

struct SameSource;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32, u32)> for SameSource {
    fn apply(lhs: (u32, u32, u32), rhs: (u32, u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 == rhs.0)
    }
}

struct Changed;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Changed {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != input.1 { 1u32 } else { 0u32 }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    max_iterations: usize,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.vertex_count();
    let mut labels = vector::map(exec, common::counting_u32(0, n as usize), Identity)?;
    if graph.edge_count() == 0 {
        return Ok(labels);
    }
    let sources = graph.segmentation().segment_ids(exec)?;
    let edge_count =
        u32::try_from(graph.edge_count()).map_err(|_| massively::Error::LengthTooLarge {
            len: graph.edge_count(),
        })?;

    for _ in 0..max_iterations {
        let neighbor_labels = vector::gather(
            exec,
            labels.slice(..),
            common::indices(graph.destinations().slice(..)),
        )?;
        let pairs = zip2(sources.slice(..), neighbor_labels.slice(..));
        let sorted_pairs = vector::sort(exec, pairs, PairLess)?;
        let (pairs, counts) = common::materialize_exact_pair(
            exec,
            vector::reduce_by_key(
                exec,
                sorted_pairs.slice(..),
                lazy::constant(1u32).take(edge_count),
                PairEqual,
                0u32,
                SumU32,
            )?,
        )?;
        let (pair_sources, pair_labels) = MStorage::into_columns(pairs);
        let pair_count = counts.len();
        let candidate_count = pair_count
            .checked_add(n)
            .expect("label candidate count exceeds u32");
        let candidate_sources = exec.alloc::<u32>(candidate_count);
        let candidate_counts = exec.alloc::<u32>(candidate_count);
        let candidate_labels = exec.alloc::<u32>(candidate_count);
        vector::copy(
            exec,
            pair_sources.slice(..),
            candidate_sources.slice_mut(..pair_count),
        )?;
        vector::copy(
            exec,
            counts.slice(..),
            candidate_counts.slice_mut(..pair_count),
        )?;
        vector::copy(
            exec,
            pair_labels.slice(..),
            candidate_labels.slice_mut(..pair_count),
        )?;
        vector::copy(
            exec,
            common::counting_u32(0, n as usize),
            candidate_sources.slice_mut(pair_count..),
        )?;
        vector::fill(exec, 0u32, candidate_counts.slice_mut(pair_count..))?;
        vector::copy(
            exec,
            labels.slice(..),
            candidate_labels.slice_mut(pair_count..),
        )?;

        let sorted_candidates = vector::sort(
            exec,
            zip3(
                candidate_sources.slice(..),
                candidate_counts.slice(..),
                candidate_labels.slice(..),
            ),
            CandidateBefore,
        )?;
        let winners = common::materialize_exact(
            exec,
            vector::unique(exec, sorted_candidates.slice(..), SameSource)?,
        )?;
        let (_winner_sources, _winner_counts, next) = MStorage::into_columns(winners);
        assert_eq!(next.len(), n);
        let changed = vector::reduce(
            exec,
            lazy::map(zip2(labels.slice(..), next.slice(..)), Changed),
            0u32,
            SumU32,
        )?;
        labels = next;
        if changed == 0 {
            break;
        }
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn uses_smallest_label_for_a_tied_mode() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 2, 3, 4], vec![1, 2, 0, 0]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let labels = solve(&exec, &graph, 1).unwrap();
        assert_eq!(exec.to_host(&labels).unwrap(), vec![1, 0, 0]);
    }
}
