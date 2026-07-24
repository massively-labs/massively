//! Shared-nearest-neighbor clustering.
//!
//! Mutual KNN edges are retained when their endpoint rows have at least the
//! requested number of shared neighbors. Core vertices are connected
//! components of that similarity graph; non-core vertices inherit the
//! smallest adjacent core component and otherwise remain noise (`u32::MAX`).

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::UnaryOp,
    seg::{Executable, ForEachSegment, Segment, Sort},
    vector, zip2, zip3,
};

use super::{cc, common, knn::KnnGraph};

pub struct Clustering<R: Runtime> {
    labels: DeviceVec<R, u32>,
    core: DeviceVec<R, u32>,
}

impl<R: Runtime> Clustering<R> {
    pub const fn labels(&self) -> &DeviceVec<R, u32> {
        &self.labels
    }

    pub const fn core(&self) -> &DeviceVec<R, u32> {
        &self.core
    }
}

struct Similarity;

#[cubecl::cube]
impl UnaryOp<(Segment<u32>, Segment<u32>, u32)> for Similarity {
    type Output = (u32, u32);

    fn apply(input: (Segment<u32>, Segment<u32>, u32)) -> Self::Output {
        let left = RuntimeCell::<u32>::new(0u32);
        let right = RuntimeCell::<u32>::new(0u32);
        let count = RuntimeCell::<u32>::new(0u32);
        let mutual = RuntimeCell::<u32>::new(0u32);
        while left.read() < input.0.len() && right.read() < input.1.len() {
            let lhs = input.0.at(left.read());
            let rhs = input.1.at(right.read());
            if rhs == input.2 {
                mutual.store(1u32);
            }
            if lhs < rhs {
                left.store(left.read() + 1u32);
            } else if rhs < lhs {
                right.store(right.read() + 1u32);
            } else {
                count.store(count.read() + 1u32);
                left.store(left.read() + 1u32);
                right.store(right.read() + 1u32);
            }
        }
        while right.read() < input.1.len() {
            if input.1.at(right.read()) == input.2 {
                mutual.store(1u32);
            }
            right.store(right.read() + 1u32);
        }
        (count.read(), mutual.read())
    }
}

struct KeepSimilarity;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for KeepSimilarity {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        if input.1 != 0u32 && input.0 >= input.2 {
            1u32
        } else {
            0u32
        }
    }
}

struct AtLeast;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AtLeast {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 >= input.1 { 1u32 } else { 0u32 }
    }
}

struct BothCore;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for BothCore {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != 0u32 && input.1 != 0u32 {
            1u32
        } else {
            0u32
        }
    }
}

struct CoreLabel;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for CoreLabel {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != 0u32 { input.1 } else { u32::MAX }
    }
}

struct AssignLabel;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for AssignLabel {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        if input.0 != 0u32 { input.1 } else { input.2 }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    knn: &KnnGraph<R>,
    shared_neighbor_threshold: u32,
    core_neighbor_threshold: u32,
) -> common::Result<Clustering<R>> {
    let n = knn.vertex_count();
    let sorted = ForEachSegment(Sort(common::LessU32)).run(
        exec,
        knn.segmentation().segments(knn.destinations().slice(..))?,
    )?;
    let (sorted_destinations, offsets) = sorted.into_parts();
    let rows = massively::seg::SegmentIterator::new(sorted_destinations.slice(..), offsets);
    let sources = knn.segmentation().segment_ids(exec)?;
    let similarity = vector::map(
        exec,
        zip3(
            lazy::permute(rows.clone(), sources.slice(..)),
            lazy::permute(rows, knn.destinations().slice(..)),
            sources.slice(..),
        ),
        Similarity,
    )?;
    let (scores, mutual) = MStorage::into_columns(similarity);
    let keep = vector::map(
        exec,
        zip3(
            scores.slice(..),
            mutual.slice(..),
            lazy::constant(shared_neighbor_threshold).take(scores.len()),
        ),
        KeepSimilarity,
    )?;
    let kept = common::materialize_exact(
        exec,
        vector::copy_where(
            exec,
            zip3(
                sources.slice(..),
                knn.destinations().slice(..),
                scores.slice(..),
            ),
            common::stencil(keep.slice(..)),
        )?,
    )?;
    let (similarity_sources, similarity_destinations, similarity_scores) =
        MStorage::into_columns(kept);
    let similarity_graph = common::weighted_csr_from_edges(
        exec,
        n,
        similarity_sources.slice(..),
        similarity_destinations.slice(..),
        similarity_scores.slice(..),
    )?;
    let (similarity_topology, _scores) = similarity_graph.into_parts();
    let similarity_degree = common::resident_degrees(exec, &similarity_topology)?;
    let core = vector::map(
        exec,
        zip2(
            similarity_degree.slice(..),
            lazy::constant(core_neighbor_threshold).take(n),
        ),
        AtLeast,
    )?;

    let similarity_sources = similarity_topology.segmentation().segment_ids(exec)?;
    let source_core = vector::gather(
        exec,
        core.slice(..),
        common::indices(similarity_sources.slice(..)),
    )?;
    let destination_core = vector::gather(
        exec,
        core.slice(..),
        common::indices(similarity_topology.destinations().slice(..)),
    )?;
    let keep_core = vector::map(
        exec,
        zip2(source_core.slice(..), destination_core.slice(..)),
        BothCore,
    )?;
    let core_edges = common::materialize_exact(
        exec,
        vector::copy_where(
            exec,
            zip2(
                similarity_sources.slice(..),
                similarity_topology.destinations().slice(..),
            ),
            common::stencil(keep_core.slice(..)),
        )?,
    )?;
    let (core_sources, core_destinations) = MStorage::into_columns(core_edges);
    let core_weights = common::filled(exec, core_sources.len() as usize, 1u32)?;
    let core_graph = common::weighted_csr_from_edges(
        exec,
        n,
        core_sources.slice(..),
        core_destinations.slice(..),
        core_weights.slice(..),
    )?
    .into_parts()
    .0;
    let component = cc::solve(exec, &core_graph)?;

    let destination_core = vector::gather(
        exec,
        core.slice(..),
        common::indices(similarity_topology.destinations().slice(..)),
    )?;
    let destination_component = vector::gather(
        exec,
        component.slice(..),
        common::indices(similarity_topology.destinations().slice(..)),
    )?;
    let adjacent_candidates = lazy::map(
        zip2(destination_core.slice(..), destination_component.slice(..)),
        CoreLabel,
    );
    let adjacent_component = common::reduce_rows(
        exec,
        &similarity_topology,
        adjacent_candidates,
        u32::MAX,
        common::MinU32,
    )?;
    let labels = vector::map(
        exec,
        zip3(
            core.slice(..),
            component.slice(..),
            adjacent_component.slice(..),
        ),
        AssignLabel,
    )?;
    Ok(Clustering { labels, core })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CsrGraph, DeviceCsr, knn};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn separates_two_mutual_neighbor_groups() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(
            vec![0, 2, 4, 6, 8, 10, 12],
            vec![1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4],
        );
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let features = exec.to_device(&[0.0f32, 0.1, 0.2, 10.0, 10.1, 10.2]);
        let nearest = knn::solve(&exec, &graph, &features, 1, 2).unwrap();
        let output = solve(&exec, &nearest, 1, 1).unwrap();
        assert_eq!(
            exec.to_host(output.labels()).unwrap(),
            vec![0, 0, 0, 3, 3, 3]
        );
        assert_eq!(exec.to_host(output.core()).unwrap(), vec![1, 1, 1, 1, 1, 1]);
    }
}
