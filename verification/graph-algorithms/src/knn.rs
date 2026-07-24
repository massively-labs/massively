//! Adjacency-local K-nearest neighbors for runtime-dimensional feature rows.
//!
//! Each vertex ranks its existing CSR neighbors by squared Euclidean distance.
//! Features are vertex-major: `features[vertex * feature_count + feature]`.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{BinaryPredicateOp, ExpandOp, UnaryOp},
    seg::{Executable, ForEachSegment, Segment, Segmentation, Sort},
    vector, zip2, zip3,
};

use super::common::{self, DeviceCsr};

pub struct KnnGraph<R: Runtime> {
    destinations: DeviceVec<R, u32>,
    distances: DeviceVec<R, f32>,
    segmentation: Segmentation<R>,
}

impl<R: Runtime> KnnGraph<R> {
    pub const fn destinations(&self) -> &DeviceVec<R, u32> {
        &self.destinations
    }

    pub const fn distances(&self) -> &DeviceVec<R, f32> {
        &self.distances
    }

    pub const fn segmentation(&self) -> &Segmentation<R> {
        &self.segmentation
    }

    pub fn offsets(&self) -> massively::DeviceSlice<u32> {
        self.segmentation.offsets()
    }

    pub fn vertex_count(&self) -> u32 {
        self.segmentation.segment_count()
    }

    pub fn edge_count(&self) -> u32 {
        self.destinations.len()
    }

    pub fn topology(&self, exec: &Executor<R>) -> common::Result<DeviceCsr<R>> {
        let offsets = exec.alloc::<u32>(self.vertex_count() as usize + 1);
        vector::copy(exec, self.offsets(), offsets.slice_mut(..))?;
        let destinations = exec.alloc::<u32>(self.edge_count() as usize);
        vector::copy(
            exec,
            self.destinations.slice(..),
            destinations.slice_mut(..),
        )?;
        DeviceCsr::from_parts(exec, destinations, offsets)
    }
}

struct FeatureIndex;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for FeatureIndex {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        input.0 * input.1 + input.2
    }
}

struct SquaredDifference;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for SquaredDifference {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        let difference = input.0 - input.1;
        difference * difference
    }
}

struct AddF32;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for AddF32 {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        input.0 + input.1
    }
}

struct NeighborBefore;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for NeighborBefore {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.0 != rhs.0 {
            lhs.0 < rhs.0
        } else {
            lhs.1 < rhs.1
        })
    }
}

struct PrefixLength;

#[cubecl::cube]
impl UnaryOp<(Segment<(f32, u32)>, u32)> for PrefixLength {
    type Output = u32;

    fn apply(input: (Segment<(f32, u32)>, u32)) -> u32 {
        u32::min(input.0.len(), input.1)
    }
}

struct Prefix;

#[cubecl::cube]
impl ExpandOp<(Segment<(f32, u32)>, u32)> for Prefix {
    type Output = (f32, u32);

    fn count(input: (Segment<(f32, u32)>, u32)) -> u32 {
        u32::min(input.0.len(), input.1)
    }

    fn generate(input: (Segment<(f32, u32)>, u32), local_index: u32) -> Self::Output {
        input.0.at(local_index)
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    features: &DeviceVec<R, f32>,
    feature_count: u32,
    k: u32,
) -> common::Result<KnnGraph<R>> {
    assert!(feature_count != 0);
    assert_eq!(
        features.len(),
        graph
            .vertex_count()
            .checked_mul(feature_count)
            .expect("feature storage exceeds u32")
    );
    let n = graph.vertex_count();
    let sources = graph.segmentation().segment_ids(exec)?;
    let mut distances = common::filled(exec, graph.edge_count(), 0.0f32)?;

    for feature in 0..feature_count {
        let edge_count = distances.len();
        let source_indices = vector::map(
            exec,
            zip3(
                sources.slice(..),
                lazy::constant(feature_count).take(edge_count),
                lazy::constant(feature).take(edge_count),
            ),
            FeatureIndex,
        )?;
        let destination_indices = vector::map(
            exec,
            zip3(
                graph.destinations().slice(..),
                lazy::constant(feature_count).take(edge_count),
                lazy::constant(feature).take(edge_count),
            ),
            FeatureIndex,
        )?;
        let lhs = vector::gather(
            exec,
            features.slice(..),
            common::indices(source_indices.slice(..)),
        )?;
        let rhs = vector::gather(
            exec,
            features.slice(..),
            common::indices(destination_indices.slice(..)),
        )?;
        let difference = vector::map(exec, zip2(lhs.slice(..), rhs.slice(..)), SquaredDifference)?;
        distances = vector::map(
            exec,
            zip2(distances.slice(..), difference.slice(..)),
            AddF32,
        )?;
    }

    let sorted = ForEachSegment(Sort(NeighborBefore)).run(
        exec,
        graph
            .segmentation()
            .segments(zip2(distances.slice(..), graph.destinations().slice(..)))?,
    )?;
    let (sorted_values, offsets) = sorted.into_parts();
    let rows = massively::seg::SegmentIterator::new(sorted_values.slice(..), offsets);
    let lengths = vector::map(
        exec,
        zip2(rows.clone(), lazy::constant(k).take(n)),
        PrefixLength,
    )?;
    let output = vector::flat_map(exec, zip2(rows, lazy::constant(k).take(n)), Prefix)?;
    let (distances, destinations) = MStorage::into_columns(output);
    let segmentation = Segmentation::from_lengths(exec, lengths.slice(..))?;
    Ok(KnnGraph {
        destinations,
        distances,
        segmentation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn ranks_runtime_dimensional_neighbor_features() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 2, 3, 4], vec![1, 2, 0, 0]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let features = exec.to_device(&[0.0f32, 0.0, 3.0, 0.0, 1.0, 1.0]);
        let output = solve(&exec, &graph, &features, 2, 1).unwrap();
        assert_eq!(exec.to_host(&output.offsets()).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(exec.to_host(output.destinations()).unwrap(), vec![2, 0, 0]);
        assert_eq!(
            exec.to_host(output.distances()).unwrap(),
            vec![2.0, 9.0, 2.0]
        );
    }
}
