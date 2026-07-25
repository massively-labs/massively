use cubecl::prelude::*;
use massively::{
    DeviceSlice, DeviceVec, Executor, MAlloc, MIndex, MIter, MStorage, MVec, lazy,
    op::{BinaryPredicateOp, ExpandOp, ReductionOp, UnaryOp},
    seg::{Executable, ForEachSegment, Reduce, Segment, Segmentation},
    vector, zip2, zip3,
};

pub(crate) type Result<T> = std::result::Result<T, massively::Error>;

/// Verifies the exact owned length expected by these host-controlled
/// reference algorithms.
pub(crate) fn materialize_exact<R, Storage>(
    _exec: &Executor<R>,
    storage: Storage,
) -> Result<Storage>
where
    R: Runtime,
    Storage: MStorage<R>,
{
    storage.len()?;
    Ok(storage)
}

pub(crate) fn materialize_exact_pair<R, Left, Right>(
    _exec: &Executor<R>,
    (left, right): (Left, Right),
) -> Result<(Left, Right)>
where
    R: Runtime,
    Left: MStorage<R>,
    Right: MStorage<R>,
{
    let len = left.len()?;
    let right_len = right.len()?;
    if right_len != len {
        return Err(massively::Error::LengthMismatch {
            left: len as usize,
            right: right_len as usize,
        });
    }
    Ok((left, right))
}

pub(crate) fn counting_u32(start: usize, len: usize) -> lazy::Taken<lazy::Counting> {
    lazy::counting(u32::try_from(start).expect("counting value exceeds u32"))
        .take(u32::try_from(len).expect("counting length exceeds u32"))
}

pub(crate) fn indices<Input>(input: Input) -> Input {
    input
}

pub(crate) fn stencil<Input>(input: Input) -> Input {
    input
}

pub(crate) trait FillValue<R: Runtime>: Sized {
    fn filled(exec: &Executor<R>, len: MIndex, value: Self) -> Result<DeviceVec<R, Self>>;
}

impl<R: Runtime> FillValue<R> for u32 {
    fn filled(exec: &Executor<R>, len: MIndex, value: Self) -> Result<DeviceVec<R, Self>> {
        let output = exec.alloc::<u32>(len);
        vector::fill(exec, value, output.slice_mut(..))?;
        Ok(output)
    }
}

impl<R: Runtime> FillValue<R> for f32 {
    fn filled(exec: &Executor<R>, len: MIndex, value: Self) -> Result<DeviceVec<R, Self>> {
        let output = exec.alloc::<f32>(len);
        vector::fill(exec, value, output.slice_mut(..))?;
        Ok(output)
    }
}

pub(crate) fn filled<R, T>(exec: &Executor<R>, len: MIndex, value: T) -> Result<DeviceVec<R, T>>
where
    R: Runtime,
    T: FillValue<R>,
{
    T::filled(exec, len, value)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrGraph {
    pub offsets: Vec<u32>,
    pub neighbors: Vec<u32>,
}

impl CsrGraph {
    pub fn new(offsets: Vec<u32>, neighbors: Vec<u32>) -> Self {
        assert!(
            !offsets.is_empty(),
            "CSR offsets must contain the initial zero"
        );
        assert_eq!(offsets[0], 0, "CSR offsets must start at zero");
        assert_eq!(offsets.last().copied().unwrap() as usize, neighbors.len());
        assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        let vertices = offsets.len() - 1;
        assert!(neighbors.iter().all(|&vertex| (vertex as usize) < vertices));
        Self { offsets, neighbors }
    }

    pub fn vertex_count(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn row(&self, vertex: usize) -> &[u32] {
        &self.neighbors[self.offsets[vertex] as usize..self.offsets[vertex + 1] as usize]
    }
}

#[derive(Clone, Debug)]
pub struct WeightedCsr {
    pub graph: CsrGraph,
    pub weights: Vec<f32>,
}

impl WeightedCsr {
    pub fn new(graph: CsrGraph, weights: Vec<f32>) -> Self {
        assert_eq!(graph.neighbors.len(), weights.len());
        Self { graph, weights }
    }
}

/// An owned CSR topology resident in device memory.
pub struct DeviceCsr<R: Runtime> {
    destinations: DeviceVec<R, u32>,
    segmentation: Segmentation<R>,
    vertex_count: u32,
}

impl<R: Runtime> Clone for DeviceCsr<R> {
    fn clone(&self) -> Self {
        Self {
            destinations: self.destinations.clone(),
            segmentation: self.segmentation.clone(),
            vertex_count: self.vertex_count,
        }
    }
}

impl<R: Runtime> DeviceCsr<R> {
    /// Creates a device CSR from already-resident storage without copying it.
    pub fn from_parts(
        exec: &Executor<R>,
        destinations: DeviceVec<R, u32>,
        offsets: DeviceVec<R, u32>,
    ) -> Result<Self> {
        let segmentation = Segmentation::from_offsets(exec, offsets.slice(..))?;
        if segmentation.value_count() != destinations.len() {
            return Err(massively::Error::LengthMismatch {
                left: segmentation.value_count() as usize,
                right: destinations.len() as usize,
            });
        }
        let vertex_count = segmentation.segment_count();
        Ok(Self {
            destinations,
            segmentation,
            vertex_count,
        })
    }

    /// Explicitly uploads a host CSR topology.
    pub fn from_host(exec: &Executor<R>, graph: &CsrGraph) -> Result<Self> {
        Self::from_parts(
            exec,
            exec.to_device(&graph.neighbors),
            exec.to_device(&graph.offsets),
        )
    }

    /// Returns the number of vertices.
    pub const fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Returns the number of directed CSR entries.
    pub fn edge_count(&self) -> usize {
        self.destinations.len() as usize
    }

    /// Returns the resident destinations.
    pub const fn destinations(&self) -> &DeviceVec<R, u32> {
        &self.destinations
    }

    /// Returns the resident offsets.
    pub fn offsets(&self) -> DeviceSlice<u32> {
        self.segmentation.offsets()
    }

    /// Returns the generic segmentation that delimits the CSR rows.
    pub const fn segmentation(&self) -> &Segmentation<R> {
        &self.segmentation
    }
}

/// A floating-point weighted CSR matrix resident in device memory.
pub struct DeviceWeightedCsr<R: Runtime, Weight = f32> {
    graph: DeviceCsr<R>,
    weights: DeviceVec<R, Weight>,
}

impl<R: Runtime, Weight> DeviceWeightedCsr<R, Weight> {
    /// Creates a resident weighted CSR from already-resident storage.
    pub fn from_parts(graph: DeviceCsr<R>, weights: DeviceVec<R, Weight>) -> Result<Self> {
        if graph.edge_count() != weights.len() as usize {
            return Err(massively::Error::LengthMismatch {
                left: graph.edge_count(),
                right: weights.len() as usize,
            });
        }
        Ok(Self { graph, weights })
    }

    pub const fn graph(&self) -> &DeviceCsr<R> {
        &self.graph
    }

    pub const fn weights(&self) -> &DeviceVec<R, Weight> {
        &self.weights
    }

    pub fn into_parts(self) -> (DeviceCsr<R>, DeviceVec<R, Weight>) {
        (self.graph, self.weights)
    }
}

impl<R: Runtime> DeviceWeightedCsr<R, f32> {
    /// Explicitly uploads a host floating-point weighted CSR matrix.
    pub fn from_host(exec: &Executor<R>, matrix: &WeightedCsr) -> Result<Self> {
        Self::from_parts(
            DeviceCsr::from_host(exec, &matrix.graph)?,
            exec.to_device(&matrix.weights),
        )
    }
}

impl<R: Runtime> DeviceWeightedCsr<R, u32> {
    /// Explicitly uploads a host CSR topology and integer edge weights.
    pub fn from_host_parts(exec: &Executor<R>, graph: &CsrGraph, weights: &[u32]) -> Result<Self> {
        Self::from_parts(DeviceCsr::from_host(exec, graph)?, exec.to_device(weights))
    }
}

/// Materialized edge rows selected by an arbitrary vertex frontier.
///
/// This is verification-local storage assembled from `seg` and `vector`
/// primitives; it is not a graph execution abstraction.
pub(crate) struct ExpandedRows<R: Runtime> {
    sources: DeviceVec<R, u32>,
    destinations: DeviceVec<R, u32>,
    edge_ids: DeviceVec<R, u32>,
    segmentation: Segmentation<R>,
}

impl<R: Runtime> ExpandedRows<R> {
    pub const fn sources(&self) -> &DeviceVec<R, u32> {
        &self.sources
    }

    pub const fn destinations(&self) -> &DeviceVec<R, u32> {
        &self.destinations
    }

    pub const fn edge_ids(&self) -> &DeviceVec<R, u32> {
        &self.edge_ids
    }

    pub const fn segmentation(&self) -> &Segmentation<R> {
        &self.segmentation
    }
}

struct SegmentLength;

#[cubecl::cube]
impl UnaryOp<Segment<(u32, u32)>> for SegmentLength {
    type Output = u32;

    fn apply(input: Segment<(u32, u32)>) -> u32 {
        input.len()
    }
}

struct ExpandRow;

#[cubecl::cube]
impl ExpandOp<(u32, Segment<(u32, u32)>)> for ExpandRow {
    type Output = (u32, u32, u32);

    fn count(input: (u32, Segment<(u32, u32)>)) -> u32 {
        input.1.len()
    }

    fn generate(input: (u32, Segment<(u32, u32)>), local_index: u32) -> Self::Output {
        let edge = input.1.at(local_index);
        (input.0, edge.0, edge.1)
    }
}

pub(crate) fn expand_rows<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    frontier: DeviceSlice<u32>,
) -> Result<ExpandedRows<R>> {
    let edge_ids = counting_u32(0, graph.edge_count());
    let rows = graph
        .segmentation()
        .segments(zip2(graph.destinations().slice(..), edge_ids))?;
    let selected_rows = lazy::permute(rows, frontier.clone());
    let lengths = vector::map(exec, selected_rows.clone(), SegmentLength)?;
    let segmentation = Segmentation::from_lengths(exec, lengths.slice(..))?;
    let expanded = vector::flat_map(exec, zip2(frontier, selected_rows), ExpandRow)?;
    let (sources, destinations, edge_ids) = MStorage::into_columns(expanded);
    Ok(ExpandedRows {
        sources,
        destinations,
        edge_ids,
        segmentation,
    })
}

pub(crate) fn reduce_rows<R, Values, Item, Op>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    values: Values,
    init: massively::Scalar<R, Item>,
    op: Op,
) -> Result<MVec<R, Item>>
where
    R: Runtime,
    Values: MIter<R, Item = Item>,
    Item: MAlloc<R>,
    Op: ReductionOp<Item>,
{
    ForEachSegment(Reduce(op, init)).run(exec, graph.segmentation().segments(values)?)
}

pub(crate) struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

pub(crate) struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

pub(crate) struct EdgePairLess;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for EdgePairLess {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.0 != rhs.0 {
            lhs.0 < rhs.0
        } else {
            lhs.1 < rhs.1
        })
    }
}

pub(crate) struct EdgePairEqual;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for EdgePairEqual {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 == rhs.0 && lhs.1 == rhs.1)
    }
}

pub(crate) struct MinU32;

#[cubecl::cube]
impl ReductionOp<u32> for MinU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::min(lhs, rhs)
    }
}

pub(crate) struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct Lowered;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Lowered {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.1 < input.0 { 1u32 } else { 0u32 }
    }
}

struct ApplyMinimum;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for ApplyMinimum {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        u32::min(input.0, input.1)
    }
}

pub(crate) fn relax_min<R: Runtime>(
    exec: &Executor<R>,
    destinations: DeviceSlice<u32>,
    proposals: DeviceSlice<u32>,
    infinity: u32,
    state: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>> {
    let sorted_destinations = vector::sort(exec, destinations.clone(), LessU32)?;
    let sorted_proposals = vector::sort_by_key(exec, destinations, proposals, LessU32)?;
    let (destinations, proposals) = vector::reduce_by_key(
        exec,
        sorted_destinations.slice(..),
        sorted_proposals.slice(..),
        EqualU32,
        exec.value(infinity)?,
        MinU32,
    )?;
    let old = vector::gather(exec, state.slice(..), destinations.slice(..))?;
    let lowered = vector::map(exec, zip2(old.slice(..), proposals.slice(..)), Lowered)?;
    let next = vector::map(exec, zip2(old.slice(..), proposals.slice(..)), ApplyMinimum)?;
    vector::scatter(
        exec,
        next.slice(..),
        destinations.slice(..),
        state.slice_mut(..),
    )?;
    vector::copy_where(exec, destinations.slice(..), stencil(lowered.slice(..)))
}

pub(crate) fn weighted_csr_from_edges<R: Runtime>(
    exec: &Executor<R>,
    vertex_count: u32,
    sources: DeviceSlice<u32>,
    destinations: DeviceSlice<u32>,
    weights: DeviceSlice<u32>,
) -> Result<DeviceWeightedCsr<R, u32>> {
    if sources.len() != destinations.len() || sources.len() != weights.len() {
        return Err(massively::Error::LengthMismatch {
            left: sources.len() as usize,
            right: u32::min(destinations.len(), weights.len()) as usize,
        });
    }
    if sources.len() == 0 {
        return DeviceWeightedCsr::from_parts(
            DeviceCsr::from_parts(
                exec,
                exec.alloc::<u32>(0),
                filled(
                    exec,
                    vertex_count
                        .checked_add(1)
                        .expect("offset count exceeds MIndex"),
                    0u32,
                )?,
            )?,
            exec.alloc::<u32>(0),
        );
    }

    let pairs = zip2(sources, destinations);
    let sorted_pairs = vector::sort(exec, pairs.clone(), EdgePairLess)?;
    let sorted_weights = vector::sort_by_key(exec, pairs, weights, EdgePairLess)?;
    let (pairs, weights) = materialize_exact_pair(
        exec,
        vector::reduce_by_key(
            exec,
            sorted_pairs.slice(..),
            sorted_weights.slice(..),
            EdgePairEqual,
            exec.value(0u32)?,
            SumU32,
        )?,
    )?;
    let (sources, destinations) = MStorage::into_columns(pairs);
    let edge_count = destinations.len();
    let (row_ids, row_counts) = materialize_exact_pair(
        exec,
        vector::reduce_by_key(
            exec,
            sources.slice(..),
            lazy::constant(1u32).take(edge_count),
            EqualU32,
            exec.value(0u32)?,
            SumU32,
        )?,
    )?;
    let counts = filled(exec, vertex_count, 0u32)?;
    vector::scatter(
        exec,
        row_counts.slice(..),
        indices(row_ids.slice(..)),
        counts.slice_mut(..),
    )?;
    let ends = vector::inclusive_scan(exec, counts.slice(..), SumU32)?;
    let offsets = filled(
        exec,
        vertex_count
            .checked_add(1)
            .expect("offset count exceeds MIndex"),
        0u32,
    )?;
    vector::scatter(
        exec,
        ends.slice(..),
        indices(counting_u32(1, vertex_count as usize)),
        offsets.slice_mut(..),
    )?;
    DeviceWeightedCsr::from_parts(DeviceCsr::from_parts(exec, destinations, offsets)?, weights)
}

struct DanglingRank;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for DanglingRank {
    type Output = f32;

    fn apply(input: (f32, u32)) -> f32 {
        if input.1 == 0u32 { input.0 } else { 0.0f32 }
    }
}

struct RankContribution;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32)> for RankContribution {
    type Output = f32;

    fn apply(input: (f32, u32, f32)) -> f32 {
        if input.1 == 0u32 {
            0.0f32
        } else {
            input.0 * input.2 / input.1 as f32
        }
    }
}

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

pub(crate) fn resident_degrees<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> Result<DeviceVec<R, u32>> {
    graph.segmentation().lengths(exec)
}

pub(crate) fn dangling_mass<R: Runtime>(
    exec: &Executor<R>,
    rank: &DeviceVec<R, f32>,
    degree: &DeviceVec<R, u32>,
) -> Result<f32> {
    vector::reduce(
        exec,
        lazy::map(zip2(rank.slice(..), degree.slice(..)), DanglingRank),
        exec.value(0.0)?,
        SumF32,
    )?
    .read(exec)
}

pub(crate) fn accumulate_rank<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    degree: &DeviceVec<R, u32>,
    rank: &DeviceVec<R, f32>,
    damping: f32,
    output: &DeviceVec<R, f32>,
) -> Result<()> {
    let sources = graph.segmentation().segment_ids(exec)?;
    let edge_count =
        u32::try_from(graph.edge_count()).map_err(|_| massively::Error::LengthTooLarge {
            len: graph.edge_count(),
        })?;
    let contributions = lazy::map(
        zip3(
            lazy::permute(rank.slice(..), sources.slice(..)),
            lazy::permute(degree.slice(..), sources.slice(..)),
            lazy::constant(damping).take(edge_count),
        ),
        RankContribution,
    );
    vector::scatter_reduce(
        exec,
        contributions,
        graph.destinations().slice(..),
        exec.value(0.0)?,
        SumF32,
        output.slice_mut(..),
    )
}

#[cfg(test)]
pub(crate) fn sample_graph() -> CsrGraph {
    CsrGraph::new(vec![0, 2, 5, 8, 10], vec![1, 2, 0, 2, 3, 0, 1, 3, 1, 2])
}

#[cfg(test)]
pub(crate) fn path_graph() -> CsrGraph {
    CsrGraph::new(vec![0, 1, 3, 5, 6], vec![1, 0, 2, 1, 3, 2])
}

#[cfg(test)]
pub(crate) fn assert_near(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "actual={actual}, expected={expected}"
        );
    }
}
