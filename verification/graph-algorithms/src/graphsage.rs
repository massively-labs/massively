//! A runtime-dimensional GraphSAGE mean-aggregation layer.
//!
//! Inputs and outputs are vertex-major flattened matrices. The layer computes
//! `ReLU(X W_self + mean_neighbors(X) W_neighbor + bias)` and can L2-normalize
//! every output row. Repeated calls compose arbitrary-depth inference models
//! without encoding feature dimensions in Rust types.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{ReductionOp, UnaryOp},
    seg::{Executable, ForEachSegment, Reduce, Segmentation},
    vector, zip2, zip3, zip4,
};

use super::common::{self, DeviceCsr};

struct SplitIndex;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for SplitIndex {
    type Output = (u32, u32);

    fn apply(input: (u32, u32)) -> Self::Output {
        (input.0 / input.1, input.0 % input.1)
    }
}

struct MatrixIndex;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for MatrixIndex {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        input.0 * input.1 + input.2
    }
}

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

struct Mean;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for Mean {
    type Output = f32;

    fn apply(input: (f32, u32)) -> f32 {
        if input.1 == 0u32 {
            0.0f32
        } else {
            input.0 / input.1 as f32
        }
    }
}

struct LinearTerm;

#[cubecl::cube]
impl UnaryOp<(f32, f32, f32, f32)> for LinearTerm {
    type Output = f32;

    fn apply(input: (f32, f32, f32, f32)) -> f32 {
        input.0 * input.2 + input.1 * input.3
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

struct Relu;

#[cubecl::cube]
impl UnaryOp<f32> for Relu {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        f32::max(input, 0.0f32)
    }
}

struct Square;

#[cubecl::cube]
impl UnaryOp<f32> for Square {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input * input
    }
}

struct Normalize;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for Normalize {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        if input.1 == 0.0f32 {
            0.0f32
        } else {
            input.0 / input.1.sqrt()
        }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    input: &DeviceVec<R, f32>,
    input_features: u32,
    self_weights: &DeviceVec<R, f32>,
    neighbor_weights: &DeviceVec<R, f32>,
    bias: &DeviceVec<R, f32>,
    output_features: u32,
    normalize: bool,
) -> common::Result<DeviceVec<R, f32>> {
    let n = graph.vertex_count();
    assert!(input_features != 0);
    assert!(output_features != 0);
    assert_eq!(input.len(), n.checked_mul(input_features).unwrap());
    let weight_count = input_features.checked_mul(output_features).unwrap();
    assert_eq!(self_weights.len(), weight_count);
    assert_eq!(neighbor_weights.len(), weight_count);
    assert_eq!(bias.len(), output_features);

    let neighbor_sum = common::filled(exec, input.len(), 0.0f32)?;
    if graph.edge_count() != 0 {
        let edge_features = u32::try_from(graph.edge_count())
            .expect("edge count exceeds u32")
            .checked_mul(input_features)
            .expect("edge-feature expansion exceeds u32");
        let split = vector::map(
            exec,
            zip2(
                common::counting_u32(0, edge_features as usize),
                lazy::constant(input_features).take(edge_features),
            ),
            SplitIndex,
        )?;
        let (edge_ids, features) = MStorage::into_columns(split);
        let edge_sources = graph.segmentation().segment_ids(exec)?;
        let sources = vector::gather(
            exec,
            edge_sources.slice(..),
            common::indices(edge_ids.slice(..)),
        )?;
        let destinations = vector::gather(
            exec,
            graph.destinations().slice(..),
            common::indices(edge_ids.slice(..)),
        )?;
        let source_indices = vector::map(
            exec,
            zip3(
                sources.slice(..),
                lazy::constant(input_features).take(edge_features),
                features.slice(..),
            ),
            MatrixIndex,
        )?;
        let destination_indices = vector::map(
            exec,
            zip3(
                destinations.slice(..),
                lazy::constant(input_features).take(edge_features),
                features.slice(..),
            ),
            MatrixIndex,
        )?;
        let neighbor_values = vector::gather(
            exec,
            input.slice(..),
            common::indices(destination_indices.slice(..)),
        )?;
        vector::scatter_reduce(
            exec,
            neighbor_values.slice(..),
            source_indices.slice(..),
            exec.value(0.0f32)?,
            SumF32,
            neighbor_sum.slice_mut(..),
        )?;
    }

    let input_len = input.len();
    let input_split = vector::map(
        exec,
        zip2(
            common::counting_u32(0, input_len as usize),
            lazy::constant(input_features).take(input_len),
        ),
        SplitIndex,
    )?;
    let (input_vertices, _input_columns) = MStorage::into_columns(input_split);
    let degree = common::resident_degrees(exec, graph)?;
    let expanded_degree = vector::gather(
        exec,
        degree.slice(..),
        common::indices(input_vertices.slice(..)),
    )?;
    let neighbor_mean = vector::map(
        exec,
        zip2(neighbor_sum.slice(..), expanded_degree.slice(..)),
        Mean,
    )?;

    let output_len = n.checked_mul(output_features).unwrap();
    let output_split = vector::map(
        exec,
        zip2(
            common::counting_u32(0, output_len as usize),
            lazy::constant(output_features).take(output_len),
        ),
        SplitIndex,
    )?;
    let (output_vertices, output_columns) = MStorage::into_columns(output_split);
    let mut output = vector::gather(
        exec,
        bias.slice(..),
        common::indices(output_columns.slice(..)),
    )?;

    for feature in 0..input_features {
        let input_indices = vector::map(
            exec,
            zip3(
                output_vertices.slice(..),
                lazy::constant(input_features).take(output_len),
                lazy::constant(feature).take(output_len),
            ),
            MatrixIndex,
        )?;
        let weight_indices = vector::map(
            exec,
            zip3(
                lazy::constant(feature).take(output_len),
                lazy::constant(output_features).take(output_len),
                output_columns.slice(..),
            ),
            MatrixIndex,
        )?;
        let own = vector::gather(
            exec,
            input.slice(..),
            common::indices(input_indices.slice(..)),
        )?;
        let neighbors = vector::gather(
            exec,
            neighbor_mean.slice(..),
            common::indices(input_indices.slice(..)),
        )?;
        let own_weights = vector::gather(
            exec,
            self_weights.slice(..),
            common::indices(weight_indices.slice(..)),
        )?;
        let neighbor_weights = vector::gather(
            exec,
            neighbor_weights.slice(..),
            common::indices(weight_indices.slice(..)),
        )?;
        let term = vector::map(
            exec,
            zip4(
                own.slice(..),
                neighbors.slice(..),
                own_weights.slice(..),
                neighbor_weights.slice(..),
            ),
            LinearTerm,
        )?;
        output = vector::map(exec, zip2(output.slice(..), term.slice(..)), AddF32)?;
    }
    output = vector::map(exec, output.slice(..), Relu)?;
    if !normalize {
        return Ok(output);
    }

    let segmentation = Segmentation::from_lengths(exec, lazy::constant(output_features).take(n))?;
    let norm_squared = ForEachSegment(Reduce(SumF32, exec.value(0.0f32)?)).run(
        exec,
        segmentation.segments(lazy::map(output.slice(..), Square))?,
    )?;
    let expanded_norm = vector::gather(
        exec,
        norm_squared.slice(..),
        common::indices(output_vertices.slice(..)),
    )?;
    vector::map(
        exec,
        zip2(output.slice(..), expanded_norm.slice(..)),
        Normalize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn combines_self_and_runtime_dimensional_neighbor_features() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 1, 2], vec![1, 0]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let input = exec.to_device(&[1.0f32, 0.0, 0.0, 1.0]);
        let identity = exec.to_device(&[1.0f32, 0.0, 0.0, 1.0]);
        let bias = exec.to_device(&[0.0f32, 0.0]);
        let output = solve(
            &exec, &graph, &input, 2, &identity, &identity, &bias, 2, false,
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    }
}
