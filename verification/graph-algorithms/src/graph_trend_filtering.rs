//! Multi-signal graph trend filtering by primal-dual hybrid gradient.
//!
//! This solves
//! `0.5 * ||x - y||² + lambda * sum_(u,v) ||x[u] - x[v]||_1`
//! over the directed CSR entries. Signals are vertex-major and may contain any
//! runtime number of columns. Symmetric CSR therefore counts both orientations.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{ReductionOp, UnaryOp},
    vector, zip2, zip3, zip5,
};

use super::common::{self, DeviceCsr};

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct AddU32;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddU32 {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

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

struct DualStep;

#[cubecl::cube]
impl UnaryOp<(f32, f32, f32, f32, f32)> for DualStep {
    type Output = f32;

    fn apply(input: (f32, f32, f32, f32, f32)) -> f32 {
        let proposal = input.0 + input.3 * (input.1 - input.2);
        f32::max(-input.4, f32::min(input.4, proposal))
    }
}

struct Negate;

#[cubecl::cube]
impl UnaryOp<f32> for Negate {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        -input
    }
}

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

struct PrimalStep;

#[cubecl::cube]
impl UnaryOp<(f32, f32, f32, f32)> for PrimalStep {
    type Output = f32;

    fn apply(input: (f32, f32, f32, f32)) -> f32 {
        (input.0 - input.3 * input.1 + input.3 * input.2) / (1.0f32 + input.3)
    }
}

struct Extrapolate;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for Extrapolate {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        2.0f32 * input.0 - input.1
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    signal: &DeviceVec<R, f32>,
    signal_count: u32,
    lambda: f32,
    iterations: usize,
) -> common::Result<DeviceVec<R, f32>> {
    let n = graph.vertex_count();
    assert!(n != 0);
    assert!(signal_count != 0);
    assert!(lambda >= 0.0);
    assert_eq!(signal.len(), n.checked_mul(signal_count).unwrap());
    if graph.edge_count() == 0 {
        let output = exec.alloc::<f32>(signal.len());
        vector::copy(exec, signal.slice(..), output.slice_mut(..))?;
        return Ok(output);
    }

    let out_degree = common::resident_degrees(exec, graph)?;
    let in_degree = common::filled(exec, n as usize, 0u32)?;
    if graph.edge_count() != 0 {
        let edge_count = u32::try_from(graph.edge_count()).expect("edge count exceeds u32");
        vector::scatter_reduce(
            exec,
            lazy::constant(1u32).take(edge_count),
            graph.destinations().slice(..),
            0u32,
            SumU32,
            in_degree.slice_mut(..),
        )?;
    }
    let incidence = vector::map(
        exec,
        zip2(out_degree.slice(..), in_degree.slice(..)),
        AddU32,
    )?;
    let max_index = vector::max_element(exec, incidence.slice(..), common::LessU32)?
        .expect("the graph has at least one vertex");
    let max_incidence = exec.to_host(&incidence.slice(max_index..max_index + 1))?[0].max(1);
    let step = 0.9f32 / (2.0f32 * max_incidence as f32).sqrt();

    let edge_features = u32::try_from(graph.edge_count())
        .expect("edge count exceeds u32")
        .checked_mul(signal_count)
        .expect("edge-signal expansion exceeds u32");
    let split = vector::map(
        exec,
        zip2(
            common::counting_u32(0, edge_features as usize),
            lazy::constant(signal_count).take(edge_features),
        ),
        SplitIndex,
    )?;
    let (edge_ids, columns) = MStorage::into_columns(split);
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
            lazy::constant(signal_count).take(edge_features),
            columns.slice(..),
        ),
        MatrixIndex,
    )?;
    let destination_indices = vector::map(
        exec,
        zip3(
            destinations.slice(..),
            lazy::constant(signal_count).take(edge_features),
            columns.slice(..),
        ),
        MatrixIndex,
    )?;

    let mut current = exec.alloc::<f32>(signal.len());
    vector::copy(exec, signal.slice(..), current.slice_mut(..))?;
    let mut extrapolated = exec.alloc::<f32>(signal.len());
    vector::copy(exec, signal.slice(..), extrapolated.slice_mut(..))?;
    let mut dual = common::filled(exec, edge_features as usize, 0.0f32)?;

    for _ in 0..iterations {
        let source_values = vector::gather(
            exec,
            extrapolated.slice(..),
            common::indices(source_indices.slice(..)),
        )?;
        let destination_values = vector::gather(
            exec,
            extrapolated.slice(..),
            common::indices(destination_indices.slice(..)),
        )?;
        dual = vector::map(
            exec,
            zip5(
                dual.slice(..),
                source_values.slice(..),
                destination_values.slice(..),
                lazy::constant(step).take(edge_features),
                lazy::constant(lambda).take(edge_features),
            ),
            DualStep,
        )?;

        let divergence = common::filled(exec, signal.len() as usize, 0.0f32)?;
        vector::scatter_reduce(
            exec,
            dual.slice(..),
            source_indices.slice(..),
            0.0f32,
            SumF32,
            divergence.slice_mut(..),
        )?;
        vector::scatter_reduce(
            exec,
            lazy::map(dual.slice(..), Negate),
            destination_indices.slice(..),
            0.0f32,
            SumF32,
            divergence.slice_mut(..),
        )?;
        let next = vector::map(
            exec,
            massively::zip4(
                current.slice(..),
                divergence.slice(..),
                signal.slice(..),
                lazy::constant(step).take(signal.len()),
            ),
            PrimalStep,
        )?;
        extrapolated = vector::map(exec, zip2(next.slice(..), current.slice(..)), Extrapolate)?;
        current = next;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn smooths_each_signal_column_across_a_path() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 1, 3, 4], vec![1, 0, 2, 1]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let signal = exec.to_device(&[0.0f32, 10.0, 0.0, 10.0, 9.0, 1.0]);
        let output = solve(&exec, &graph, &signal, 2, 2.0, 80).unwrap();
        let output = exec.to_host(&output).unwrap();
        assert!(output[0] > 0.0 && output[4] < 9.0);
        assert!(output[1] < 10.0 && output[5] > 1.0);
    }
}
