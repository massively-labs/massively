//! HITS with device-resident hub and authority vectors.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, lazy, op::ReductionOp, op::UnaryOp, vector, zip2};

use super::common::{self, DeviceCsr};

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

struct Square;

#[cubecl::cube]
impl UnaryOp<f32> for Square {
    type Output = f32;

    fn apply(value: f32) -> f32 {
        value * value
    }
}

struct Scale;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for Scale {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        input.0 * input.1
    }
}

fn normalize<R: Runtime>(
    exec: &Executor<R>,
    values: DeviceVec<R, f32>,
) -> common::Result<DeviceVec<R, f32>> {
    let len = values.len();
    let norm_squared = vector::reduce(
        exec,
        lazy::map(values.slice(..), Square),
        exec.value(0.0)?,
        SumF32,
    )?
    .read(exec)?;
    let scale = if norm_squared == 0.0 {
        1.0
    } else {
        norm_squared.sqrt().recip()
    };
    vector::map(
        exec,
        zip2(values.slice(..), lazy::constant(scale).take(len)),
        Scale,
    )
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    iterations: usize,
) -> common::Result<(DeviceVec<R, f32>, DeviceVec<R, f32>)> {
    let n = graph.vertex_count();
    assert!(n != 0);
    let mut hubs = common::filled(exec, n, 1.0f32)?;
    let mut authorities = common::filled(exec, n, 1.0f32)?;
    let zero = 0.0f32;
    let zero_value = exec.value(zero)?;
    let sources = graph.segmentation().segment_ids(exec)?;

    for _ in 0..iterations {
        authorities = common::filled(exec, n, zero)?;
        vector::scatter_reduce(
            exec,
            lazy::permute(hubs.slice(..), sources.slice(..)),
            graph.destinations().slice(..),
            zero_value.clone(),
            SumF32,
            authorities.slice_mut(..),
        )?;
        authorities = normalize(exec, authorities)?;

        hubs = common::reduce_rows(
            exec,
            graph,
            lazy::permute(authorities.slice(..), graph.destinations().slice(..)),
            zero_value.clone(),
            SumF32,
        )?;
        hubs = normalize(exec, hubs)?;
    }

    Ok((hubs, authorities))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn vectors_are_normalized() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        let (hubs, authorities) = solve(&exec, &graph, 4).unwrap();
        let hubs = exec.to_host(&hubs).unwrap();
        let authorities = exec.to_host(&authorities).unwrap();
        let hub_norm = hubs.iter().map(|value| value * value).sum::<f32>().sqrt();
        let authority_norm = authorities
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((hub_norm - 1.0).abs() < 1e-5);
        assert!((authority_norm - 1.0).abs() < 1e-5);
    }
}
