//! Iterative geolocation with device-resident coordinates and known flags.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, MVec, lazy, op::ReductionOp, op::UnaryOp, vector, zip2,
};

use super::common::{self, DeviceCsr};

struct SumCoordinates;

#[cubecl::cube]
impl ReductionOp<(f32, f32)> for SumCoordinates {
    fn apply(lhs: (f32, f32), rhs: (f32, f32)) -> (f32, f32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

struct UpdateCoordinates;

#[cubecl::cube]
impl UnaryOp<(f32, f32, f32, f32, u32, u32)> for UpdateCoordinates {
    type Output = (f32, f32);

    fn apply(input: (f32, f32, f32, f32, u32, u32)) -> (f32, f32) {
        let coordinates = (input.0, input.1);
        let sums = (input.2, input.3);
        let degree = input.4;
        let known = input.5;
        if known != 0u32 || degree == 0u32 {
            coordinates
        } else {
            (sums.0 / degree as f32, sums.1 / degree as f32)
        }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    initial: &MVec<R, (f32, f32)>,
    known: &DeviceVec<R, u32>,
    iterations: usize,
) -> common::Result<MVec<R, (f32, f32)>> {
    let n = graph.vertex_count();
    assert_eq!(initial.len()?, n);
    assert_eq!(known.len(), n);
    let degree = common::resident_degrees(exec, graph)?;
    let mut coordinates = initial.clone();
    let zero = (0.0, 0.0);

    for _ in 0..iterations {
        let sums = common::reduce_rows(
            exec,
            graph,
            lazy::permute(coordinates.slice(..), graph.destinations().slice(..)),
            zero,
            SumCoordinates,
        )?;

        coordinates = vector::map(
            exec,
            zip2(
                zip2(coordinates.slice(..), sums.slice(..)),
                zip2(degree.slice(..), known.slice(..)),
            ),
            UpdateCoordinates,
        )?;
    }

    Ok(coordinates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn unknown_center_is_neighbor_mean() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = common::CsrGraph::new(vec![0, 2, 3, 4], vec![1, 2, 0, 0]);
        let graph = DeviceCsr::from_host(&exec, &graph).unwrap();
        let latitude = exec.to_device(&[0.0f32, 0.0, 2.0]);
        let longitude = exec.to_device(&[0.0f32, 0.0, 2.0]);
        let initial = vector::map(
            &exec,
            zip2(latitude.slice(..), longitude.slice(..)),
            massively::op::Identity,
        )
        .unwrap();
        let known = exec.to_device(&[0u32, 1, 1]);
        let result = solve(&exec, &graph, &initial, &known, 1).unwrap();
        let (latitude, longitude) = MStorage::into_columns(result);
        assert_eq!(exec.to_host(&latitude).unwrap(), vec![1.0, 0.0, 2.0]);
        assert_eq!(exec.to_host(&longitude).unwrap(), vec![1.0, 0.0, 2.0]);
    }
}
