//! Unweighted Forman–Ricci curvature emitted in CSR edge order.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, lazy, op::UnaryOp, vector, zip2};

use super::common::{self, DeviceCsr};

struct Curvature;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Curvature {
    type Output = i32;

    fn apply(input: (u32, u32)) -> i32 {
        4i32 - i32::cast_from(input.0) - i32::cast_from(input.1)
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceVec<R, i32>> {
    let degree = common::resident_degrees(exec, graph)?;
    let sources = graph.segmentation().segment_ids(exec)?;
    vector::map(
        exec,
        zip2(
            lazy::permute(degree.slice(..), sources.slice(..)),
            lazy::permute(degree.slice(..), graph.destinations().slice(..)),
        ),
        Curvature,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn curvature_matches_endpoint_degrees() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        let output = solve(&exec, &graph).unwrap();
        assert_eq!(
            exec.to_host(&output).unwrap(),
            vec![-1, -1, -1, -2, -1, -1, -2, -1, -1, -1]
        );
    }
}
