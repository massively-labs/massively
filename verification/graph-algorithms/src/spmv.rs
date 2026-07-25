//! CSR sparse matrix-vector multiplication reduced independently by row.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, lazy, op::ReductionOp, op::UnaryOp, zip2};

use super::common::{self, DeviceWeightedCsr};

struct MulF32;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for MulF32 {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        input.0 * input.1
    }
}

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    matrix: &DeviceWeightedCsr<R>,
    vector: &DeviceVec<R, f32>,
) -> common::Result<DeviceVec<R, f32>> {
    assert_eq!(vector.len(), matrix.graph().vertex_count());
    let products = lazy::map(
        zip2(
            matrix.weights().slice(..),
            lazy::permute(vector.slice(..), matrix.graph().destinations().slice(..)),
        ),
        MulF32,
    );
    common::reduce_rows(exec, matrix.graph(), products, exec.value(0.0)?, SumF32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightedCsr;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_adjacency_multiplies_vector() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let matrix = WeightedCsr::new(common::path_graph(), vec![1.0; 6]);
        let matrix = DeviceWeightedCsr::<_, f32>::from_host(&exec, &matrix).unwrap();
        let vector = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]);
        assert_eq!(
            exec.to_host(&solve(&exec, &matrix, &vector).unwrap())
                .unwrap(),
            vec![2.0, 4.0, 6.0, 3.0]
        );
    }
}
