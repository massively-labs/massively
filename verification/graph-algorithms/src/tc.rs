//! Triangle counting by segment intersection over every directed CSR edge.

use cubecl::prelude::*;
use massively::{
    Executor, lazy,
    op::{ReductionOp, UnaryOp},
    seg::Segment,
    vector, zip2,
};

use super::common::{self, DeviceCsr};

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct IntersectCount;

#[cubecl::cube]
impl UnaryOp<(Segment<u32>, Segment<u32>)> for IntersectCount {
    type Output = u32;

    fn apply(input: (Segment<u32>, Segment<u32>)) -> u32 {
        let left = RuntimeCell::<u32>::new(0u32);
        let right = RuntimeCell::<u32>::new(0u32);
        let count = RuntimeCell::<u32>::new(0u32);
        while left.read() < input.0.len() && right.read() < input.1.len() {
            let lhs = input.0.at(left.read());
            let rhs = input.1.at(right.read());
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
        count.read()
    }
}

pub fn solve<R: Runtime>(exec: &Executor<R>, graph: &DeviceCsr<R>) -> common::Result<u32> {
    if graph.edge_count() == 0 {
        return Ok(0);
    }
    let sources = graph.segmentation().segment_ids(exec)?;
    let rows = graph
        .segmentation()
        .segments(graph.destinations().slice(..))?;
    let counts = vector::map(
        exec,
        zip2(
            lazy::permute(rows.clone(), sources.slice(..)),
            lazy::permute(rows, graph.destinations().slice(..)),
        ),
        IntersectCount,
    )?;
    Ok(vector::reduce(exec, counts.slice(..), 0, SumU32)? / 6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn shared_edge_forms_two_triangles() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        assert_eq!(solve(&exec, &graph).unwrap(), 2);
    }
}
