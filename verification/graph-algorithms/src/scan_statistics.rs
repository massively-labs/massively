//! Vertex scan statistics from degree and incident triangle participation.
//!
//! For every vertex this returns its degree plus the sum, over its outgoing
//! edges, of the number of common neighbors of the edge endpoints. Adjacency
//! rows must be sorted. On a symmetric simple graph the second term is twice
//! the number of triangles incident to the vertex.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, lazy,
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

struct AddU32;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddU32 {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
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

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceVec<R, u32>> {
    let degree = common::resident_degrees(exec, graph)?;
    if graph.edge_count() == 0 {
        return Ok(degree);
    }
    let sources = graph.segmentation().segment_ids(exec)?;
    let rows = graph
        .segmentation()
        .segments(graph.destinations().slice(..))?;
    let common_neighbors = vector::map(
        exec,
        zip2(
            lazy::permute(rows.clone(), sources.slice(..)),
            lazy::permute(rows, graph.destinations().slice(..)),
        ),
        IntersectCount,
    )?;
    let incident = common::reduce_rows(exec, graph, common_neighbors.slice(..), 0u32, SumU32)?;
    vector::map(exec, zip2(degree.slice(..), incident.slice(..)), AddU32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn adds_two_per_incident_triangle() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        let output = solve(&exec, &graph).unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![4, 7, 7, 4]);
    }
}
