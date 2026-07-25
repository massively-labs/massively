//! Degree-ordered greedy graph coloring over resident CSR storage.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, lazy,
    op::{BinaryPredicateOp, PredicateOp},
    vector, zip2,
};

use super::common::{self, DeviceCsr};

struct GreaterU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for GreaterU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs > rhs)
    }
}

struct EqualPair;

#[cubecl::cube]
impl PredicateOp<(u32, u32)> for EqualPair {
    fn apply(input: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(input.0 == input.1)
    }
}

fn read_u32<R: Runtime>(
    exec: &Executor<R>,
    values: &DeviceVec<R, u32>,
    index: u32,
) -> common::Result<u32> {
    Ok(exec.to_host(&values.slice(index..index + 1))?[0])
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.vertex_count();
    let degree = common::resident_degrees(exec, graph)?;
    let order = vector::sort_by_key(
        exec,
        degree.slice(..),
        common::counting_u32(0, n as usize),
        GreaterU32,
    )?;
    let colors = common::filled(exec, n, u32::MAX)?;

    for position in 0..n as usize {
        let vertex = read_u32(exec, &order, position as u32)?;
        let offsets = exec.to_host(&graph.offsets().slice(vertex..vertex + 2))?;
        let neighbors = graph.destinations().slice(offsets[0]..offsets[1]);
        let neighbor_count = offsets[1] - offsets[0];

        let mut color = 0u32;
        loop {
            let used = vector::count_if(
                exec,
                zip2(
                    lazy::permute(colors.slice(..), common::indices(neighbors.clone())),
                    lazy::constant(color).take(neighbor_count),
                ),
                EqualPair,
            )?
            .read(exec)?;
            if used == 0 {
                break;
            }
            color += 1;
        }

        vector::scatter(
            exec,
            lazy::constant(color).take(1),
            common::indices(lazy::constant(vertex).take(1)),
            colors.slice_mut(..),
        )?;
    }

    Ok(colors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn adjacent_vertices_have_distinct_colors() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host_graph = common::sample_graph();
        let graph = DeviceCsr::from_host(&exec, &host_graph).unwrap();
        let output = solve(&exec, &graph).unwrap();
        let colors = exec.to_host(&output).unwrap();
        for source in 0..host_graph.vertex_count() {
            for &destination in host_graph.row(source) {
                assert_ne!(colors[source], colors[destination as usize]);
            }
        }
        assert_eq!(colors.iter().copied().max().unwrap() + 1, 3);
    }
}
