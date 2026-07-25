//! Kruskal minimum spanning forest over resident weighted CSR storage.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MVec, lazy,
    op::{BinaryPredicateOp, Identity, UnaryOp},
    vector, zip2, zip3,
};

use super::common::{self, DeviceWeightedCsr};

struct LessF32;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for LessF32 {
    fn apply(lhs: f32, rhs: f32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

struct EqualPair;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for EqualPair {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 == input.1 { 1u32 } else { 0u32 }
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
    graph: &DeviceWeightedCsr<R, f32>,
) -> common::Result<MVec<R, (u32, u32, f32)>> {
    let topology = graph.graph();
    let edge_count = topology.destinations().len();
    let sources = topology.segmentation().segment_ids(exec)?;
    let edge_order = vector::sort_by_key(
        exec,
        graph.weights().slice(..),
        common::counting_u32(0, edge_count as usize),
        LessF32,
    )?;
    let components = vector::map(
        exec,
        common::counting_u32(0, topology.vertex_count() as usize),
        Identity,
    )?;
    let selected = common::filled(exec, edge_count, 0u32)?;

    for position in 0..edge_count {
        let edge = read_u32(exec, &edge_order, position)?;
        let source = read_u32(exec, &sources, edge)?;
        let destination = read_u32(exec, topology.destinations(), edge)?;
        let source_component = read_u32(exec, &components, source)?;
        let destination_component = read_u32(exec, &components, destination)?;
        if source_component == destination_component {
            continue;
        }

        let stencil = vector::map(
            exec,
            zip2(
                components.slice(..),
                lazy::constant(destination_component).take(topology.vertex_count()),
            ),
            EqualPair,
        )?;
        vector::replace_where(
            exec,
            exec.value(source_component)?,
            common::stencil(stencil.slice(..)),
            components.slice_mut(..),
        )?;
        vector::scatter(
            exec,
            lazy::constant(1u32).take(1),
            common::indices(lazy::constant(edge).take(1)),
            selected.slice_mut(..),
        )?;
    }

    common::materialize_exact(
        exec,
        vector::copy_where(
            exec,
            zip3(
                sources.slice(..),
                topology.destinations().slice(..),
                graph.weights().slice(..),
            ),
            common::stencil(selected.slice(..)),
        )?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightedCsr;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use massively::MStorage;

    #[test]
    fn weighted_path_is_its_own_tree() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = WeightedCsr::new(common::path_graph(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let graph = DeviceWeightedCsr::from_host(&exec, &graph).unwrap();
        let tree = solve(&exec, &graph).unwrap();
        let (sources, _, weights) = MStorage::into_columns(tree);
        assert_eq!(sources.len(), 3);
        assert_eq!(exec.to_host(&weights).unwrap().iter().sum::<f32>(), 6.0);
    }
}
