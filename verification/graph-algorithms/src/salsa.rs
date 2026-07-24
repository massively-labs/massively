//! SALSA hub and authority random walks.
//!
//! A hub step chooses an outgoing edge uniformly and an authority step chooses
//! an incoming edge uniformly. Each half-step is L1-normalized, which also
//! makes the behavior explicit for directed graphs with dangling vertices.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, lazy,
    op::{ReductionOp, UnaryOp},
    vector, zip2,
};

use super::common::{self, DeviceCsr};

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct DivideByDegree;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for DivideByDegree {
    type Output = f32;

    fn apply(input: (f32, u32)) -> f32 {
        if input.1 == 0u32 {
            0.0f32
        } else {
            input.0 / input.1 as f32
        }
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
    let sum = vector::reduce(exec, values.slice(..), 0.0, SumF32)?;
    if sum == 0.0 {
        return Ok(values);
    }
    vector::map(
        exec,
        zip2(
            values.slice(..),
            lazy::constant(sum.recip()).take(values.len()),
        ),
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
    let initial = 1.0 / n as f32;
    let mut hubs = common::filled(exec, n as usize, initial)?;
    let mut authorities = common::filled(exec, n as usize, initial)?;
    if graph.edge_count() == 0 {
        return Ok((hubs, authorities));
    }

    let out_degree = common::resident_degrees(exec, graph)?;
    let in_degree = common::filled(exec, n as usize, 0u32)?;
    let edge_count =
        u32::try_from(graph.edge_count()).map_err(|_| massively::Error::LengthTooLarge {
            len: graph.edge_count(),
        })?;
    vector::scatter_reduce(
        exec,
        lazy::constant(1u32).take(edge_count),
        graph.destinations().slice(..),
        0u32,
        SumU32,
        in_degree.slice_mut(..),
    )?;
    let sources = graph.segmentation().segment_ids(exec)?;

    for _ in 0..iterations {
        authorities = common::filled(exec, n as usize, 0.0f32)?;
        let hub_shares = lazy::map(
            zip2(
                lazy::permute(hubs.slice(..), sources.slice(..)),
                lazy::permute(out_degree.slice(..), sources.slice(..)),
            ),
            DivideByDegree,
        );
        vector::scatter_reduce(
            exec,
            hub_shares,
            graph.destinations().slice(..),
            0.0,
            SumF32,
            authorities.slice_mut(..),
        )?;
        authorities = normalize(exec, authorities)?;

        let authority_shares = lazy::map(
            zip2(
                lazy::permute(authorities.slice(..), graph.destinations().slice(..)),
                lazy::permute(in_degree.slice(..), graph.destinations().slice(..)),
            ),
            DivideByDegree,
        );
        hubs = common::reduce_rows(exec, graph, authority_shares, 0.0f32, SumF32)?;
        hubs = normalize(exec, hubs)?;
    }

    Ok((hubs, authorities))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn favors_the_shared_authority() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 1, 2, 2], vec![2, 2]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let (hubs, authorities) = solve(&exec, &graph, 3).unwrap();
        assert_eq!(exec.to_host(&hubs).unwrap(), vec![0.5, 0.5, 0.0]);
        assert_eq!(exec.to_host(&authorities).unwrap(), vec![0.0, 0.0, 1.0]);
    }
}
