//! Degree-centrality Top K ranking.
//!
//! Scores are `out_degree + in_degree`. Results are ordered by decreasing
//! score and then increasing vertex identifier.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{BinaryPredicateOp, ReductionOp, UnaryOp},
    vector, zip2,
};

use super::common::{self, DeviceCsr};

pub struct Ranking<R: Runtime> {
    vertices: DeviceVec<R, u32>,
    scores: DeviceVec<R, u32>,
}

impl<R: Runtime> Ranking<R> {
    pub const fn vertices(&self) -> &DeviceVec<R, u32> {
        &self.vertices
    }

    pub const fn scores(&self) -> &DeviceVec<R, u32> {
        &self.scores
    }

    pub fn len(&self) -> u32 {
        self.vertices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct AddDegrees;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddDegrees {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

struct RankBefore;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for RankBefore {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.0 != rhs.0 {
            lhs.0 > rhs.0
        } else {
            lhs.1 < rhs.1
        })
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    k: u32,
) -> common::Result<Ranking<R>> {
    let n = graph.vertex_count();
    let take = u32::min(k, n);
    if n == 0 {
        return Ok(Ranking {
            vertices: exec.alloc::<u32>(0),
            scores: exec.alloc::<u32>(0),
        });
    }

    let out_degree = common::resident_degrees(exec, graph)?;
    let in_degree = common::filled(exec, n, 0u32)?;
    if graph.edge_count() != 0 {
        let edge_count =
            u32::try_from(graph.edge_count()).map_err(|_| massively::Error::LengthTooLarge {
                len: graph.edge_count(),
            })?;
        vector::scatter_reduce(
            exec,
            lazy::constant(1u32).take(edge_count),
            graph.destinations().slice(..),
            exec.value(0)?,
            SumU32,
            in_degree.slice_mut(..),
        )?;
    }
    let scores = vector::map(
        exec,
        zip2(out_degree.slice(..), in_degree.slice(..)),
        AddDegrees,
    )?;

    if take == 0 {
        return Ok(Ranking {
            vertices: exec.alloc::<u32>(0),
            scores: exec.alloc::<u32>(0),
        });
    }

    let sorted = vector::sort(
        exec,
        zip2(scores.slice(..), common::counting_u32(0, n as usize)),
        RankBefore,
    )?;
    let (sorted_scores, sorted_vertices) = MStorage::into_columns(sorted);
    let vertices = exec.alloc::<u32>(take);
    let scores = exec.alloc::<u32>(take);
    vector::copy(exec, sorted_vertices.slice(..take), vertices.slice_mut(..))?;
    vector::copy(exec, sorted_scores.slice(..take), scores.slice_mut(..))?;
    Ok(Ranking { vertices, scores })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn ranks_total_degree_with_vertex_tie_break() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 2, 3, 3], vec![1, 2, 2]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let ranking = solve(&exec, &graph, 3).unwrap();
        assert_eq!(exec.to_host(ranking.vertices()).unwrap(), vec![0, 1, 2]);
        assert_eq!(exec.to_host(ranking.scores()).unwrap(), vec![2, 2, 2]);
    }
}
