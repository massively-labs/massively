//! Personalized Who-To-Follow recommendations.
//!
//! Personalized PageRank first selects a circle of trust. SALSA then reranks
//! the induced circle subgraph by authority score. The source is never emitted.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{BinaryPredicateOp, UnaryOp},
    vector, zip2, zip3,
};

use super::{
    common::{self, DeviceCsr},
    ppr, salsa,
};

pub struct Recommendations<R: Runtime> {
    vertices: DeviceVec<R, u32>,
    scores: DeviceVec<R, f32>,
}

impl<R: Runtime> Recommendations<R> {
    pub const fn vertices(&self) -> &DeviceVec<R, u32> {
        &self.vertices
    }

    pub const fn scores(&self) -> &DeviceVec<R, f32> {
        &self.scores
    }

    pub fn len(&self) -> u32 {
        self.vertices.len()
    }
}

struct PprBefore;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for PprBefore {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.0 != rhs.0 {
            lhs.0 > rhs.0
        } else {
            lhs.1 < rhs.1
        })
    }
}

struct NotSource;

#[cubecl::cube]
impl UnaryOp<(f32, u32, u32)> for NotSource {
    type Output = u32;

    fn apply(input: (f32, u32, u32)) -> u32 {
        if input.1 != input.2 { 1u32 } else { 0u32 }
    }
}

struct BothMembers;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for BothMembers {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != 0u32 && input.1 != 0u32 {
            1u32
        } else {
            0u32
        }
    }
}

struct RecommendationBefore;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, f32, u32)> for RecommendationBefore {
    fn apply(lhs: (f32, f32, u32), rhs: (f32, f32, u32)) -> massively::MFlag {
        massively::flag::from_bool(if lhs.0 != rhs.0 {
            lhs.0 > rhs.0
        } else if lhs.1 != rhs.1 {
            lhs.1 > rhs.1
        } else {
            lhs.2 < rhs.2
        })
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    source: u32,
    circle_size: u32,
    recommendation_count: u32,
    damping: f32,
    ppr_iterations: usize,
    salsa_iterations: usize,
) -> common::Result<Recommendations<R>> {
    let n = graph.vertex_count();
    assert!(source < n);
    let circle_size = u32::min(circle_size, n.saturating_sub(1));
    let recommendation_count = u32::min(recommendation_count, circle_size);
    if recommendation_count == 0 {
        return Ok(Recommendations {
            vertices: exec.alloc::<u32>(0),
            scores: exec.alloc::<f32>(0),
        });
    }

    let personalized = ppr::solve(exec, graph, source, damping, ppr_iterations)?;
    let ranked = vector::sort(
        exec,
        zip2(personalized.slice(..), common::counting_u32(0, n as usize)),
        PprBefore,
    )?;
    let keep = vector::map(
        exec,
        zip2(ranked.slice(..), lazy::constant(source).take(n)),
        NotSource,
    )?;
    let candidates = common::materialize_exact(
        exec,
        vector::copy_where(exec, ranked.slice(..), common::stencil(keep.slice(..)))?,
    )?;
    let (_circle_ppr_all, circle_vertices_all) = MStorage::into_columns(candidates);
    let circle_vertices = exec.alloc::<u32>(circle_size);
    vector::copy(
        exec,
        circle_vertices_all.slice(..circle_size),
        circle_vertices.slice_mut(..),
    )?;

    let member = common::filled(exec, n, 0u32)?;
    vector::scatter(
        exec,
        lazy::constant(1u32).take(circle_size),
        common::indices(circle_vertices.slice(..)),
        member.slice_mut(..),
    )?;
    let sources = graph.segmentation().segment_ids(exec)?;
    let source_member = vector::gather(exec, member.slice(..), common::indices(sources.slice(..)))?;
    let destination_member = vector::gather(
        exec,
        member.slice(..),
        common::indices(graph.destinations().slice(..)),
    )?;
    let induced_stencil = vector::map(
        exec,
        zip2(source_member.slice(..), destination_member.slice(..)),
        BothMembers,
    )?;
    let induced_edges = common::materialize_exact(
        exec,
        vector::copy_where(
            exec,
            zip2(sources.slice(..), graph.destinations().slice(..)),
            common::stencil(induced_stencil.slice(..)),
        )?,
    )?;
    let (induced_sources, induced_destinations) = MStorage::into_columns(induced_edges);
    let induced_weights = common::filled(exec, induced_sources.len(), 1u32)?;
    let induced = common::weighted_csr_from_edges(
        exec,
        n,
        induced_sources.slice(..),
        induced_destinations.slice(..),
        induced_weights.slice(..),
    )?
    .into_parts()
    .0;
    let (_hubs, authorities) = salsa::solve(exec, &induced, usize::max(salsa_iterations, 1))?;
    let circle_authority = vector::gather(
        exec,
        authorities.slice(..),
        common::indices(circle_vertices.slice(..)),
    )?;
    let circle_ppr = vector::gather(
        exec,
        personalized.slice(..),
        common::indices(circle_vertices.slice(..)),
    )?;
    let recommendations = vector::sort(
        exec,
        zip3(
            circle_authority.slice(..),
            circle_ppr.slice(..),
            circle_vertices.slice(..),
        ),
        RecommendationBefore,
    )?;
    let (authority, _ppr, vertices) = MStorage::into_columns(recommendations);
    let output_vertices = exec.alloc::<u32>(recommendation_count);
    let output_scores = exec.alloc::<f32>(recommendation_count);
    vector::copy(
        exec,
        vertices.slice(..recommendation_count),
        output_vertices.slice_mut(..),
    )?;
    vector::copy(
        exec,
        authority.slice(..recommendation_count),
        output_scores.slice_mut(..),
    )?;
    Ok(Recommendations {
        vertices: output_vertices,
        scores: output_scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn reranks_the_circle_toward_a_shared_authority() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 2, 3, 4, 4], vec![1, 2, 3, 3]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let output = solve(&exec, &graph, 0, 3, 2, 0.85, 20, 5).unwrap();
        let vertices = exec.to_host(output.vertices()).unwrap();
        assert_eq!(vertices.len(), 2);
        assert_eq!(vertices[0], 3);
        assert!(!vertices.contains(&0));
    }
}
