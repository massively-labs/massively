//! Batched uniform random walks.
//!
//! One or more walkers start at every vertex.  At each step a supplied random
//! word selects `word % degree(current)` from the current CSR row.  A walker
//! that reaches a zero-degree vertex is terminated and subsequent path entries
//! are `u32::MAX`.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, lazy,
    op::{Identity, UnaryOp},
    util::random,
    vector, zip2, zip3,
};

use super::common::{self, DeviceCsr};

pub struct Walks<R: Runtime> {
    vertices: DeviceVec<R, u32>,
    walker_count: u32,
    walk_length: u32,
}

impl<R: Runtime> Walks<R> {
    /// Walker-major paths: `vertices[walker * walk_length + step]`.
    pub const fn vertices(&self) -> &DeviceVec<R, u32> {
        &self.vertices
    }

    pub const fn walker_count(&self) -> u32 {
        self.walker_count
    }

    pub const fn walk_length(&self) -> u32 {
        self.walk_length
    }
}

struct WalkerStart;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for WalkerStart {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 % input.1
    }
}

struct Valid;

#[cubecl::cube]
impl UnaryOp<u32> for Valid {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        if input != u32::MAX { 1u32 } else { 0u32 }
    }
}

struct Positive;

#[cubecl::cube]
impl UnaryOp<u32> for Positive {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        if input != 0u32 { 1u32 } else { 0u32 }
    }
}

struct PathIndex;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for PathIndex {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        input.0 * input.1 + input.2
    }
}

struct ChoiceIndex;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for ChoiceIndex {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        input.0 * input.1 + input.2
    }
}

struct SelectEdge;

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for SelectEdge {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        input.0 + input.2 % input.1
    }
}

/// Generates deterministic GPU-side random words and runs uniform walks.
pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    walk_length: u32,
    walks_per_vertex: u32,
    seed: u64,
) -> common::Result<Walks<R>> {
    assert!(walk_length != 0);
    let walker_count = graph
        .vertex_count()
        .checked_mul(walks_per_vertex)
        .expect("walker count exceeds u32");
    let choice_count = walker_count
        .checked_mul(walk_length.saturating_sub(1))
        .expect("random-choice count exceeds u32");
    let choices = vector::map(
        exec,
        random::uniform_u32(0, u32::MAX, seed)?.take(choice_count),
        Identity,
    )?;
    solve_with_choices(exec, graph, walk_length, walks_per_vertex, &choices)
}

/// Runs walks from caller-supplied random words.
///
/// This entry point makes random-walk semantics independently testable.  The
/// choice layout is walker-major and contains `walk_length - 1` words per
/// walker.
pub fn solve_with_choices<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    walk_length: u32,
    walks_per_vertex: u32,
    choices: &DeviceVec<R, u32>,
) -> common::Result<Walks<R>> {
    assert!(graph.vertex_count() != 0);
    assert!(walk_length != 0);
    assert!(walks_per_vertex != 0);
    let walker_count = graph
        .vertex_count()
        .checked_mul(walks_per_vertex)
        .expect("walker count exceeds u32");
    let transitions = walk_length - 1;
    let choice_count = walker_count
        .checked_mul(transitions)
        .expect("random-choice count exceeds u32");
    assert_eq!(choices.len(), choice_count);
    let path_count = walker_count
        .checked_mul(walk_length)
        .expect("walk output exceeds u32");

    let mut current = vector::map(
        exec,
        zip2(
            common::counting_u32(0, walker_count as usize),
            lazy::constant(graph.vertex_count()).take(walker_count),
        ),
        WalkerStart,
    )?;
    let paths = common::filled(exec, path_count, u32::MAX)?;
    let degrees = common::resident_degrees(exec, graph)?;

    for step in 0..walk_length {
        let path_indices = lazy::map(
            zip3(
                common::counting_u32(0, walker_count as usize),
                lazy::constant(walk_length).take(walker_count),
                lazy::constant(step).take(walker_count),
            ),
            PathIndex,
        );
        vector::scatter(
            exec,
            current.slice(..),
            common::indices(path_indices),
            paths.slice_mut(..),
        )?;
        if step == transitions {
            break;
        }

        let active_stencil = vector::map(exec, current.slice(..), Valid)?;
        let active_positions = common::materialize_exact(
            exec,
            vector::copy_where(
                exec,
                common::counting_u32(0, walker_count as usize),
                common::stencil(active_stencil.slice(..)),
            )?,
        )?;
        if active_positions.len() == 0 {
            break;
        }
        let active_vertices = vector::gather(
            exec,
            current.slice(..),
            common::indices(active_positions.slice(..)),
        )?;
        let degree = vector::gather(exec, degrees.slice(..), active_vertices.slice(..))?;
        let live_stencil = vector::map(exec, degree.slice(..), Positive)?;
        let active_count = active_positions.len();
        let live_indices = common::materialize_exact(
            exec,
            vector::copy_where(
                exec,
                common::counting_u32(0, active_count as usize),
                common::stencil(live_stencil.slice(..)),
            )?,
        )?;
        let next = common::filled(exec, walker_count, u32::MAX)?;
        if live_indices.len() != 0 {
            let live_positions = vector::gather(
                exec,
                active_positions.slice(..),
                common::indices(live_indices.slice(..)),
            )?;
            let live_vertices = vector::gather(
                exec,
                active_vertices.slice(..),
                common::indices(live_indices.slice(..)),
            )?;
            let live_degree = vector::gather(
                exec,
                degree.slice(..),
                common::indices(live_indices.slice(..)),
            )?;
            let offsets = vector::gather(
                exec,
                graph.offsets().slice(..),
                common::indices(live_vertices.slice(..)),
            )?;
            let live_count = live_positions.len();
            let choice_indices = vector::map(
                exec,
                zip3(
                    live_positions.slice(..),
                    lazy::constant(transitions).take(live_count),
                    lazy::constant(step).take(live_count),
                ),
                ChoiceIndex,
            )?;
            let random_words = vector::gather(
                exec,
                choices.slice(..),
                common::indices(choice_indices.slice(..)),
            )?;
            let edge_indices = vector::map(
                exec,
                zip3(
                    offsets.slice(..),
                    live_degree.slice(..),
                    random_words.slice(..),
                ),
                SelectEdge,
            )?;
            let destinations = vector::gather(
                exec,
                graph.destinations().slice(..),
                common::indices(edge_indices.slice(..)),
            )?;
            vector::scatter(
                exec,
                destinations.slice(..),
                common::indices(live_positions.slice(..)),
                next.slice_mut(..),
            )?;
        }
        current = next;
    }

    Ok(Walks {
        vertices: paths,
        walker_count,
        walk_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn supplied_choices_define_exact_paths() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::path_graph()).unwrap();
        let choices = exec.to_device(&[0u32; 8]);
        let walks = solve_with_choices(&exec, &graph, 3, 1, &choices).unwrap();
        assert_eq!(walks.walker_count(), 4);
        assert_eq!(walks.walk_length(), 3);
        assert_eq!(
            exec.to_host(walks.vertices()).unwrap(),
            vec![0, 1, 0, 1, 0, 1, 2, 1, 0, 3, 2, 1]
        );
    }
}
