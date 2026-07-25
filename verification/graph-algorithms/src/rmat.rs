//! Deterministic GPU-side R-MAT graph generation.
//!
//! Random choices are laid out level-major. Duplicate edges are coalesced,
//! rows are sorted, and optional symmetrization happens before self-loop
//! removal and coalescing.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, lazy,
    op::{Identity, UnaryOp},
    util::random,
    vector, zip2, zip7,
};

use super::common::{self, DeviceCsr};

struct Quadrant;

#[cubecl::cube]
impl UnaryOp<(u32, u32, f32, u32, f32, f32, f32)> for Quadrant {
    type Output = (u32, u32);

    fn apply(input: (u32, u32, f32, u32, f32, f32, f32)) -> Self::Output {
        let source = RuntimeCell::<u32>::new(input.0);
        let destination = RuntimeCell::<u32>::new(input.1);
        if input.2 < input.4 {
        } else if input.2 < input.4 + input.5 {
            destination.store(destination.read() | input.3);
        } else if input.2 < input.4 + input.5 + input.6 {
            source.store(source.read() | input.3);
        } else {
            source.store(source.read() | input.3);
            destination.store(destination.read() | input.3);
        }
        (source.read(), destination.read())
    }
}

struct NotSelf;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for NotSelf {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != input.1 { 1u32 } else { 0u32 }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    scale: u32,
    edge_count: u32,
    probabilities: [f32; 4],
    seed: u64,
    undirected: bool,
    remove_self_loops: bool,
) -> common::Result<DeviceCsr<R>> {
    let choice_count = edge_count
        .checked_mul(scale)
        .expect("R-MAT random choice count exceeds u32");
    let choices = vector::map(
        exec,
        random::uniform_f32(0.0, 1.0, seed)?.take(choice_count),
        Identity,
    )?;
    solve_with_choices(
        exec,
        scale,
        edge_count,
        probabilities,
        &choices,
        undirected,
        remove_self_loops,
    )
}

pub fn solve_with_choices<R: Runtime>(
    exec: &Executor<R>,
    scale: u32,
    edge_count: u32,
    probabilities: [f32; 4],
    choices: &DeviceVec<R, f32>,
    undirected: bool,
    remove_self_loops: bool,
) -> common::Result<DeviceCsr<R>> {
    assert!(scale < 32);
    assert!(probabilities.iter().all(|&value| value >= 0.0));
    assert!((probabilities.iter().sum::<f32>() - 1.0).abs() <= 1.0e-5);
    assert_eq!(
        choices.len(),
        edge_count
            .checked_mul(scale)
            .expect("R-MAT random choice count exceeds u32")
    );
    let vertex_count = 1u32 << scale;
    let mut sources = common::filled(exec, edge_count as usize, 0u32)?;
    let mut destinations = common::filled(exec, edge_count as usize, 0u32)?;

    for level in 0..scale {
        let begin = level * edge_count;
        let bit = 1u32 << (scale - level - 1);
        let pairs = vector::map(
            exec,
            zip7(
                sources.slice(..),
                destinations.slice(..),
                choices.slice(begin..begin + edge_count),
                lazy::constant(bit).take(edge_count),
                lazy::constant(probabilities[0]).take(edge_count),
                lazy::constant(probabilities[1]).take(edge_count),
                lazy::constant(probabilities[2]).take(edge_count),
            ),
            Quadrant,
        )?;
        (sources, destinations) = MStorage::into_columns(pairs);
    }

    if undirected && edge_count != 0 {
        let doubled = edge_count
            .checked_mul(2)
            .expect("symmetrized R-MAT edge count exceeds u32");
        let symmetric_sources = exec.alloc::<u32>(doubled);
        let symmetric_destinations = exec.alloc::<u32>(doubled);
        vector::copy(
            exec,
            sources.slice(..),
            symmetric_sources.slice_mut(..edge_count),
        )?;
        vector::copy(
            exec,
            destinations.slice(..),
            symmetric_destinations.slice_mut(..edge_count),
        )?;
        vector::copy(
            exec,
            destinations.slice(..),
            symmetric_sources.slice_mut(edge_count..),
        )?;
        vector::copy(
            exec,
            sources.slice(..),
            symmetric_destinations.slice_mut(edge_count..),
        )?;
        sources = symmetric_sources;
        destinations = symmetric_destinations;
    }

    if remove_self_loops && sources.len() != 0 {
        let keep = vector::map(
            exec,
            zip2(sources.slice(..), destinations.slice(..)),
            NotSelf,
        )?;
        let pairs = common::materialize_exact(
            exec,
            vector::copy_where(
                exec,
                zip2(sources.slice(..), destinations.slice(..)),
                common::stencil(keep.slice(..)),
            )?,
        )?;
        (sources, destinations) = MStorage::into_columns(pairs);
    }

    let weights = common::filled(exec, sources.len() as usize, 1u32)?;
    let weighted = common::weighted_csr_from_edges(
        exec,
        vertex_count,
        sources.slice(..),
        destinations.slice(..),
        weights.slice(..),
    )?;
    Ok(weighted.into_parts().0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn supplied_quadrants_create_sorted_unique_symmetric_edges() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        // Four edges, two levels. These choices generate 0->1, 1->2,
        // 2->3, and 3->0 before symmetrization.
        let choices = exec.to_device(&[
            0.1f32, 0.3, 0.9, 0.6, // high bits
            0.3, 0.6, 0.3, 0.6, // low bits
        ]);
        let graph = solve_with_choices(&exec, 2, 4, [0.25, 0.25, 0.25, 0.25], &choices, true, true)
            .unwrap();
        assert_eq!(exec.to_host(&graph.offsets()).unwrap(), vec![0, 2, 4, 6, 8]);
        assert_eq!(
            exec.to_host(graph.destinations()).unwrap(),
            vec![1, 3, 0, 2, 1, 3, 0, 2]
        );
    }
}
