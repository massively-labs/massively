//! Personalized PageRank with device-resident rank and degree vectors.

use cubecl::prelude::Runtime;
use massively::{DeviceVec, Executor, vector};

use super::common::{self, DeviceCsr};

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    source: u32,
    damping: f32,
    iterations: usize,
) -> common::Result<DeviceVec<R, f32>> {
    let n = graph.vertex_count();
    assert!(n != 0);
    assert!(source < n);
    let degree = common::resident_degrees(exec, graph)?;
    let source_index = common::filled(exec, 1, source)?;
    let mut rank = common::filled(exec, n as usize, 1.0 / n as f32)?;

    for _ in 0..iterations {
        let dangling = common::dangling_mass(exec, &rank, &degree)?;
        let base = damping * dangling / n as f32;
        let output = common::filled(exec, n as usize, base)?;
        let source_rank = common::filled(exec, 1, base + 1.0 - damping)?;
        vector::scatter(
            exec,
            source_rank.slice(..),
            common::indices(source_index.slice(..)),
            output.slice_mut(..),
        )?;
        common::accumulate_rank(exec, graph, &degree, &rank, damping, &output)?;
        rank = output;
    }

    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn probability_mass_is_preserved() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        let rank = solve(&exec, &graph, 0, 0.85, 20).unwrap();
        let rank = exec.to_host(&rank).unwrap();
        assert!((rank.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(rank[0] > 0.15);
    }
}
