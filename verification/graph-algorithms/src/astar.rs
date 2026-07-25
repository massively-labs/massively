//! A* shortest path with caller-supplied admissible integer heuristics.
//!
//! The open set is represented by resident flags and the minimum `f = g + h`
//! item is selected by a generic reduction. Improved closed vertices are
//! reopened, so admissible but inconsistent heuristics remain correct.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, lazy, op::UnaryOp, vector, zip2};

use super::common::{self, DeviceWeightedCsr};

const INF: u32 = 1_000_000_000;

pub struct Path<R: Runtime> {
    vertices: DeviceVec<R, u32>,
    cost: Option<u32>,
    expanded: u32,
}

impl<R: Runtime> Path<R> {
    pub const fn vertices(&self) -> &DeviceVec<R, u32> {
        &self.vertices
    }

    pub const fn cost(&self) -> Option<u32> {
        self.cost
    }

    pub const fn expanded(&self) -> u32 {
        self.expanded
    }

    pub fn is_reachable(&self) -> bool {
        self.cost.is_some()
    }
}

struct AddScore;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddScore {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 >= INF || input.1 >= INF - input.0 {
            INF
        } else {
            input.0 + input.1
        }
    }
}

struct AddDistance;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddDistance {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 >= INF {
            INF
        } else if input.1 >= INF - input.0 {
            INF
        } else {
            input.0 + input.1
        }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
    source: u32,
    target: u32,
    heuristic: &DeviceVec<R, u32>,
) -> common::Result<Path<R>> {
    let n = graph.graph().vertex_count();
    assert!(source < n);
    assert!(target < n);
    assert_eq!(heuristic.len(), n);

    let distance = common::filled(exec, n, INF)?;
    let predecessor = common::filled(exec, n, u32::MAX)?;
    let open = common::filled(exec, n, 0u32)?;
    vector::scatter(
        exec,
        lazy::constant(0u32).take(1),
        common::indices(lazy::constant(source).take(1)),
        distance.slice_mut(..),
    )?;
    vector::scatter(
        exec,
        lazy::constant(1u32).take(1),
        common::indices(lazy::constant(source).take(1)),
        open.slice_mut(..),
    )?;

    let mut expanded = 0u32;
    let mut final_cost = None;
    loop {
        let open_vertices = common::materialize_exact(
            exec,
            vector::copy_where(
                exec,
                common::counting_u32(0, n as usize),
                common::stencil(open.slice(..)),
            )?,
        )?;
        if open_vertices.len() == 0 {
            break;
        }
        let open_distance = vector::gather(
            exec,
            distance.slice(..),
            common::indices(open_vertices.slice(..)),
        )?;
        let open_heuristic = vector::gather(
            exec,
            heuristic.slice(..),
            common::indices(open_vertices.slice(..)),
        )?;
        let scores = vector::map(
            exec,
            zip2(open_distance.slice(..), open_heuristic.slice(..)),
            AddScore,
        )?;
        let best = vector::min_element(
            exec,
            zip2(scores.slice(..), open_vertices.slice(..)),
            common::EdgePairLess,
        )?
        .read(exec)?
        .expect("the open set is non-empty");
        let current = exec.to_host(&open_vertices.slice(best..best + 1))?[0];
        let current_distance = exec.to_host(&distance.slice(current..current + 1))?[0];
        if current == target {
            final_cost = Some(current_distance);
            break;
        }
        vector::scatter(
            exec,
            lazy::constant(0u32).take(1),
            common::indices(lazy::constant(current).take(1)),
            open.slice_mut(..),
        )?;
        expanded += 1;

        let frontier = common::filled(exec, 1, current)?;
        let edges = common::expand_rows(exec, graph.graph(), frontier.slice(..))?;
        if edges.destinations().len() == 0 {
            continue;
        }
        let proposals = vector::map(
            exec,
            zip2(
                lazy::constant(current_distance).take(edges.destinations().len()),
                lazy::permute(graph.weights().slice(..), edges.edge_ids().slice(..)),
            ),
            AddDistance,
        )?;
        let changed = common::relax_min(
            exec,
            edges.destinations().slice(..),
            proposals.slice(..),
            INF,
            &distance,
        )?;
        if changed.len() != 0 {
            vector::scatter(
                exec,
                lazy::constant(current).take(changed.len()),
                common::indices(changed.slice(..)),
                predecessor.slice_mut(..),
            )?;
            vector::scatter(
                exec,
                lazy::constant(1u32).take(changed.len()),
                common::indices(changed.slice(..)),
                open.slice_mut(..),
            )?;
        }
    }

    let vertices = if final_cost.is_some() {
        let predecessors = exec.to_host(&predecessor)?;
        let mut reversed = vec![target];
        let mut current = target;
        while current != source {
            current = predecessors[current as usize];
            assert!(current != u32::MAX, "reached target has no predecessor");
            reversed.push(current);
            assert!(
                reversed.len() <= n as usize,
                "predecessor chain contains a cycle"
            );
        }
        reversed.reverse();
        exec.to_device(&reversed)
    } else {
        exec.alloc::<u32>(0)
    };
    Ok(Path {
        vertices,
        cost: final_cost,
        expanded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn returns_the_weighted_shortest_path() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 2, 4, 5, 5], vec![1, 2, 2, 3, 3]);
        let graph =
            DeviceWeightedCsr::<_, u32>::from_host_parts(&exec, &host, &[1, 4, 1, 5, 1]).unwrap();
        let heuristic = exec.to_device(&[3u32, 2, 1, 0]);
        let path = solve(&exec, &graph, 0, 3, &heuristic).unwrap();
        assert_eq!(path.cost(), Some(3));
        assert_eq!(exec.to_host(path.vertices()).unwrap(), vec![0, 1, 2, 3]);
    }
}
