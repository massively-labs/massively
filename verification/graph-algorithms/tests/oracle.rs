use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use graph_algorithms::{self as graph, CsrGraph, DeviceCsr, DeviceWeightedCsr, WeightedCsr};
use massively::{Executor, MStorage, op::Identity, vector, zip2};
use proptest::{
    prelude::*,
    test_runner::{Config, TestCaseResult, TestRunner},
};

const CASES: u32 = 24;
const ITERATIONS: usize = 3;
const INF: u32 = 1_000_000_000;

#[derive(Clone, Debug)]
struct GraphCase {
    graph: CsrGraph,
    weights_u32: Vec<u32>,
    weights_f32: Vec<f32>,
    vector: Vec<f32>,
    coordinates: Vec<(f32, f32)>,
    known: Vec<bool>,
    source: u32,
}

fn graph_case() -> impl Strategy<Value = GraphCase> {
    (1usize..8)
        .prop_flat_map(|vertices| {
            let possible_edges = vertices * vertices.saturating_sub(1) / 2;
            (
                Just(vertices),
                prop::collection::vec(any::<bool>(), possible_edges),
                prop::collection::vec(1u32..10, possible_edges),
                prop::collection::vec(-8i16..9, vertices),
                prop::collection::vec((-8i16..9, -8i16..9), vertices),
                prop::collection::vec(any::<bool>(), vertices),
                0usize..vertices,
            )
        })
        .prop_map(
            |(vertices, present, edge_weights, vector, coordinates, known, source)| {
                let mut rows = vec![Vec::<(u32, u32)>::new(); vertices];
                let mut pair = 0;
                for lhs in 0..vertices {
                    for rhs in lhs + 1..vertices {
                        if present[pair] {
                            let weight = edge_weights[pair];
                            rows[lhs].push((rhs as u32, weight));
                            rows[rhs].push((lhs as u32, weight));
                        }
                        pair += 1;
                    }
                }

                let mut offsets = Vec::with_capacity(vertices + 1);
                let mut neighbors = Vec::new();
                let mut weights_u32 = Vec::new();
                offsets.push(0);
                for row in &mut rows {
                    row.sort_unstable_by_key(|&(destination, _)| destination);
                    for &(destination, weight) in row.iter() {
                        neighbors.push(destination);
                        weights_u32.push(weight);
                    }
                    offsets.push(neighbors.len() as u32);
                }

                GraphCase {
                    graph: CsrGraph::new(offsets, neighbors),
                    weights_f32: weights_u32.iter().map(|&weight| weight as f32).collect(),
                    weights_u32,
                    vector: vector.into_iter().map(f32::from).collect(),
                    coordinates: coordinates
                        .into_iter()
                        .map(|(x, y)| (f32::from(x), f32::from(y)))
                        .collect(),
                    known,
                    source: source as u32,
                }
            },
        )
}

fn assert_near(actual: &[f32], expected: &[f32], tolerance: f32) -> TestCaseResult {
    prop_assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        prop_assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual}, expected={expected}"
        );
    }
    Ok(())
}

fn assert_coordinates_near(
    actual: &[(f32, f32)],
    expected: &[(f32, f32)],
    tolerance: f32,
) -> TestCaseResult {
    prop_assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        prop_assert!(
            (actual.0 - expected.0).abs() <= tolerance
                && (actual.1 - expected.1).abs() <= tolerance,
            "index={index}, actual={actual:?}, expected={expected:?}"
        );
    }
    Ok(())
}

fn run_graph_cases(test: impl Fn(&Executor<WgpuRuntime>, GraphCase) -> TestCaseResult) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });

    runner.run(&graph_case(), |case| test(&exec, case)).unwrap();
}

#[test]
fn bfs_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::bfs(&case.graph, case.source);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::bfs::solve(exec, &device_graph, case.source).unwrap();
        let actual = exec.to_host(&output).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn connected_components_match_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::cc(&case.graph);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::cc::solve(exec, &device_graph).unwrap();
        let actual = exec.to_host(&output).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn random_walks_match_cpu_reference_for_supplied_choices() {
    run_graph_cases(|exec, case| {
        let choice_count = case.graph.vertex_count() * 2;
        let choices = (0..choice_count)
            .map(|index| {
                (index as u32)
                    .wrapping_mul(747_796_405)
                    .wrapping_add(case.source ^ 2_891_336_453)
            })
            .collect::<Vec<_>>();
        let expected = reference::rw(&case.graph, 3, 1, &choices);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let choices = exec.to_device(&choices);
        let output = graph::rw::solve_with_choices(exec, &device_graph, 3, 1, &choices).unwrap();
        let actual = exec.to_host(output.vertices()).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn sssp_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::sssp(&case.graph, &case.weights_u32, case.source);
        let device_graph =
            DeviceWeightedCsr::<_, u32>::from_host_parts(exec, &case.graph, &case.weights_u32)
                .unwrap();
        let output = graph::sssp::solve(exec, &device_graph, case.source).unwrap();
        let actual = exec.to_host(&output).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn spmv_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::spmv(&case.graph, &case.weights_f32, &case.vector);
        let matrix = WeightedCsr::new(case.graph, case.weights_f32);
        let matrix = DeviceWeightedCsr::<_, f32>::from_host(exec, &matrix).unwrap();
        let vector = exec.to_device(&case.vector);
        let output = graph::spmv::solve(exec, &matrix, &vector).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_near(&actual, &expected, 1e-5)
    });
}

#[test]
fn page_rank_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::page_rank(&case.graph, 0.85, ITERATIONS);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::pr::solve(exec, &device_graph, 0.85, ITERATIONS).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_near(&actual, &expected, 2e-4)
    });
}

#[test]
fn personalized_page_rank_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected =
            reference::personalized_page_rank(&case.graph, case.source, 0.85, ITERATIONS);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::ppr::solve(exec, &device_graph, case.source, 0.85, ITERATIONS).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_near(&actual, &expected, 2e-4)
    });
}

#[test]
fn pr_nibble_matches_cpu_sweep_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::pr_nibble(&case.graph, case.source, 0.85, ITERATIONS);
        let expected_conductance = reference::conductance(&case.graph, &expected);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output =
            graph::pr_nibble::solve(exec, &device_graph, case.source, 0.85, ITERATIONS).unwrap();
        let actual = exec.to_host(&output).unwrap();
        prop_assert!(!actual.is_empty());
        prop_assert!(
            actual
                .iter()
                .all(|&vertex| vertex < case.graph.vertex_count() as u32)
        );
        prop_assert_eq!(
            actual.iter().copied().collect::<BTreeSet<_>>().len(),
            actual.len()
        );
        let actual_conductance = reference::conductance(&case.graph, &actual);
        // Mathematically tied PPR scores can round differently across
        // reduction orders. Compare the independently computed objective
        // rather than requiring one arbitrary tied vertex ID.
        prop_assert!(actual_conductance <= expected_conductance + 1.0e-5);
        Ok(())
    });
}

#[test]
fn hits_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::hits(&case.graph, ITERATIONS);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::hits::solve(exec, &device_graph, ITERATIONS).unwrap();
        let actual_hubs = exec.to_host(&output.0).unwrap();
        let actual_authorities = exec.to_host(&output.1).unwrap();
        assert_near(&actual_hubs, &expected.0, 2e-4)?;
        assert_near(&actual_authorities, &expected.1, 2e-4)
    });
}

#[test]
fn geolocation_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::geo(&case.graph, &case.coordinates, &case.known, ITERATIONS);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let xs = exec.to_device(
            &case
                .coordinates
                .iter()
                .map(|coordinate| coordinate.0)
                .collect::<Vec<_>>(),
        );
        let ys = exec.to_device(
            &case
                .coordinates
                .iter()
                .map(|coordinate| coordinate.1)
                .collect::<Vec<_>>(),
        );
        let coordinates = vector::map(exec, zip2(xs.slice(..), ys.slice(..)), Identity).unwrap();
        let known = exec.to_device(
            &case
                .known
                .iter()
                .map(|&known| u32::from(known))
                .collect::<Vec<_>>(),
        );
        let output =
            graph::geo::solve(exec, &device_graph, &coordinates, &known, ITERATIONS).unwrap();
        let (xs, ys) = MStorage::into_columns(output);
        let xs = exec.to_host(&xs).unwrap();
        let ys = exec.to_host(&ys).unwrap();
        let actual = xs.into_iter().zip(ys).collect::<Vec<_>>();
        assert_coordinates_near(&actual, &expected, 2e-4)
    });
}

#[test]
fn spgemm_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::spgemm(&case.graph, &case.graph);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::spgemm::solve(exec, &device_graph, &device_graph).unwrap();
        let actual = CsrGraph::new(
            exec.to_host(&output.offsets()).unwrap(),
            exec.to_host(output.destinations()).unwrap(),
        );
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn triangle_count_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::triangle_count(&case.graph);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let actual = graph::tc::solve(exec, &device_graph).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn subgraph_matching_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let query = match case.source % 4 {
            0 => CsrGraph::new(vec![0, 0], vec![]),
            1 => CsrGraph::new(vec![0, 1, 2], vec![1, 0]),
            2 => CsrGraph::new(vec![0, 1, 3, 4], vec![1, 0, 2, 1]),
            _ => CsrGraph::new(vec![0, 2, 4, 6], vec![1, 2, 0, 2, 0, 1]),
        };
        let expected = reference::sm(&case.graph, &query);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::sm::solve(exec, &device_graph, &query).unwrap();
        let actual = exec.to_host(output.mappings()).unwrap();
        prop_assert_eq!(output.match_count() as usize, expected.len());
        prop_assert_eq!(actual, expected.into_iter().flatten().collect::<Vec<_>>());
        Ok(())
    });
}

#[test]
fn graph_coloring_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::color(&case.graph);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::color::solve(exec, &device_graph).unwrap();
        let actual = exec.to_host(&output).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn kcore_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::kcore(&case.graph);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::kcore::solve(exec, &device_graph).unwrap();
        let actual = exec.to_host(&output).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn forman_ricci_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::forman_ricci(&case.graph);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::forman_ricci::solve(exec, &device_graph).unwrap();
        let actual = exec.to_host(&output).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn betweenness_centrality_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::betweenness_centrality(&case.graph);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::bc::solve(exec, &device_graph).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_near(&actual, &expected, 2e-4)
    });
}

#[test]
fn minimum_spanning_forest_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::minimum_spanning_forest(&case.graph, &case.weights_f32);
        let matrix = WeightedCsr::new(case.graph, case.weights_f32);
        let matrix = DeviceWeightedCsr::from_host(exec, &matrix).unwrap();
        let actual = graph::mst::solve(exec, &matrix).unwrap();
        let (sources, _, weights) = MStorage::into_columns(actual);
        prop_assert_eq!(sources.len() as usize, expected.0);
        let actual_weight = exec.to_host(&weights).unwrap().iter().sum::<f32>();
        prop_assert!((actual_weight - expected.1).abs() <= 1e-5);
        Ok(())
    });
}

#[test]
fn louvain_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::louvain(&case.graph, 20, 10);
        let device_graph = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let output = graph::louvain::solve(exec, &device_graph, 20, 10).unwrap();
        let actual = exec.to_host(&output).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

mod reference {
    use super::*;

    pub fn bfs(graph: &CsrGraph, source: u32) -> Vec<u32> {
        let mut distance = vec![u32::MAX; graph.vertex_count()];
        let mut queue = std::collections::VecDeque::from([source as usize]);
        distance[source as usize] = 0;
        while let Some(vertex) = queue.pop_front() {
            for &destination in graph.row(vertex) {
                let destination = destination as usize;
                if distance[destination] == u32::MAX {
                    distance[destination] = distance[vertex] + 1;
                    queue.push_back(destination);
                }
            }
        }
        distance
    }

    pub fn cc(graph: &CsrGraph) -> Vec<u32> {
        let mut labels = (0..graph.vertex_count() as u32).collect::<Vec<_>>();
        for _ in 0..graph.vertex_count().saturating_sub(1) {
            let previous = labels.clone();
            for source in 0..graph.vertex_count() {
                for &destination in graph.row(source) {
                    labels[destination as usize] =
                        labels[destination as usize].min(previous[source]);
                }
            }
        }
        labels
    }

    pub fn rw(
        graph: &CsrGraph,
        walk_length: usize,
        walks_per_vertex: usize,
        choices: &[u32],
    ) -> Vec<u32> {
        let walker_count = graph.vertex_count() * walks_per_vertex;
        assert_eq!(choices.len(), walker_count * walk_length.saturating_sub(1));
        let mut paths = vec![u32::MAX; walker_count * walk_length];
        for walker in 0..walker_count {
            let mut current = (walker % graph.vertex_count()) as u32;
            for step in 0..walk_length {
                paths[walker * walk_length + step] = current;
                if step + 1 == walk_length || current == u32::MAX {
                    continue;
                }
                let row = graph.row(current as usize);
                current = if row.is_empty() {
                    u32::MAX
                } else {
                    let choice = choices[walker * (walk_length - 1) + step];
                    row[choice as usize % row.len()]
                };
            }
        }
        paths
    }

    pub fn sssp(graph: &CsrGraph, weights: &[u32], source: u32) -> Vec<u32> {
        let mut distance = vec![INF; graph.vertex_count()];
        distance[source as usize] = 0;
        for _ in 0..graph.vertex_count().saturating_sub(1) {
            let mut changed = false;
            for vertex in 0..graph.vertex_count() {
                for edge in graph.offsets[vertex] as usize..graph.offsets[vertex + 1] as usize {
                    let destination = graph.neighbors[edge] as usize;
                    let proposal = distance[vertex].saturating_add(weights[edge]).min(INF);
                    if proposal < distance[destination] {
                        distance[destination] = proposal;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        distance
    }

    pub fn spmv(graph: &CsrGraph, weights: &[f32], vector: &[f32]) -> Vec<f32> {
        (0..graph.vertex_count())
            .map(|vertex| {
                (graph.offsets[vertex] as usize..graph.offsets[vertex + 1] as usize)
                    .map(|edge| weights[edge] * vector[graph.neighbors[edge] as usize])
                    .sum()
            })
            .collect()
    }

    pub fn page_rank(graph: &CsrGraph, damping: f32, iterations: usize) -> Vec<f32> {
        let n = graph.vertex_count();
        let degree = degrees(graph);
        let mut rank = vec![1.0 / n as f32; n];
        for _ in 0..iterations {
            let dangling = (0..n)
                .filter(|&vertex| degree[vertex] == 0)
                .map(|vertex| rank[vertex])
                .sum::<f32>();
            let mut next = vec![(1.0 - damping + damping * dangling) / n as f32; n];
            for source in 0..n {
                if degree[source] != 0 {
                    let contribution = damping * rank[source] / degree[source] as f32;
                    for &destination in graph.row(source) {
                        next[destination as usize] += contribution;
                    }
                }
            }
            rank = next;
        }
        rank
    }

    pub fn personalized_page_rank(
        graph: &CsrGraph,
        root: u32,
        damping: f32,
        iterations: usize,
    ) -> Vec<f32> {
        let n = graph.vertex_count();
        let degree = degrees(graph);
        let mut rank = vec![1.0 / n as f32; n];
        for _ in 0..iterations {
            let dangling = (0..n)
                .filter(|&vertex| degree[vertex] == 0)
                .map(|vertex| rank[vertex])
                .sum::<f32>();
            let mut next = vec![damping * dangling / n as f32; n];
            next[root as usize] += 1.0 - damping;
            for source in 0..n {
                if degree[source] != 0 {
                    let contribution = damping * rank[source] / degree[source] as f32;
                    for &destination in graph.row(source) {
                        next[destination as usize] += contribution;
                    }
                }
            }
            rank = next;
        }
        rank
    }

    pub fn pr_nibble(graph: &CsrGraph, source: u32, damping: f32, iterations: usize) -> Vec<u32> {
        let degree = degrees(graph);
        if degree[source as usize] == 0 {
            return vec![source];
        }
        let rank = personalized_page_rank(graph, source, damping, iterations);
        let mut order = (0..graph.vertex_count() as u32).collect::<Vec<_>>();
        order.sort_by(|&lhs, &rhs| {
            let lhs_score = if degree[lhs as usize] == 0 {
                -1.0
            } else {
                rank[lhs as usize] / degree[lhs as usize] as f32
            };
            let rhs_score = if degree[rhs as usize] == 0 {
                -1.0
            } else {
                rank[rhs as usize] / degree[rhs as usize] as f32
            };
            rhs_score
                .partial_cmp(&lhs_score)
                .unwrap()
                .then_with(|| lhs.cmp(&rhs))
        });
        let mut position = vec![0usize; graph.vertex_count()];
        for (index, &vertex) in order.iter().enumerate() {
            position[vertex as usize] = index;
        }

        let mut cut = 0.0f32;
        let mut volume = 0.0f32;
        let total_volume = graph.neighbors.len() as f32;
        let mut best = 0usize;
        let mut best_conductance = f32::MAX;
        for (index, &vertex) in order.iter().enumerate() {
            let earlier = graph
                .row(vertex as usize)
                .iter()
                .filter(|&&neighbor| position[neighbor as usize] < index)
                .count() as f32;
            cut += degree[vertex as usize] as f32 - 2.0 * earlier;
            volume += degree[vertex as usize] as f32;
            let denominator = volume.min(total_volume - volume);
            let conductance = if denominator > 0.0 {
                cut / denominator
            } else {
                f32::MAX
            };
            if conductance < best_conductance {
                best = index;
                best_conductance = conductance;
            }
        }
        order.truncate(best + 1);
        order
    }

    pub fn conductance(graph: &CsrGraph, vertices: &[u32]) -> f32 {
        let set = vertices.iter().copied().collect::<BTreeSet<_>>();
        let volume = set
            .iter()
            .map(|&vertex| graph.row(vertex as usize).len())
            .sum::<usize>();
        let internal = set
            .iter()
            .map(|&vertex| {
                graph
                    .row(vertex as usize)
                    .iter()
                    .filter(|destination| set.contains(destination))
                    .count()
            })
            .sum::<usize>();
        let cut = volume - internal;
        let denominator = volume.min(graph.neighbors.len() - volume);
        if denominator == 0 {
            f32::MAX
        } else {
            cut as f32 / denominator as f32
        }
    }

    pub fn hits(graph: &CsrGraph, iterations: usize) -> (Vec<f32>, Vec<f32>) {
        let n = graph.vertex_count();
        let mut hubs = vec![1.0f32; n];
        let mut authorities = vec![1.0f32; n];
        for _ in 0..iterations {
            authorities.fill(0.0);
            for source in 0..n {
                for &destination in graph.row(source) {
                    authorities[destination as usize] += hubs[source];
                }
            }
            normalize(&mut authorities);

            hubs.fill(0.0);
            for source in 0..n {
                for &destination in graph.row(source) {
                    hubs[source] += authorities[destination as usize];
                }
            }
            normalize(&mut hubs);
        }
        (hubs, authorities)
    }

    pub fn geo(
        graph: &CsrGraph,
        initial: &[(f32, f32)],
        known: &[bool],
        iterations: usize,
    ) -> Vec<(f32, f32)> {
        let mut coordinates = initial.to_vec();
        for _ in 0..iterations {
            let previous = coordinates.clone();
            for vertex in 0..graph.vertex_count() {
                if known[vertex] || graph.row(vertex).is_empty() {
                    continue;
                }
                let (sum_x, sum_y) =
                    graph
                        .row(vertex)
                        .iter()
                        .fold((0.0, 0.0), |(sum_x, sum_y), &destination| {
                            let value = previous[destination as usize];
                            (sum_x + value.0, sum_y + value.1)
                        });
                coordinates[vertex] = (
                    sum_x / graph.row(vertex).len() as f32,
                    sum_y / graph.row(vertex).len() as f32,
                );
            }
        }
        coordinates
    }

    pub fn spgemm(lhs: &CsrGraph, rhs: &CsrGraph) -> CsrGraph {
        let mut offsets = Vec::with_capacity(lhs.vertex_count() + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for source in 0..lhs.vertex_count() {
            let mut row = BTreeSet::new();
            for &middle in lhs.row(source) {
                row.extend(rhs.row(middle as usize).iter().copied());
            }
            neighbors.extend(row);
            offsets.push(neighbors.len() as u32);
        }
        CsrGraph::new(offsets, neighbors)
    }

    pub fn triangle_count(graph: &CsrGraph) -> u32 {
        let mut triangles = 0;
        for lhs in 0..graph.vertex_count() {
            for rhs in lhs + 1..graph.vertex_count() {
                if graph.row(lhs).binary_search(&(rhs as u32)).is_err() {
                    continue;
                }
                for third in rhs + 1..graph.vertex_count() {
                    if graph.row(lhs).binary_search(&(third as u32)).is_ok()
                        && graph.row(rhs).binary_search(&(third as u32)).is_ok()
                    {
                        triangles += 1;
                    }
                }
            }
        }
        triangles
    }

    pub fn sm(data: &CsrGraph, query: &CsrGraph) -> Vec<Vec<u32>> {
        let n = data.vertex_count() as u32;
        let k = query.vertex_count() as u32;
        if k > n {
            return Vec::new();
        }
        let candidate_count = n.pow(k);
        let decode = |mut code: u32| {
            let mut mapping = Vec::with_capacity(k as usize);
            for _ in 0..k {
                mapping.push(code % n);
                code /= n;
            }
            mapping
        };

        (0..candidate_count)
            .map(decode)
            .filter(|mapping| mapping.iter().copied().collect::<BTreeSet<_>>().len() == k as usize)
            .filter(|mapping| {
                (0..query.vertex_count()).all(|source| {
                    query.row(source).iter().all(|&destination| {
                        data.row(mapping[source] as usize)
                            .binary_search(&mapping[destination as usize])
                            .is_ok()
                    })
                })
            })
            .collect()
    }

    pub fn color(graph: &CsrGraph) -> Vec<u32> {
        let degree = degrees(graph);
        let mut order: Vec<_> = (0..graph.vertex_count()).collect();
        order.sort_by_key(|&vertex| Reverse(degree[vertex]));
        let mut colors = vec![u32::MAX; graph.vertex_count()];
        for vertex in order {
            let mut used = vec![false; graph.vertex_count() + 1];
            for &destination in graph.row(vertex) {
                let color = colors[destination as usize];
                if color != u32::MAX {
                    used[color as usize] = true;
                }
            }
            colors[vertex] = used.iter().position(|&used| !used).unwrap() as u32;
        }
        colors
    }

    pub fn kcore(graph: &CsrGraph) -> Vec<u32> {
        let mut current_degree = degrees(graph);
        let mut removed = vec![false; graph.vertex_count()];
        let mut core = vec![0; graph.vertex_count()];
        let mut running_core = 0;
        for _ in 0..graph.vertex_count() {
            let vertex = (0..graph.vertex_count())
                .filter(|&vertex| !removed[vertex])
                .min_by_key(|&vertex| current_degree[vertex])
                .unwrap();
            running_core = running_core.max(current_degree[vertex]);
            core[vertex] = running_core;
            removed[vertex] = true;
            for &destination in graph.row(vertex) {
                let destination = destination as usize;
                if !removed[destination] {
                    current_degree[destination] = current_degree[destination].saturating_sub(1);
                }
            }
        }
        core
    }

    pub fn forman_ricci(graph: &CsrGraph) -> Vec<i32> {
        let degree = degrees(graph);
        let mut output = Vec::with_capacity(graph.neighbors.len());
        for source in 0..graph.vertex_count() {
            for &destination in graph.row(source) {
                output.push(4 - degree[source] as i32 - degree[destination as usize] as i32);
            }
        }
        output
    }

    pub fn betweenness_centrality(graph: &CsrGraph) -> Vec<f32> {
        let n = graph.vertex_count();
        let mut centrality = vec![0.0; n];
        for source in 0..n {
            let distance = bfs(graph, source as u32);
            let mut order: Vec<_> = (0..n)
                .filter(|&vertex| distance[vertex] != u32::MAX)
                .collect();
            order.sort_by_key(|&vertex| distance[vertex]);
            let mut paths = vec![0.0; n];
            paths[source] = 1.0;
            for &vertex in &order {
                for &destination in graph.row(vertex) {
                    let destination = destination as usize;
                    if distance[destination] == distance[vertex] + 1 {
                        paths[destination] += paths[vertex];
                    }
                }
            }
            let mut dependency = vec![0.0; n];
            for &vertex in order.iter().rev() {
                for &destination in graph.row(vertex) {
                    let destination = destination as usize;
                    if distance[destination] == distance[vertex] + 1 && paths[destination] != 0.0 {
                        dependency[vertex] +=
                            paths[vertex] / paths[destination] * (1.0 + dependency[destination]);
                    }
                }
                if vertex != source {
                    centrality[vertex] += dependency[vertex];
                }
            }
        }
        centrality
    }

    pub fn minimum_spanning_forest(graph: &CsrGraph, weights: &[f32]) -> (usize, f32) {
        let mut edges = Vec::new();
        for source in 0..graph.vertex_count() {
            for edge in graph.offsets[source] as usize..graph.offsets[source + 1] as usize {
                let destination = graph.neighbors[edge] as usize;
                if source <= destination {
                    edges.push((source, destination, weights[edge]));
                }
            }
        }
        edges.sort_by(|lhs, rhs| lhs.2.partial_cmp(&rhs.2).unwrap());
        let mut sets = DisjointSet::new(graph.vertex_count());
        let mut count = 0;
        let mut total = 0.0;
        for (lhs, rhs, weight) in edges {
            if sets.union(lhs, rhs) {
                count += 1;
                total += weight;
            }
        }
        (count, total)
    }

    pub fn louvain(graph: &CsrGraph, max_passes: usize, max_levels: usize) -> Vec<u32> {
        let mut rows = (0..graph.vertex_count())
            .map(|source| {
                graph
                    .row(source)
                    .iter()
                    .map(|&destination| (destination as usize, 1u32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut assignment = (0..graph.vertex_count()).collect::<Vec<_>>();

        for _ in 0..max_levels {
            let communities = louvain_local_move(&rows, max_passes);
            let mut unique = communities.iter().copied().collect::<Vec<_>>();
            unique.sort_unstable();
            unique.dedup();
            let names = unique
                .iter()
                .enumerate()
                .map(|(name, &community)| (community, name))
                .collect::<BTreeMap<_, _>>();
            let labels = communities
                .iter()
                .map(|community| names[community])
                .collect::<Vec<_>>();
            for community in &mut assignment {
                *community = labels[*community];
            }
            if unique.len() == rows.len() {
                break;
            }

            let mut contracted = BTreeMap::<(usize, usize), u32>::new();
            for (source, row) in rows.iter().enumerate() {
                for &(destination, weight) in row {
                    *contracted
                        .entry((labels[source], labels[destination]))
                        .or_default() += weight;
                }
            }
            rows = vec![Vec::new(); unique.len()];
            for ((source, destination), weight) in contracted {
                rows[source].push((destination, weight));
            }
        }

        assignment.into_iter().map(|value| value as u32).collect()
    }

    fn louvain_local_move(rows: &[Vec<(usize, u32)>], max_passes: usize) -> Vec<usize> {
        let strength = rows
            .iter()
            .map(|row| row.iter().map(|&(_, weight)| weight).sum::<u32>())
            .collect::<Vec<_>>();
        let m2 = strength.iter().sum::<u32>();
        let mut communities = (0..rows.len()).collect::<Vec<_>>();
        let mut totals = strength.clone();
        if m2 == 0 {
            return communities;
        }

        for _ in 0..max_passes {
            let mut moved = false;
            for vertex in 0..rows.len() {
                let current = communities[vertex];
                let vertex_strength = strength[vertex];
                let current_total = totals[current];
                totals[current] -= vertex_strength;

                let mut incident = BTreeMap::<usize, u32>::new();
                for &(destination, weight) in &rows[vertex] {
                    if destination != vertex {
                        *incident.entry(communities[destination]).or_default() += weight;
                    }
                }
                incident.entry(current).or_default();
                let score = |community: usize, weight: u32, totals: &[u32]| {
                    weight as f32 - vertex_strength as f32 * totals[community] as f32 / m2 as f32
                };
                let current_score = score(current, incident[&current], &totals);
                let mut best = current;
                let mut best_score = current_score;
                for (&community, &weight) in &incident {
                    let candidate = score(community, weight, &totals);
                    if candidate > best_score || (candidate == best_score && community < best) {
                        best = community;
                        best_score = candidate;
                    }
                }
                if best != current && best_score > current_score + 1.0e-6 {
                    totals[best] += vertex_strength;
                    communities[vertex] = best;
                    moved = true;
                } else {
                    totals[current] = current_total;
                }
            }
            if !moved {
                break;
            }
        }
        communities
    }

    fn degrees(graph: &CsrGraph) -> Vec<u32> {
        (0..graph.vertex_count())
            .map(|vertex| graph.row(vertex).len() as u32)
            .collect()
    }

    fn normalize(values: &mut [f32]) {
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm != 0.0 {
            for value in values {
                *value /= norm;
            }
        }
    }

    struct DisjointSet {
        parent: Vec<usize>,
        rank: Vec<u8>,
    }

    impl DisjointSet {
        fn new(len: usize) -> Self {
            Self {
                parent: (0..len).collect(),
                rank: vec![0; len],
            }
        }

        fn find(&mut self, value: usize) -> usize {
            if self.parent[value] != value {
                self.parent[value] = self.find(self.parent[value]);
            }
            self.parent[value]
        }

        fn union(&mut self, lhs: usize, rhs: usize) -> bool {
            let mut lhs = self.find(lhs);
            let mut rhs = self.find(rhs);
            if lhs == rhs {
                return false;
            }
            if self.rank[lhs] < self.rank[rhs] {
                std::mem::swap(&mut lhs, &mut rhs);
            }
            self.parent[rhs] = lhs;
            if self.rank[lhs] == self.rank[rhs] {
                self.rank[lhs] += 1;
            }
            true
        }
    }
}
