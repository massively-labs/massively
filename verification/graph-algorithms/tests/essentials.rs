use std::{
    cmp::Ordering,
    collections::{BTreeMap, VecDeque},
};

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use graph_algorithms::{
    CsrGraph, DeviceCsr, DeviceWeightedCsr, astar, graph_trend_filtering, graphsage, knn,
    label_propagation, maxflow, projection, rmat, salsa, scan_statistics, snn, topk,
    vertex_nomination, who_to_follow,
};
use massively::Executor;
use proptest::{
    prelude::*,
    test_runner::{Config, TestCaseResult, TestRunner},
};

const CASES: u32 = 8;
const INF: u32 = 1_000_000_000;

#[derive(Clone, Debug)]
struct Case {
    graph: CsrGraph,
    weights: Vec<u32>,
    features: Vec<f32>,
    source: u32,
    target: u32,
}

fn cases() -> impl Strategy<Value = Case> {
    (2usize..7)
        .prop_flat_map(|vertices| {
            let possible = vertices * (vertices - 1);
            (
                Just(vertices),
                prop::collection::vec(any::<bool>(), possible),
                prop::collection::vec(1u32..7, possible),
                prop::collection::vec(-6i16..7, vertices * 2),
                0usize..vertices,
            )
        })
        .prop_map(|(vertices, present, capacities, features, source)| {
            let mut offsets = vec![0u32];
            let mut neighbors = Vec::new();
            let mut weights = Vec::new();
            let mut edge = 0;
            for lhs in 0..vertices {
                for rhs in 0..vertices {
                    if lhs == rhs {
                        continue;
                    }
                    if present[edge] {
                        neighbors.push(rhs as u32);
                        weights.push(capacities[edge]);
                    }
                    edge += 1;
                }
                offsets.push(neighbors.len() as u32);
            }
            Case {
                graph: CsrGraph::new(offsets, neighbors),
                weights,
                features: features.into_iter().map(f32::from).collect(),
                source: source as u32,
                target: ((source + 1) % vertices) as u32,
            }
        })
}

fn run_cases(test: impl Fn(&Executor<WgpuRuntime>, Case) -> TestCaseResult) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    })
    .run(&cases(), |case| test(&exec, case))
    .unwrap();
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

fn degrees(graph: &CsrGraph) -> Vec<u32> {
    graph
        .offsets
        .windows(2)
        .map(|bounds| bounds[1] - bounds[0])
        .collect()
}

fn dijkstra(graph: &CsrGraph, weights: &[u32], sources: &[u32]) -> Vec<u32> {
    let n = graph.vertex_count();
    let mut distance = vec![INF; n];
    for &source in sources {
        distance[source as usize] = 0;
    }
    for _ in 0..n {
        let Some(vertex) = (0..n)
            .filter(|&vertex| distance[vertex] < INF)
            .min_by_key(|&vertex| distance[vertex])
        else {
            break;
        };
        let base = distance[vertex];
        let begin = graph.offsets[vertex] as usize;
        let end = graph.offsets[vertex + 1] as usize;
        for edge in begin..end {
            let destination = graph.neighbors[edge] as usize;
            distance[destination] = distance[destination].min(base.saturating_add(weights[edge]));
        }
        // Marking is unnecessary for positive weights; repeated scans still
        // converge within n Bellman-style passes on these tiny references.
    }
    // Finish as Bellman-Ford so the deliberately simple selection above need
    // not maintain a separate settled set.
    for _ in 0..n {
        for source in 0..n {
            for edge in graph.offsets[source] as usize..graph.offsets[source + 1] as usize {
                let destination = graph.neighbors[edge] as usize;
                distance[destination] =
                    distance[destination].min(distance[source].saturating_add(weights[edge]));
            }
        }
    }
    distance
}

fn topk_ref(graph: &CsrGraph, k: usize) -> (Vec<u32>, Vec<u32>) {
    let mut score = degrees(graph);
    for &destination in &graph.neighbors {
        score[destination as usize] += 1;
    }
    let mut vertices = (0..graph.vertex_count() as u32).collect::<Vec<_>>();
    vertices.sort_by(|&lhs, &rhs| {
        score[rhs as usize]
            .cmp(&score[lhs as usize])
            .then(lhs.cmp(&rhs))
    });
    vertices.truncate(k.min(vertices.len()));
    let scores = vertices
        .iter()
        .map(|&vertex| score[vertex as usize])
        .collect();
    (vertices, scores)
}

fn intersect_count(lhs: &[u32], rhs: &[u32]) -> u32 {
    let (mut left, mut right, mut count) = (0, 0, 0);
    while left < lhs.len() && right < rhs.len() {
        match lhs[left].cmp(&rhs[right]) {
            Ordering::Less => left += 1,
            Ordering::Greater => right += 1,
            Ordering::Equal => {
                count += 1;
                left += 1;
                right += 1;
            }
        }
    }
    count
}

fn scan_ref(graph: &CsrGraph) -> Vec<u32> {
    (0..graph.vertex_count())
        .map(|source| {
            graph.row(source).len() as u32
                + graph
                    .row(source)
                    .iter()
                    .map(|&destination| {
                        intersect_count(graph.row(source), graph.row(destination as usize))
                    })
                    .sum::<u32>()
        })
        .collect()
}

fn label_propagation_ref(graph: &CsrGraph, iterations: usize) -> Vec<u32> {
    let mut labels = (0..graph.vertex_count() as u32).collect::<Vec<_>>();
    for _ in 0..iterations {
        let mut next = labels.clone();
        for source in 0..graph.vertex_count() {
            if graph.row(source).is_empty() {
                continue;
            }
            let mut counts = BTreeMap::<u32, u32>::new();
            for &destination in graph.row(source) {
                *counts.entry(labels[destination as usize]).or_default() += 1;
            }
            next[source] = counts
                .into_iter()
                .max_by(|lhs, rhs| lhs.1.cmp(&rhs.1).then(rhs.0.cmp(&lhs.0)))
                .unwrap()
                .0;
        }
        if next == labels {
            break;
        }
        labels = next;
    }
    labels
}

fn knn_ref(
    graph: &CsrGraph,
    features: &[f32],
    feature_count: usize,
    k: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut offsets = vec![0u32];
    let mut destinations = Vec::new();
    let mut distances = Vec::new();
    for source in 0..graph.vertex_count() {
        let mut row = graph
            .row(source)
            .iter()
            .map(|&destination| {
                let distance = (0..feature_count)
                    .map(|feature| {
                        let difference = features[source * feature_count + feature]
                            - features[destination as usize * feature_count + feature];
                        difference * difference
                    })
                    .sum::<f32>();
                (destination, distance)
            })
            .collect::<Vec<_>>();
        row.sort_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap().then(lhs.0.cmp(&rhs.0)));
        row.truncate(k);
        for (destination, distance) in row {
            destinations.push(destination);
            distances.push(distance);
        }
        offsets.push(destinations.len() as u32);
    }
    (offsets, destinations, distances)
}

fn salsa_ref(graph: &CsrGraph, iterations: usize) -> (Vec<f32>, Vec<f32>) {
    let n = graph.vertex_count();
    let out_degree = degrees(graph);
    let mut in_degree = vec![0u32; n];
    for &destination in &graph.neighbors {
        in_degree[destination as usize] += 1;
    }
    let mut hubs = vec![1.0 / n as f32; n];
    let mut authorities = hubs.clone();
    if graph.neighbors.is_empty() {
        return (hubs, authorities);
    }
    for _ in 0..iterations {
        authorities.fill(0.0);
        for source in 0..n {
            for &destination in graph.row(source) {
                authorities[destination as usize] += hubs[source] / out_degree[source] as f32;
            }
        }
        let sum = authorities.iter().sum::<f32>();
        if sum != 0.0 {
            for value in &mut authorities {
                *value /= sum;
            }
        }
        hubs.fill(0.0);
        for source in 0..n {
            for &destination in graph.row(source) {
                hubs[source] +=
                    authorities[destination as usize] / in_degree[destination as usize] as f32;
            }
        }
        let sum = hubs.iter().sum::<f32>();
        if sum != 0.0 {
            for value in &mut hubs {
                *value /= sum;
            }
        }
    }
    (hubs, authorities)
}

fn projection_ref(graph: &CsrGraph) -> (CsrGraph, Vec<u32>) {
    let mut edges = BTreeMap::<(u32, u32), u32>::new();
    for source in 0..graph.vertex_count() {
        for &lhs in graph.row(source) {
            for &rhs in graph.row(source) {
                if lhs != rhs {
                    *edges.entry((lhs, rhs)).or_default() += 1;
                }
            }
        }
    }
    weighted_map_to_csr(graph.vertex_count(), edges)
}

fn weighted_map_to_csr(
    vertex_count: usize,
    edges: BTreeMap<(u32, u32), u32>,
) -> (CsrGraph, Vec<u32>) {
    let mut offsets = vec![0u32];
    let mut destinations = Vec::new();
    let mut weights = Vec::new();
    for source in 0..vertex_count as u32 {
        for (&(lhs, rhs), &weight) in edges.range((source, 0)..=(source, u32::MAX)) {
            debug_assert_eq!(lhs, source);
            destinations.push(rhs);
            weights.push(weight);
        }
        offsets.push(destinations.len() as u32);
    }
    (CsrGraph::new(offsets, destinations), weights)
}

#[test]
fn direct_compositions_match_independent_cpu_references() {
    run_cases(|exec, case| {
        let device = DeviceCsr::from_host(exec, &case.graph).unwrap();

        let expected = topk_ref(&case.graph, 3);
        let actual = topk::solve(exec, &device, 3).unwrap();
        prop_assert_eq!(exec.to_host(actual.vertices()).unwrap(), expected.0);
        prop_assert_eq!(exec.to_host(actual.scores()).unwrap(), expected.1);

        let expected = scan_ref(&case.graph);
        let actual = scan_statistics::solve(exec, &device).unwrap();
        prop_assert_eq!(exec.to_host(&actual).unwrap(), expected);

        let expected = label_propagation_ref(&case.graph, 3);
        let actual = label_propagation::solve(exec, &device, 3).unwrap();
        prop_assert_eq!(exec.to_host(&actual).unwrap(), expected);

        let features = exec.to_device(&case.features);
        let expected = knn_ref(&case.graph, &case.features, 2, 2);
        let actual = knn::solve(exec, &device, &features, 2, 2).unwrap();
        prop_assert_eq!(exec.to_host(&actual.offsets()).unwrap(), expected.0);
        prop_assert_eq!(exec.to_host(actual.destinations()).unwrap(), expected.1);
        assert_near(
            &exec.to_host(actual.distances()).unwrap(),
            &expected.2,
            1.0e-5,
        )?;

        let expected = projection_ref(&case.graph);
        let actual = projection::solve(exec, &device).unwrap();
        let actual_graph = CsrGraph::new(
            exec.to_host(&actual.graph().offsets()).unwrap(),
            exec.to_host(actual.graph().destinations()).unwrap(),
        );
        prop_assert_eq!(actual_graph, expected.0);
        prop_assert_eq!(exec.to_host(actual.weights()).unwrap(), expected.1);

        let expected = salsa_ref(&case.graph, 3);
        let actual = salsa::solve(exec, &device, 3).unwrap();
        assert_near(&exec.to_host(&actual.0).unwrap(), &expected.0, 2.0e-4)?;
        assert_near(&exec.to_host(&actual.1).unwrap(), &expected.1, 2.0e-4)?;

        let weighted =
            DeviceWeightedCsr::<_, u32>::from_host_parts(exec, &case.graph, &case.weights).unwrap();
        let seeds = [case.source, case.target];
        let expected = dijkstra(&case.graph, &case.weights, &seeds);
        let seeds = exec.to_device(&seeds);
        let actual = vertex_nomination::solve(exec, &weighted, seeds.slice(..)).unwrap();
        prop_assert_eq!(exec.to_host(&actual).unwrap(), expected);
        Ok(())
    });
}

fn maxflow_ref(graph: &CsrGraph, capacities: &[u32], source: usize, sink: usize) -> u32 {
    let n = graph.vertex_count();
    let mut residual = vec![vec![0u32; n]; n];
    for lhs in 0..n {
        for edge in graph.offsets[lhs] as usize..graph.offsets[lhs + 1] as usize {
            residual[lhs][graph.neighbors[edge] as usize] += capacities[edge];
        }
    }
    let mut value = 0u32;
    loop {
        let mut parent = vec![usize::MAX; n];
        parent[source] = source;
        let mut queue = VecDeque::from([source]);
        while let Some(lhs) = queue.pop_front() {
            for rhs in 0..n {
                if residual[lhs][rhs] != 0 && parent[rhs] == usize::MAX {
                    parent[rhs] = lhs;
                    queue.push_back(rhs);
                }
            }
        }
        if parent[sink] == usize::MAX {
            break;
        }
        let mut amount = u32::MAX;
        let mut vertex = sink;
        while vertex != source {
            amount = amount.min(residual[parent[vertex]][vertex]);
            vertex = parent[vertex];
        }
        vertex = sink;
        while vertex != source {
            let lhs = parent[vertex];
            residual[lhs][vertex] -= amount;
            residual[vertex][lhs] += amount;
            vertex = lhs;
        }
        value += amount;
    }
    value
}

fn graphsage_ref(graph: &CsrGraph, input: &[f32]) -> Vec<f32> {
    let self_weights = [1.0f32, -0.5, 0.25, 1.0];
    let neighbor_weights = [0.5f32, 0.25, -0.25, 0.75];
    let bias = [0.1f32, -0.2];
    let mut output = vec![0.0; graph.vertex_count() * 2];
    for vertex in 0..graph.vertex_count() {
        let mut mean = [0.0f32; 2];
        if !graph.row(vertex).is_empty() {
            for &destination in graph.row(vertex) {
                for feature in 0..2 {
                    mean[feature] += input[destination as usize * 2 + feature];
                }
            }
            for value in &mut mean {
                *value /= graph.row(vertex).len() as f32;
            }
        }
        for column in 0..2 {
            let mut value = bias[column];
            for feature in 0..2 {
                value += input[vertex * 2 + feature] * self_weights[feature * 2 + column]
                    + mean[feature] * neighbor_weights[feature * 2 + column];
            }
            output[vertex * 2 + column] = value.max(0.0);
        }
    }
    output
}

fn trend_filter_ref(
    graph: &CsrGraph,
    signal: &[f32],
    columns: usize,
    lambda: f32,
    iterations: usize,
) -> Vec<f32> {
    let n = graph.vertex_count();
    let mut incidence = degrees(graph);
    for &destination in &graph.neighbors {
        incidence[destination as usize] += 1;
    }
    let max_incidence = incidence.into_iter().max().unwrap().max(1);
    let step = 0.9 / (2.0 * max_incidence as f32).sqrt();
    let mut current = signal.to_vec();
    let mut extrapolated = signal.to_vec();
    let mut dual = vec![0.0f32; graph.neighbors.len() * columns];
    for _ in 0..iterations {
        for source in 0..n {
            for edge in graph.offsets[source] as usize..graph.offsets[source + 1] as usize {
                let destination = graph.neighbors[edge] as usize;
                for column in 0..columns {
                    let index = edge * columns + column;
                    dual[index] = (dual[index]
                        + step
                            * (extrapolated[source * columns + column]
                                - extrapolated[destination * columns + column]))
                        .clamp(-lambda, lambda);
                }
            }
        }
        let mut divergence = vec![0.0f32; signal.len()];
        for source in 0..n {
            for edge in graph.offsets[source] as usize..graph.offsets[source + 1] as usize {
                let destination = graph.neighbors[edge] as usize;
                for column in 0..columns {
                    let value = dual[edge * columns + column];
                    divergence[source * columns + column] += value;
                    divergence[destination * columns + column] -= value;
                }
            }
        }
        let next = current
            .iter()
            .zip(&divergence)
            .zip(signal)
            .map(|((&current, &divergence), &signal)| {
                (current - step * divergence + step * signal) / (1.0 + step)
            })
            .collect::<Vec<_>>();
        extrapolated = next
            .iter()
            .zip(&current)
            .map(|(&next, &current)| 2.0 * next - current)
            .collect();
        current = next;
    }
    current
}

#[test]
fn high_control_compositions_match_independent_cpu_references() {
    run_cases(|exec, case| {
        let device = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let weighted =
            DeviceWeightedCsr::<_, u32>::from_host_parts(exec, &case.graph, &case.weights).unwrap();

        let expected_distance = dijkstra(&case.graph, &case.weights, &[case.source]);
        let heuristic = exec.to_device(&vec![0u32; case.graph.vertex_count()]);
        let path = astar::solve(exec, &weighted, case.source, case.target, &heuristic).unwrap();
        let expected = expected_distance[case.target as usize];
        prop_assert_eq!(path.cost(), (expected < INF).then_some(expected));
        if path.is_reachable() {
            let vertices = exec.to_host(path.vertices()).unwrap();
            prop_assert_eq!(vertices.first().copied(), Some(case.source));
            prop_assert_eq!(vertices.last().copied(), Some(case.target));
        }

        let expected = maxflow_ref(
            &case.graph,
            &case.weights,
            case.source as usize,
            case.target as usize,
        );
        let actual = maxflow::solve(exec, &weighted, case.source, case.target).unwrap();
        prop_assert_eq!(actual.value(), expected);

        let self_weights = [1.0f32, -0.5, 0.25, 1.0];
        let neighbor_weights = [0.5f32, 0.25, -0.25, 0.75];
        let bias = [0.1f32, -0.2];
        let expected = graphsage_ref(&case.graph, &case.features);
        let input = exec.to_device(&case.features);
        let actual = graphsage::solve(
            exec,
            &device,
            &input,
            2,
            &exec.to_device(&self_weights),
            &exec.to_device(&neighbor_weights),
            &exec.to_device(&bias),
            2,
            false,
        )
        .unwrap();
        assert_near(&exec.to_host(&actual).unwrap(), &expected, 2.0e-4)?;

        let expected = trend_filter_ref(&case.graph, &case.features, 2, 1.25, 12);
        let actual = graph_trend_filtering::solve(exec, &device, &input, 2, 1.25, 12).unwrap();
        assert_near(&exec.to_host(&actual).unwrap(), &expected, 5.0e-4)?;
        Ok(())
    });
}

fn snn_ref(
    offsets: &[u32],
    destinations: &[u32],
    shared_threshold: u32,
    core_threshold: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = offsets.len() - 1;
    let mut rows = (0..n)
        .map(|vertex| {
            let mut row =
                destinations[offsets[vertex] as usize..offsets[vertex + 1] as usize].to_vec();
            row.sort_unstable();
            row
        })
        .collect::<Vec<_>>();
    let mut similarity = vec![Vec::<u32>::new(); n];
    for source in 0..n {
        for &destination in &rows[source] {
            if rows[destination as usize]
                .binary_search(&(source as u32))
                .is_ok()
                && intersect_count(&rows[source], &rows[destination as usize]) >= shared_threshold
            {
                similarity[source].push(destination);
            }
        }
    }
    let core = similarity
        .iter()
        .map(|row| u32::from(row.len() as u32 >= core_threshold))
        .collect::<Vec<_>>();
    let mut component = (0..n as u32).collect::<Vec<_>>();
    let mut seen = vec![false; n];
    for start in 0..n {
        if core[start] == 0 || seen[start] {
            continue;
        }
        let mut vertices = vec![start];
        seen[start] = true;
        let mut cursor = 0;
        while cursor < vertices.len() {
            let source = vertices[cursor];
            for &destination in &similarity[source] {
                let destination = destination as usize;
                if core[destination] != 0 && !seen[destination] {
                    seen[destination] = true;
                    vertices.push(destination);
                }
            }
            cursor += 1;
        }
        let label = *vertices.iter().min().unwrap() as u32;
        for vertex in vertices {
            component[vertex] = label;
        }
    }
    let labels = (0..n)
        .map(|source| {
            if core[source] != 0 {
                component[source]
            } else {
                similarity[source]
                    .iter()
                    .filter(|&&destination| core[destination as usize] != 0)
                    .map(|&destination| component[destination as usize])
                    .min()
                    .unwrap_or(u32::MAX)
            }
        })
        .collect();
    // Keep ownership explicit: rows is intentionally built independently.
    rows.clear();
    (labels, core)
}

fn personalized_page_rank(
    graph: &CsrGraph,
    source: usize,
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
        let base = damping * dangling / n as f32;
        let mut next = vec![base; n];
        next[source] += 1.0 - damping;
        for lhs in 0..n {
            if degree[lhs] != 0 {
                for &rhs in graph.row(lhs) {
                    next[rhs as usize] += damping * rank[lhs] / degree[lhs] as f32;
                }
            }
        }
        rank = next;
    }
    rank
}

fn who_to_follow_ref(
    graph: &CsrGraph,
    source: usize,
    circle_size: usize,
    count: usize,
) -> Vec<u32> {
    let ppr = personalized_page_rank(graph, source, 0.85, 5);
    let mut circle = (0..graph.vertex_count())
        .filter(|&vertex| vertex != source)
        .collect::<Vec<_>>();
    circle.sort_by(|&lhs, &rhs| ppr[rhs].partial_cmp(&ppr[lhs]).unwrap().then(lhs.cmp(&rhs)));
    circle.truncate(circle_size.min(circle.len()));
    let mut member = vec![false; graph.vertex_count()];
    for &vertex in &circle {
        member[vertex] = true;
    }
    let mut offsets = vec![0u32];
    let mut destinations = Vec::new();
    for lhs in 0..graph.vertex_count() {
        if member[lhs] {
            destinations.extend(
                graph
                    .row(lhs)
                    .iter()
                    .copied()
                    .filter(|&rhs| member[rhs as usize]),
            );
        }
        offsets.push(destinations.len() as u32);
    }
    let induced = CsrGraph::new(offsets, destinations);
    let authority = salsa_ref(&induced, 3).1;
    circle.sort_by(|&lhs, &rhs| {
        authority[rhs]
            .partial_cmp(&authority[lhs])
            .unwrap()
            .then_with(|| ppr[rhs].partial_cmp(&ppr[lhs]).unwrap())
            .then(lhs.cmp(&rhs))
    });
    circle.truncate(count.min(circle.len()));
    circle.into_iter().map(|vertex| vertex as u32).collect()
}

#[test]
fn composed_clustering_and_recommendations_match_cpu_references() {
    run_cases(|exec, case| {
        let device = DeviceCsr::from_host(exec, &case.graph).unwrap();
        let features = exec.to_device(&case.features);
        let nearest = knn::solve(exec, &device, &features, 2, 2).unwrap();
        let offsets = exec.to_host(&nearest.offsets()).unwrap();
        let destinations = exec.to_host(nearest.destinations()).unwrap();
        let expected = snn_ref(&offsets, &destinations, 1, 1);
        let actual = snn::solve(exec, &nearest, 1, 1).unwrap();
        prop_assert_eq!(exec.to_host(actual.labels()).unwrap(), expected.0);
        prop_assert_eq!(exec.to_host(actual.core()).unwrap(), expected.1);

        let circle_size = usize::min(3, case.graph.vertex_count() - 1);
        let count = usize::min(2, circle_size);
        let expected = who_to_follow_ref(&case.graph, case.source as usize, circle_size, count);
        let actual = who_to_follow::solve(
            exec,
            &device,
            case.source,
            circle_size as u32,
            count as u32,
            0.85,
            5,
            3,
        )
        .unwrap();
        prop_assert_eq!(exec.to_host(actual.vertices()).unwrap(), expected);
        Ok(())
    });
}

fn rmat_ref(scale: u32, edge_count: usize, choices: &[f32], probabilities: [f32; 4]) -> CsrGraph {
    let mut edges = BTreeMap::<(u32, u32), ()>::new();
    for edge in 0..edge_count {
        let (mut source, mut destination) = (0u32, 0u32);
        for level in 0..scale {
            let choice = choices[level as usize * edge_count + edge];
            let bit = 1 << (scale - level - 1);
            if choice < probabilities[0] {
            } else if choice < probabilities[0] + probabilities[1] {
                destination |= bit;
            } else if choice < probabilities[0] + probabilities[1] + probabilities[2] {
                source |= bit;
            } else {
                source |= bit;
                destination |= bit;
            }
        }
        if source != destination {
            edges.insert((source, destination), ());
            edges.insert((destination, source), ());
        }
    }
    let n = 1usize << scale;
    let weighted = edges.into_keys().map(|edge| (edge, 1)).collect();
    weighted_map_to_csr(n, weighted).0
}

#[test]
fn rmat_matches_cpu_for_supplied_choices() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    })
    .run(&prop::collection::vec(0u16..10_000, 24), |words| {
        let choices = words
            .into_iter()
            .map(|word| word as f32 / 10_000.0)
            .collect::<Vec<_>>();
        let expected = rmat_ref(3, 8, &choices, [0.25; 4]);
        let choices = exec.to_device(&choices);
        let actual =
            rmat::solve_with_choices(&exec, 3, 8, [0.25; 4], &choices, true, true).unwrap();
        let actual = CsrGraph::new(
            exec.to_host(&actual.offsets()).unwrap(),
            exec.to_host(actual.destinations()).unwrap(),
        );
        prop_assert_eq!(actual, expected);
        Ok(())
    })
    .unwrap();
}
