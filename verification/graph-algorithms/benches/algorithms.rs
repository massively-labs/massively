mod common;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use graph_algorithms::{
    CsrGraph, DeviceCsr, DeviceWeightedCsr, astar, bc, bfs, cc, color, forman_ricci, geo,
    graph_trend_filtering, graphsage, hits, kcore, knn, label_propagation, louvain, maxflow, mst,
    ppr, pr, pr_nibble, projection, rmat, rw, salsa, scan_statistics, sm, snn, spgemm, spmv, sssp,
    tc, topk, vertex_nomination, who_to_follow,
};
use massively::{Executor, op::Identity, vector, zip2};

fn bench_single_pass(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_device_resident_single_pass");
    for &vertices in common::SINGLE_PASS_SIZES {
        let fixture = common::Fixture::new(vertices);
        let matrix = DeviceWeightedCsr::<_, f32>::from_host(exec, &fixture.matrix).unwrap();
        let vector = exec.to_device(&fixture.vector);
        let features = exec.to_device(
            &fixture
                .coordinates
                .iter()
                .flat_map(|&(x, y)| [x, y])
                .collect::<Vec<_>>(),
        );
        exec.sync().unwrap();

        group.throughput(Throughput::Elements(matrix.graph().edge_count() as u64));
        group.bench_function(BenchmarkId::new("spmv", vertices), |b| {
            b.iter(|| {
                let output = spmv::solve(exec, &matrix, &vector).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("forman_ricci", vertices), |b| {
            b.iter(|| {
                let output = forman_ricci::solve(exec, matrix.graph()).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("triangle_count", vertices), |b| {
            b.iter(|| {
                let output = tc::solve(exec, matrix.graph()).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("scan_statistics", vertices), |b| {
            b.iter(|| {
                let output = scan_statistics::solve(exec, matrix.graph()).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("top_k", vertices), |b| {
            b.iter(|| {
                let output = topk::solve(exec, matrix.graph(), 32).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("knn", vertices), |b| {
            b.iter(|| {
                let output = knn::solve(exec, matrix.graph(), &features, 2, 4).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("graph_projection", vertices), |b| {
            b.iter(|| {
                let output = projection::solve(exec, matrix.graph()).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
    }
    group.finish();
}

fn bench_iterative(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_device_resident_iterative");
    for &vertices in common::ITERATIVE_SIZES {
        let fixture = common::Fixture::new(vertices);
        let graph = DeviceCsr::from_host(exec, &fixture.graph).unwrap();
        let weighted_graph =
            DeviceWeightedCsr::from_parts(graph.clone(), exec.to_device(&fixture.weights_u32))
                .unwrap();
        let latitude = exec.to_device(
            &fixture
                .coordinates
                .iter()
                .map(|coordinate| coordinate.0)
                .collect::<Vec<_>>(),
        );
        let longitude = exec.to_device(
            &fixture
                .coordinates
                .iter()
                .map(|coordinate| coordinate.1)
                .collect::<Vec<_>>(),
        );
        let coordinates = vector::map(
            exec,
            zip2(latitude.slice(..), longitude.slice(..)),
            Identity,
        )
        .unwrap();
        let known = exec.to_device(
            &fixture
                .known
                .iter()
                .map(|&known| u32::from(known))
                .collect::<Vec<_>>(),
        );
        let rw_choices = exec.to_device(
            &(0..vertices * 7)
                .map(|index| (index as u32).wrapping_mul(747_796_405))
                .collect::<Vec<_>>(),
        );
        let seeds = exec.to_device(&[0u32, vertices.saturating_sub(1) as u32]);
        let heuristic = exec.to_device(&vec![0u32; vertices]);
        let features = exec.to_device(
            &fixture
                .coordinates
                .iter()
                .flat_map(|&(x, y)| [x, y])
                .collect::<Vec<_>>(),
        );
        let identity = exec.to_device(&[1.0f32, 0.0, 0.0, 1.0]);
        let bias = exec.to_device(&[0.0f32, 0.0]);
        exec.sync().unwrap();

        group.throughput(Throughput::Elements(graph.edge_count() as u64));
        group.bench_function(BenchmarkId::new("bfs", vertices), |b| {
            b.iter(|| {
                let output = bfs::solve(exec, &graph, 0).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("cc", vertices), |b| {
            b.iter(|| {
                let output = cc::solve(exec, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("sssp", vertices), |b| {
            b.iter(|| {
                let output = sssp::solve(exec, &weighted_graph, 0).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("page_rank", vertices), |b| {
            b.iter(|| {
                let output = pr::solve(exec, &graph, 0.85, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("personalized_page_rank", vertices), |b| {
            b.iter(|| {
                let output = ppr::solve(exec, &graph, 0, 0.85, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("pr_nibble", vertices), |b| {
            b.iter(|| {
                let output = pr_nibble::solve(exec, &graph, 0, 0.85, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("rw", vertices), |b| {
            b.iter(|| {
                let output = rw::solve_with_choices(exec, &graph, 8, 1, &rw_choices).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("hits", vertices), |b| {
            b.iter(|| {
                let output = hits::solve(exec, &graph, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("geolocation", vertices), |b| {
            b.iter(|| {
                let output =
                    geo::solve(exec, &graph, &coordinates, &known, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("label_propagation", vertices), |b| {
            b.iter(|| {
                let output = label_propagation::solve(exec, &graph, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("salsa", vertices), |b| {
            b.iter(|| {
                let output = salsa::solve(exec, &graph, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("vertex_nomination", vertices), |b| {
            b.iter(|| {
                let output =
                    vertex_nomination::solve(exec, &weighted_graph, seeds.slice(..)).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("a_star", vertices), |b| {
            b.iter(|| {
                let output = astar::solve(
                    exec,
                    &weighted_graph,
                    0,
                    vertices.saturating_sub(1) as u32,
                    &heuristic,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("graph_trend_filtering", vertices), |b| {
            b.iter(|| {
                let output = graph_trend_filtering::solve(
                    exec,
                    &graph,
                    &features,
                    2,
                    1.0,
                    common::ITERATIONS,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("graphsage", vertices), |b| {
            b.iter(|| {
                let output = graphsage::solve(
                    exec, &graph, &features, 2, &identity, &identity, &bias, 2, true,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("who_to_follow", vertices), |b| {
            b.iter(|| {
                let output = who_to_follow::solve(
                    exec,
                    &graph,
                    0,
                    u32::min(64, vertices.saturating_sub(1) as u32),
                    u32::min(10, vertices.saturating_sub(1) as u32),
                    0.85,
                    common::ITERATIONS,
                    common::ITERATIONS,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
    }
    group.finish();
}

fn bench_control(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_device_resident_control");
    for &vertices in common::CONTROL_SIZES {
        let fixture = common::Fixture::new(vertices);
        let graph = DeviceCsr::from_host(exec, &fixture.graph).unwrap();
        let matrix = DeviceWeightedCsr::<_, f32>::from_host(exec, &fixture.matrix).unwrap();
        let capacities = DeviceWeightedCsr::<_, u32>::from_host_parts(
            exec,
            &fixture.graph,
            &fixture.weights_u32,
        )
        .unwrap();
        let features = exec.to_device(
            &fixture
                .coordinates
                .iter()
                .flat_map(|&(x, y)| [x, y])
                .collect::<Vec<_>>(),
        );
        let nearest = knn::solve(exec, &graph, &features, 2, 4).unwrap();
        exec.sync().unwrap();

        group.throughput(Throughput::Elements(graph.edge_count() as u64));
        group.bench_function(BenchmarkId::new("coloring", vertices), |b| {
            b.iter(|| {
                let output = color::solve(exec, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("k_core", vertices), |b| {
            b.iter(|| {
                let output = kcore::solve(exec, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("louvain", vertices), |b| {
            b.iter(|| {
                let output = louvain::solve(exec, &graph, 10, 10).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("minimum_spanning_forest", vertices), |b| {
            b.iter(|| {
                let output = mst::solve(exec, &matrix).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("betweenness_centrality", vertices), |b| {
            b.iter(|| {
                let output = bc::solve(exec, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("boolean_spgemm", vertices), |b| {
            b.iter(|| {
                let output = spgemm::solve(exec, &graph, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("max_flow", vertices), |b| {
            b.iter(|| {
                let output =
                    maxflow::solve(exec, &capacities, 0, vertices.saturating_sub(1) as u32)
                        .unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(
            BenchmarkId::new("shared_nearest_neighbors", vertices),
            |b| {
                b.iter(|| {
                    let output = snn::solve(exec, &nearest, 1, 1).unwrap();
                    exec.sync().unwrap();
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn bench_generation(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_device_resident_generation");
    for &scale in &[8u32, 10, 12] {
        let vertices = 1u32 << scale;
        group.throughput(Throughput::Elements((vertices * 8) as u64));
        group.bench_function(BenchmarkId::new("rmat", vertices), |b| {
            b.iter(|| {
                let output = rmat::solve(
                    exec,
                    scale,
                    vertices * 8,
                    [0.57, 0.19, 0.19, 0.05],
                    7,
                    true,
                    true,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
    }
    group.finish();
}

fn bench_subgraph_matching(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_device_resident_subgraph_matching");
    let query = CsrGraph::new(vec![0, 2, 4, 6], vec![1, 2, 0, 2, 0, 1]);
    for &vertices in common::SM_SIZES {
        let fixture = common::Fixture::new(vertices);
        let graph = DeviceCsr::from_host(exec, &fixture.graph).unwrap();
        exec.sync().unwrap();

        group.throughput(Throughput::Elements(graph.edge_count() as u64));
        group.bench_function(BenchmarkId::new("sm_triangle_query", vertices), |b| {
            b.iter(|| {
                let output = sm::solve(exec, &graph, &query).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
    }
    group.finish();
}

fn bench_algorithms(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    bench_single_pass(c, &exec);
    bench_iterative(c, &exec);
    bench_control(c, &exec);
    bench_subgraph_matching(c, &exec);
    bench_generation(c, &exec);
}

criterion_group! { name = benches; config = common::criterion(); targets = bench_algorithms }
criterion_main!(benches);
