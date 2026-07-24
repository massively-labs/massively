use std::{collections::BTreeSet, time::Duration};

use criterion::Criterion;
use graph_algorithms::{CsrGraph, WeightedCsr};

pub const SINGLE_PASS_SIZES: &[usize] = &[256, 1_024, 4_096];
pub const ITERATIVE_SIZES: &[usize] = &[128, 512, 2_048];
pub const CONTROL_SIZES: &[usize] = &[16, 64];
pub const SM_SIZES: &[usize] = &[8, 16, 32];
pub const ITERATIONS: usize = 3;

pub struct Fixture {
    pub graph: CsrGraph,
    pub matrix: WeightedCsr,
    pub weights_u32: Vec<u32>,
    pub vector: Vec<f32>,
    pub coordinates: Vec<(f32, f32)>,
    pub known: Vec<bool>,
}

impl Fixture {
    pub fn new(vertices: usize) -> Self {
        let graph = generated_graph(vertices);
        let weights_u32 = (0..graph.neighbors.len())
            .map(|edge| (edge % 31 + 1) as u32)
            .collect::<Vec<_>>();
        let weights_f32 = weights_u32
            .iter()
            .map(|&weight| weight as f32)
            .collect::<Vec<_>>();
        let vector = (0..vertices)
            .map(|vertex| (vertex % 97) as f32 / 97.0)
            .collect();
        let coordinates = (0..vertices)
            .map(|vertex| ((vertex % 101) as f32, ((vertex * 37 + 11) % 103) as f32))
            .collect();
        let known = (0..vertices).map(|vertex| vertex % 5 == 0).collect();
        let matrix = WeightedCsr::new(graph.clone(), weights_f32);

        Self {
            graph,
            matrix,
            weights_u32,
            vector,
            coordinates,
            known,
        }
    }
}

fn generated_graph(vertices: usize) -> CsrGraph {
    assert!(vertices >= 2);
    let mut rows = vec![BTreeSet::new(); vertices];

    for source in 0..vertices {
        connect(&mut rows, source, (source + 1) % vertices);

        let mut state = (source as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for _ in 0..4 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mut target = (state % vertices as u64) as usize;
            if target == source {
                target = (target + 1) % vertices;
            }
            connect(&mut rows, source, target);
        }
    }

    let mut offsets = Vec::with_capacity(vertices + 1);
    let mut neighbors = Vec::new();
    offsets.push(0);
    for row in rows {
        neighbors.extend(row);
        offsets.push(neighbors.len() as u32);
    }
    CsrGraph::new(offsets, neighbors)
}

fn connect(rows: &mut [BTreeSet<u32>], lhs: usize, rhs: usize) {
    rows[lhs].insert(rhs as u32);
    rows[rhs].insert(lhs as u32);
}

pub fn criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250))
}
