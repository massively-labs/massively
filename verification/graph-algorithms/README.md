# graph-algorithms

This crate is practical coverage evidence for Massively's public vector,
lazy-expression, and segment primitives. It answers a developer-facing
question: can a small, graph-independent primitive set implement a broad,
recognizable set of real graph problems?

The comparison target is the newer Essentials abstraction in Gunrock's
published
[Graph Algorithms table](https://gunrock.github.io/gunrock/gunrock.wiki/Graph-Algorithms.html),
because it represents Gunrock's current research frontier. In this document,
that column is therefore labeled simply **Gunrock**; the older abstraction is
not part of the comparison. As of 2026-07-25, this crate implements all 30
applications in that table, including every row whose Essentials entry is
`❌`.

This is a coverage result: compositions of Massively primitives cover all 18
applications whose Essentials entry still reports no implementation. It is
not yet a claim that Massively is faster than Gunrock; that requires a controlled
benchmark of both libraries on the same graphs, GPU, output semantics, and
stopping criteria.

## Gunrock comparison

The **Gunrock** column reproduces the official table's Essentials status. `✅`
means this crate has a complete entry point, independent CPU verification, and
a device-resident benchmark.

| Application | File | Gunrock | Massively primitives |
| --- | --- | --- | --- |
| A* Search | [astar](src/astar.rs) | ❌ | ✅ |
| Betweenness Centrality | [bc](src/bc.rs) | v0.0.1 | ✅ |
| Breadth-First Search | [bfs](src/bfs.rs) | v0.0.1 | ✅ |
| Connected Components | [cc](src/cc.rs) | ❌🌟 | ✅ |
| Graph Coloring | [color](src/color.rs) | v0.0.1 | ✅ |
| Geolocation | [geo](src/geo.rs) | v0.0.1 | ✅ |
| Graph Trend Filtering | [graph_trend_filtering](src/graph_trend_filtering.rs) | ❌ | ✅ |
| Graph Projections | [projection](src/projection.rs) | ❌ | ✅ |
| GraphSAGE | [graphsage](src/graphsage.rs) | ❌ | ✅ |
| Hyperlink-Induced Topic Search | [hits](src/hits.rs) | v0.0.1 | ✅ |
| K-Nearest Neighbors | [knn](src/knn.rs) | ❌ | ✅ |
| K-Core Decomposition | [kcore](src/kcore.rs) | v0.0.1 | ✅ |
| Label Propagation | [label_propagation](src/label_propagation.rs) | ❌ | ✅ |
| Louvain Modularity | [louvain](src/louvain.rs) | ❌🌟 | ✅ |
| MaxFlow | [maxflow](src/maxflow.rs) | ❌ | ✅ |
| Minimum Spanning Tree | [mst](src/mst.rs) | v0.0.1 | ✅ |
| PageRank | [pr](src/pr.rs) | v0.0.1 | ✅ |
| Local Graph Clustering | [pr_nibble](src/pr_nibble.rs) | v0.0.1 | ✅ |
| RMAT Graph Generator | [rmat](src/rmat.rs) | ❌ | ✅ |
| Random Walk | [rw](src/rw.rs) | ❌🌟 | ✅ |
| SALSA | [salsa](src/salsa.rs) | ❌ | ✅ |
| Scan Statistics | [scan_statistics](src/scan_statistics.rs) | ❌ | ✅ |
| Shared Nearest Neighbors | [snn](src/snn.rs) | ❌ | ✅ |
| Subgraph Matching | [sm](src/sm.rs) | ❌🌟 | ✅ |
| Sparse-Matrix Vector Multiplication | [spmv](src/spmv.rs) | v0.0.1 | ✅ |
| Single Source Shortest Path | [sssp](src/sssp.rs) | v0.0.1 | ✅ |
| Top K | [topk](src/topk.rs) | ❌ | ✅ |
| Triangle Counting | [tc](src/tc.rs) | v0.0.1 | ✅ |
| Vertex Nomination | [vertex_nomination](src/vertex_nomination.rs) | ❌ | ✅ |
| Who To Follow | [who_to_follow](src/who_to_follow.rs) | ❌ | ✅ |

The crate also contains three complete applications not listed in that Gunrock
table: [Forman–Ricci curvature](src/forman_ricci.rs),
[personalized PageRank](src/ppr.rs), and [Boolean SpGEMM](src/spgemm.rs).

## Semantic boundaries

- `cc` returns the minimum vertex identifier in each connected component by
  sparse-frontier minimum-label relaxation.
- `louvain` performs deterministic standard modularity-gain local moves and repeated
  weighted community contraction, not label propagation under a Louvain name.
- `rw` implements batched uniform random walks, including multiple walks per
  vertex, deterministic seeded generation, explicit random-word injection for
  testing, and dead-end termination.
- `sm` performs exact, unlabelled, non-induced subgraph isomorphism and returns
  every ordered embedding. Its exhaustive candidate space is intended for
  small query graphs.
- `pr_nibble` computes personalized PageRank and then returns the
  minimum-conductance prefix of the degree-normalized sweep order.
- `astar` accepts a caller-supplied admissible integer heuristic and reopens
  improved vertices, including for inconsistent heuristics.
- `knn` ranks existing CSR neighbors using vertex-major, runtime-dimensional
  feature rows; it is adjacency-local rather than an implicit all-pairs graph
  builder.
- `graph_trend_filtering` solves quadratic-fidelity graph total variation with
  any runtime signal-column count by PDHG.
- `graphsage` is a runtime-dimensional mean-aggregation inference layer.
  Repeated calls form deeper networks without feature-arity specialization.
- `maxflow` coalesces parallel integral capacities and runs residual
  augmenting-path rounds with explicit reverse arcs.
- `projection` emits a sparse weighted one-mode projection rather than a dense
  `n²` matrix.
- `snn` consumes `knn` output, forms a mutual shared-neighbor graph, finds core
  components, and attaches border vertices.
- `who_to_follow` uses personalized PageRank for the circle of trust and SALSA
  authority scores for reranking.

The undirected algorithms expect symmetric CSR input. Adjacency rows used by
set operations must be sorted. These semantic boundaries are documented in the
corresponding modules and reproduced by their CPU references.

## Reproducing the evidence

Generated CSR property tests compare all 33 algorithms with independent CPU
implementations or independently checked mathematical invariants. Random walks
and R-MAT are compared from caller-supplied random words so graph semantics are
tested independently of RNG generation.

```sh
cargo nextest run -p graph-algorithms --test oracle
cargo nextest run -p graph-algorithms --test essentials
```

The benchmark suite measures device-resident algorithm entry points. CSR
topology, weights, vectors, and random-walk choices are prepared outside the
timed region where applicable. Each iteration includes algorithm execution,
pooled output allocation, and explicit synchronization, but no bulk result
download by the benchmark harness. Sequential-control algorithms may
materialize parent chains or return individual control scalars as part of
their implementation while graph state and outputs otherwise remain resident.

```sh
cargo bench -p graph-algorithms --bench algorithms
```

The independent CPU oracles test the claim that each Rust program solves its
named textbook problem. Implementing this broad suite is also empirical
evidence that the general vector and segment primitives are sufficient without
a graph-specific execution abstraction in `massively`.
