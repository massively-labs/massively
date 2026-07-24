//! Graph algorithms composed from Massively's vector and segment primitives.
//!
//! This crate is verification evidence that graph algorithms need no
//! graph-specific execution layer in the public library.

mod common;

pub use common::{CsrGraph, DeviceCsr, DeviceWeightedCsr, WeightedCsr};

pub mod astar;
pub mod bc;
pub mod bfs;
pub mod cc;
pub mod color;
pub mod forman_ricci;
pub mod geo;
pub mod graph_trend_filtering;
pub mod graphsage;
pub mod hits;
pub mod kcore;
pub mod knn;
pub mod label_propagation;
pub mod louvain;
pub mod maxflow;
pub mod mst;
pub mod ppr;
pub mod pr;
pub mod pr_nibble;
pub mod projection;
pub mod rmat;
pub mod rw;
pub mod salsa;
pub mod scan_statistics;
pub mod sm;
pub mod snn;
pub mod spgemm;
pub mod spmv;
pub mod sssp;
pub mod tc;
pub mod topk;
pub mod vertex_nomination;
pub mod who_to_follow;
