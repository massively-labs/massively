default:
    @just --list --list-submodules

mod verification 'verification'

doc:
    cargo doc -p massively --no-deps
    python3 -m http.server --directory target/doc 3000

bench:
    cargo bench -p massively

performance:
    cargo bench -p massively --bench performance
    python3 scripts/render-performance.py

test-api:
    cargo doc -p massively --no-deps
    bash scripts/check-public-api.sh

test: test-api verification::proof
    cargo nextest run
    cargo test -p massively --doc
