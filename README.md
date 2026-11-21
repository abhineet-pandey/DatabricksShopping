# Shopping Sessions: native Rust pipeline

A single Rust binary that implements `rules.md` end to end. It has no
dataframe engine and only two dependencies (`memmap2`, `libc`). It reads `/data`,
writes `/output/user_daily/` and `/output/campaign_daily/` as Parquet, and computes
everything from the mounted input on every run.

## Layout

| Path | What |
| --- | --- |
| `Dockerfile` | Scorer image: builds with `cargo build --release --locked` (target `x86-64-v3`), runs as UID 1000 on `debian:bookworm-slim` |
| `src/main.rs` | Pipeline: file chunking, parallel parse, identity resolution, per-person sessionizing and attribution, output |
| `src/csv.rs` | Strict CSV dialect scanner, 64 bytes per step (AVX2 when built for it, SSE2 otherwise) |
| `src/timeparse.rs` | F1 / F2 / F3 timestamp parsing and validation |
| `src/ids.rs` | Exact, order-preserving integer codes for IDs, with interning as the fallback |
| `src/parquet.rs` | Minimal Parquet writer for the two output tables (`day` as `DATE`) |
| `src/alloc.rs` | Allocator that requests transparent huge pages for large blocks |
| `src/arena.rs` | Per-thread bump arena and block lists for the bucket scatters, so buckets get huge pages |
| `reference/reference.py` | Slow, independent Python reference used for differential testing. Shares no code with the binary |
| `fixtures/` | The 36 official edge-case fixtures |
| `scripts/` | Fixture runner, scorer-flags `docker_run.sh`, and output comparer (from the starter kit) |
| `.github/workflows/ci.yml` | Builds the image and runs all fixtures in it on every push |

## How it works

1. **Chunk** every `events/`, `clicks/` and `links/` CSV into line-aligned pieces (memory-mapped) and parse them in parallel on `NCPU` threads.
2. **Identity:** links are joined with union-find. Each person is named by their byte-wise smallest ID.
3. **Events:** the copy with the smallest `ingest_ts` is kept per `event_id`, then validated (type, product, price, window) and late events are dropped.
4. **Per person**, all of their IDs together: events are sorted and split into sessions at gaps over 30 minutes. Each session is credited to the latest qualifying click (ties go to the smallest `click_id`) or to `organic`, and gets the distinct campaigns that assist it.
5. **Aggregate** into `user_daily` and `campaign_daily` and write Parquet.

Environment: `DATA_DIR`, `OUTPUT_DIR`, `SCRATCH_DIR` (defaults `/data`, `/output`, `/scratch`) and `NCPU` (thread count). For local runs the binary also takes `DATA_DIR OUTPUT_DIR` as two positional arguments. Set `SS_TIMING=1` for phase timings on stderr.

## Testing

```sh
# Build the scorer image and run all 36 fixtures in it with scorer flags
docker buildx build --platform linux/amd64 --load -t shopping-sessions:latest .
pip install polars==2.1.0
python3 scripts/run_fixtures.py fixtures -- sh scripts/docker_run.sh

# Without Docker: run the fixtures against a local build
cargo build --release
python3 scripts/run_fixtures.py fixtures -- target/release/shopping-sessions

# Compare against the published dev answer, or against the reference on any generated dataset
target/release/shopping-sessions dev out
python3 scripts/compare.py out dev-answer
python3 reference/reference.py DATA_DIR ref_out   # writes CSVs in the fixtures' expected format
```

Results so far: 36/36 fixtures, an exact match with the dev answer (117,038 `user_daily` and 27,400 `campaign_daily` rows), and an exact match with `reference.py` on every random generator seed tried. Measured on 8 threads (Linux, warm cache, a 2010-era Xeon): about 2.3 s for 20M events, peaking at about 3.4 GB RSS.
