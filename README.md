# FABO

This repository contains the algorithm release for **FABO (Flow-Aware
Breakpoint Optimization)**, a timing-driven rectilinear routing-tree algorithm
developed in:

> **FABO: Agent-Guided Discovery of Joint Breakpoint Optimization for
> Timing-Driven Routing Trees**

FABO starts from SALT-compatible shallow-light routing trees and jointly
optimizes where groups of root-to-sink flows stop sharing their common support
wires. 


## Repository layout

```text
src/                    FABO, SALT baseline, evaluator, and FLUTE support code
configs/                runtime/quality operating points
scripts/run_fabo.sh     build and run a `.nets` file
scripts/run_fabo_task_queue.py
                        parallel worker queue for full benchmark runs
examples/smoke.nets     small synthetic input for a quick check
POST9.dat, POWV9.dat    FLUTE lookup tables
```

## Requirements

- Linux
- CMake 3.5.1 or newer
- A C++14 compiler
- Python 3 for the optional parallel task queue

## Build

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target run_fabo eval_fabo_sweep_worker -j 8
```

The FLUTE lookup tables are loaded from the current working directory. Run the
binaries from the repository root, or place `POST9.dat` and `POWV9.dat` in the
chosen working directory.

## Quick start

```bash
scripts/run_fabo.sh examples/smoke.nets out/smoke t3
```

The command writes:

- `out/smoke.csv`: one FABO result per net;
- `out/smoke_cands.csv`: FLUTE, SALT, and FABO candidate metrics.

Use an ICCAD-style `.nets` file for a real run:

```bash
scripts/run_fabo.sh /path/to/design.nets out/design t3
```

An optional final argument limits the number of input nets:

```bash
scripts/run_fabo.sh /path/to/design.nets out/debug t3 100
```

## Operating points

All configurations run the same FABO algorithm and change only its
runtime/quality budget:

| Tier | Configuration | Intended use |
|---|---|---|
| `t3` | `configs/tier_t3_fast.env` | Main paper configuration and fast default |
| `t7` | `configs/tier_t7_yx.env` | Expanded Y/X support search |
| `t8` | `configs/tier_t8_anchor7.env` | Expanded anchor search |
| `t14` | `configs/tier_t14_quality.env` | Higher-quality operating point |

The implementation name `FAR_SALT_FULL` in the source and candidate CSV denotes
FABO. `SALT_R3` is the SALT baseline evaluated by the same checker.

## Full benchmark queue

For a precomputed task TSV, first load one operating point and then launch the
persistent worker queue:

```bash
set -a
source configs/tier_t3_fast.env
set +a

python3 scripts/run_fabo_task_queue.py \
  --tasks /path/to/tasks.tsv \
  --out out/fabo_full.csv \
  --workers 64
```

Each task line is tab-separated:

```text
task_id  epsilon_index  epsilon  design  /absolute/path/to/design.nets  byte_offset
```

Nets are independent, so the queue can scale across available CPU cores.

## Correctness checks

The evaluator checks every reported tree before its result is used:

- the output is a valid tree;
- source and sink coordinates are unchanged;
- every terminal is covered exactly once;
- only Steiner points may move;
- every sink satisfies `d_T(r,t) <= (1 + epsilon) * d_M(r,t)`.

## Third-party software

FABO code is released under the [BSD 3-Clause License](LICENSE).
FABO builds on the public SALT routing-tree implementation and embeds FLUTE;
third-party components retain their own terms. FLUTE is distributed under its
included Attribution Assurance License; see
[`src/salt/base/flute/license.txt`](src/salt/base/flute/license.txt). The FLUTE
lookup tables are `POST9.dat` and `POWV9.dat`. Better Enums is included under
its BSD 2-Clause license notice in `src/salt/utils/enum.h`.

Review the included third-party terms before redistribution or commercial use.

## Citation

Citation metadata will be added when the paper is publicly available.
