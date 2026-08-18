# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in args.inputs:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if {"shape", "M", "marlin_us", "hybrid_us"} <= row.keys():
                groups[(row["shape"], row["routing"])].append(row)
    if not groups:
        raise SystemExit("no production benchmark rows found")

    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 4.5 * len(groups)))
    if len(groups) == 1:
        axes = [axes]
    for axis, ((shape, routing), rows) in zip(axes, sorted(groups.items())):
        rows.sort(key=lambda row: row["M"])
        x = [row["M"] for row in rows]
        axis.plot(x, [row["marlin_us"] for row in rows], "o-", label="W4A16 Marlin")
        axis.plot(x, [row["hybrid_us"] for row in rows], "o-", label="W4A8 hybrid")
        gain_axis = axis.twinx()
        gain_axis.plot(
            x,
            [(row["speedup"] - 1) * 100 for row in rows],
            "o--",
            color="tab:green",
            label="Hybrid gain",
        )
        gain_axis.axhline(0, color="0.4", linewidth=1)

        first = rows[0]
        factor = first["topk"] / first["active_experts"]
        top_axis = axis.secondary_xaxis(
            "top",
            functions=(
                lambda m, factor=factor: m * factor,
                lambda routed, factor=factor: routed / factor,
            ),
        )
        top_axis.set_xlabel("Mean synthetic routed rows/expert (not a histogram)")
        axis.set_title(
            f"{shape}: E={first['E']}, top-k={first['topk']}, "
            f"K={first['K']}, N={first['N']}, routing={routing}"
        )
        axis.set_xlabel("Input-token matrix dimension M")
        axis.set_ylabel("Median CUDA-graph latency (µs)")
        gain_axis.set_ylabel("Hybrid gain vs Marlin (%)")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left")
        gain_axis.legend(loc="upper right")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg")
    args.output.write_text(
        "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
