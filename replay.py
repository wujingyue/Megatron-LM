"""Replay recorded CUDA allocation requests using only PyTorch (no model or C++).

Sizes, allocation streams, pool scopes, free requests, and cache flushes are
recorded. A helper stream approximates capture-time deferred frees; the original
dependency graph and eager free completion timing are not replayed. See README
for measured differences.
"""
import argparse
import gzip
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path,
                        default=Path(__file__).with_name("trace.json.gz"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--scale", type=int, default=1,
                        help="Divide requested sizes for smaller GPUs; changes fragmentation.")
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be positive")
    torch.cuda.set_device(args.device)
    with gzip.open(args.trace, "rt") as f:
        trace = json.load(f)
    print(json.dumps({"trace": args.trace.name, "torch": torch.__version__,
                      "allocation_streams": len(trace["streams"]),
                      "allocations": len(trace["allocations"]),
                      "completion_streams": 1}), flush=True)

    # PyTorch cycles through 32 streams per priority. Avoid handle aliasing.
    if len(trace["streams"]) > 64:
        raise ValueError("This replay supports at most 64 allocation streams")
    streams = [torch.cuda.default_stream()]
    streams += [torch.cuda.Stream(priority=0 if i <= 32 else -1)
                for i in range(1, len(trace["streams"]))]
    completion_stream = torch.cuda.Stream(priority=-1)
    assert len({s.cuda_stream for s in streams + [completion_stream]}) == len(streams) + 1
    pool = torch.cuda.MemPool()
    scope = None
    live = {}

    def report(phase):
        stats = torch.cuda.memory_stats()
        row = {"phase": phase}
        for label in ("allocated", "active", "reserved"):
            row[label] = stats[f"{label}_bytes.all.current"]
            row[f"peak_{label}"] = stats[f"{label}_bytes.all.peak"]
        row["peak_gap"] = row["peak_reserved"] - row["peak_allocated"]
        if phase == "final" and args.scale == 1 and "expected" in trace:
            row["expected"] = trace["expected"]
            row["difference"] = {key: row[key] - value
                                 for key, value in trace["expected"].items()}
        print(json.dumps(row), flush=True)

    try:
        for index, action, *payload in trace["events"]:
            if action == "a":
                allocation_id = payload[0]
                size, stream, pool_id, _, requested, completed = trace["allocations"][allocation_id]
                assert bool(pool_id) == (scope is not None), index
                with torch.cuda.stream(streams[stream]):
                    live[allocation_id] = torch.empty(
                        max(1, (size + args.scale - 1) // args.scale),
                        dtype=torch.uint8, device="cuda")
                # Approximate observed deferred frees, not the original stream graph.
                if scope is not None and requested is not None and completed != requested + 1:
                    live[allocation_id].record_stream(completion_stream)
            elif action == "f":
                del live[payload[0]]  # Free at the recorded request, not completion.
            elif action == "c":
                pass  # Recorded completion; eager timing is not enforced here.
            elif action == "begin":
                assert scope is None
                scope = torch.cuda.use_mem_pool(pool)
                scope.__enter__()
            elif action == "end":
                scope.__exit__(None, None, None)
                scope = None
            elif action == "cache":
                torch.cuda.empty_cache()
            elif action == "mark":
                report(payload[0])
            else:
                raise ValueError(f"Unknown event: {action}")
        report("final")
    finally:
        if scope is not None:
            scope.__exit__(None, None, None)
        live.clear()
        torch.cuda.synchronize()
        del pool
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
