# benchmark: my fused kernel vs a few other ways of doing rmsnorm
# run with: python bench.py --peak-gbps 320   (320 = a T4's max memory bandwidth)
# prints a table and saves results.png

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton

from kernel import rmsnorm

EPS = 1e-6


# the formula typed out in pytorch. in eager mode every op here is its own gpu
# kernel, so lots of trips to memory. this is the slow baseline
def naive(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


naive_compiled = torch.compile(naive)  # same function, but pytorch fuses it for us

impls = {
    "naive eager": naive,
    "torch rms_norm": lambda x, w: F.rms_norm(x, (x.shape[-1],), w, EPS),
    "torch.compile": naive_compiled,
    "fused triton (ours)": lambda x, w: rmsnorm(x, w, EPS),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--peak-gbps", type=float, required=True,
                        help="your gpu's max memory bandwidth in GB/s")
    args = parser.parse_args()

    dtype = torch.float16
    sizes = [1024, 2048, 4096, 8192]

    latencies = {name: [] for name in impls}
    bandwidths = {name: [] for name in impls}

    for n in sizes:
        x = torch.randn(args.rows, n, device="cuda", dtype=dtype)
        w = torch.randn(n, device="cuda", dtype=dtype)

        # minimum bytes any version has to move: read x, write y, read w.
        # same number for every method, so for the unfused ones it's "effective" bandwidth
        nbytes = (2 * args.rows * n + n) * x.element_size()

        for name, fn in impls.items():
            fn(x, w)  # warmup so compiling doesn't get timed
            ms = triton.testing.do_bench(lambda: fn(x, w), warmup=25, rep=100)
            gbps = nbytes / (ms * 1e-3) / 1e9
            latencies[name].append(ms)
            bandwidths[name].append(gbps)
            print(f"N={n:5d} {name:22s} {ms:8.4f} ms {gbps:8.1f} GB/s")

    fig, (ax_lat, ax_bw) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name in impls:
        ax_lat.plot(sizes, latencies[name], marker="o", label=name)
        ax_bw.plot(sizes, bandwidths[name], marker="o", label=name)

    ax_bw.axhline(args.peak_gbps, color="gray", ls="--", label="peak HBM bandwidth")

    ax_lat.set_title("Latency")
    ax_lat.set_ylabel("ms (lower is better)")
    ax_bw.set_title("Achieved bandwidth")
    ax_bw.set_ylabel("GB/s (higher is better)")

    for ax in (ax_lat, ax_bw):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        ax.set_xlabel("hidden size")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"RMSNorm fp16, {args.rows} rows, {torch.cuda.get_device_name()}")
    fig.tight_layout()
    fig.savefig("results.png", dpi=150)


if __name__ == "__main__":
    main()
