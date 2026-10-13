# Fast RMSNorm on the GPU (Triton)

RMSNorm is a small step that runs many times inside every large language model (Llama, for example). This repo has a GPU kernel written in [Triton](https://triton-lang.org) that does the whole step in one pass, and a benchmark showing it is faster than the straightforward PyTorch version. (TESTED IN GOOGLE COLAB) 

<img width="1800" height="675" alt="results" src="https://github.com/user-attachments/assets/b7656007-e1c7-449f-91e5-ac8da6e16f98" />


## Why it's faster

GPUs spend most of their time moving data around, not doing math. The straightforward PyTorch version does RMSNorm in about six small steps, and every step saves its result to GPU memory so the next step can read it back. My kernel reads each row once, does all the math, and writes it once, so it moves much less data.

## Results

Tested on a Tesla T4 (Google Colab), fp16, 4096 rows. Time per call, lower is better:

| Hidden size | Straightforward PyTorch | `torch.compile` | **My kernel** | Speedup vs straightforward |
|---|---|---|---|---|
| 1024 | 0.370 ms | 0.077 ms | **0.074 ms** | 5.0x |
| 2048 | 0.597 ms | 0.146 ms | 0.148 ms | 4.0x |
| 4096 | 1.059 ms | 0.291 ms | 0.294 ms | 3.6x |
| 8192 | 2.096 ms | 0.632 ms | **0.589 ms** | 3.6x |

- It is **3.6-5x faster** than the straightforward PyTorch version.
- It is **about the same speed as `torch.compile`**, which fuses the steps automatically. The small differences are within normal run-to-run noise.
- In the right-hand plot, my kernel reaches about 71% of the T4's maximum memory bandwidth. That plot uses the same byte count for every method, so the bandwidth for the straightforward version is an estimate.

## Files

- `kernel.py`: the Triton kernel
- `test_kernel.py`: checks the output against a plain PyTorch version
- `bench.py`: the benchmark that makes `results.png`

## Limits

Forward pass only (no backward pass), fp16/fp32 only, tested on one GPU (a T4). Results can vary a little between runs and between PyTorch versions.

## Run it

Needs an NVIDIA GPU.

```bash
pip install -r requirements.txt
pytest -q
python bench.py --peak-gbps 320   # use your GPU's peak memory bandwidth
```

Tested with torch `TODO`, triton `TODO`.
