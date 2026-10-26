# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--quantization xdna_int8`: W8A8 linears for the CPU platform, large batches split with the NPU.

Weights load as bf16 from an ordinary checkpoint and are quantized per output
channel when loading finishes; activations are quantized per token on the fly.
Each linear then runs as

  * few rows (decode, short prompts): oneDNN's int8 GEMM on the CPU;
  * many rows (prefill): the rows are split, a multiple of 512 of them to the
    NPU (`vllm.kernels.xdna.gemm_i8.NpuGemm`) and the rest to the CPU, both at once.
    The split follows each layer's measured rates so that both finish together.

Embedding tables are quantized as they load (int8, per row; the per-layer
table in groups of 256), so no bf16 copy of them is ever held, and the LM head
runs from the int8 table it is tied to.

Env: XDNA_VLLM_NPU=0 keeps every GEMM on the CPU; XDNA_VLLM_NPU_LAYERS is a
regex over layer names choosing which linears get an NPU copy (each costs its
int8 size again in host memory); XDNA_VLLM_NPU_MIN_ROWS is the fewest rows
worth splitting.
"""
import os
import queue
import re
import sys
import threading
import time

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.config import get_current_vllm_config_or_none
from vllm.kernels.xdna import gemm_i8 as npu
from vllm.kernels.xdna import runtime as npu_runtime
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs


logger = init_logger(__name__)
_QUANT = torch.ops._C.dynamic_scaled_int8_quant
_ONEDNN_MM = torch.ops._C.onednn_scaled_mm

USE_NPU = os.environ.get("XDNA_VLLM_NPU", "1") == "1"
# Splits use the NPU while this holds; set_npu_enabled flips it at run time
# (an A/B of the same process, weights and all).
NPU_ENABLED = [True]
FIXED_SHARE = [None]  # when set, the NPU's share of every split (a fraction), tuner bypassed


def set_npu_enabled(on: bool, fixed_share: float | None = None) -> None:
    NPU_ENABLED[0] = bool(on)
    FIXED_SHARE[0] = fixed_share
NPU_MIN_ROWS = int(os.environ.get("XDNA_VLLM_NPU_MIN_ROWS", "1024"))  # fewer rows: CPU only
# The linears worth an NPU copy (each costs its int8 size again in host
# memory): all but the per-layer-input ones, whose N or K is 256.
NPU_LAYERS = re.compile(os.environ.get(
    "XDNA_VLLM_NPU_LAYERS", r"qkv_proj|o_proj|gate_up_proj|down_proj|per_layer_model_projection"))
_SKIP = ("vision", "audio", "embed_vision", "embed_audio")  # towers stay bf16

TRACE = os.environ.get("XDNA_VLLM_TRACE", "0") == "1"  # one stderr line per call of >= 256 rows
# Rows per NPU launch. Each grid is its own compiled kernel and so its own NPU
# hardware context: launching both 512- and 1024-row grids doubles a model's
# ~9 contexts past what the backend keeps, and every evicted one costs ~35 ms
# to reopen (measured: 3-11 reopens a prefill). Bigger launches measured no
# faster anyway.
LAUNCH_ROWS = int(os.environ.get("XDNA_VLLM_NPU_LAUNCH_ROWS", "512"))
_LAYERS = []  # Int8Linear by key; the custom op takes the key
_TUNERS = {}  # (N, K, M // 512) -> _SplitTuner
# Row counts to time the NPU split for at load; by default every multiple of
# 512 from NPU_MIN_ROWS up to the scheduler's chunk (max_num_batched_tokens),
# since the tuner keeps one split per M // 512 and a prefill's last chunk, or a
# batch of prompts, can land in any of them.
_CALIBRATE_ENV = os.environ.get("XDNA_VLLM_CALIBRATE_ROWS")
STATS = {"npu_rows": 0, "cpu_rows": 0, "calls": 0, "t_cpu_share": 0.0, "t_wait": 0.0, "t_npu": 0.0}


def _quantize(w, group=None, rows=8192):
    """bf16 [N, K] -> int8 [N, K], float32 scales [N] (per row) or [N, K // group].

    A few thousand rows at a time: a float copy of a whole embedding table
    would be several GB.
    """
    N, K = w.shape
    q = torch.empty((N, K), dtype=torch.int8)
    s = torch.empty((N, K // group) if group else (N,), dtype=torch.float32)
    for r0 in range(0, N, rows):
        x = w[r0: r0 + rows].float()
        if group:
            x = x.view(x.shape[0], -1, group)
            sc = x.abs().amax(dim=2).clamp(min=1e-8) / 127
            q[r0: r0 + rows] = torch.round(x / sc[..., None]).clamp(-127, 127).view(-1, K)
        else:
            sc = x.abs().amax(dim=1).clamp(min=1e-8) / 127
            q[r0: r0 + rows] = torch.round(x / sc[:, None]).clamp(-127, 127)
        s[r0: r0 + rows] = sc
    return q, s


def _calibrate_rows():
    if _CALIBRATE_ENV is not None:
        return [int(m) for m in _CALIBRATE_ENV.split(",") if m]
    cfg = get_current_vllm_config_or_none()
    chunk = cfg.scheduler_config.max_num_batched_tokens if cfg is not None else 4096
    lo = -(-NPU_MIN_ROWS // npu.BLOCK_M) * npu.BLOCK_M
    return list(range(lo, chunk + 1, npu.BLOCK_M))


def _siblings(cpu):
    try:
        text = open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read()
    except OSError:
        return {cpu}
    out = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _npu_thread_cpu():
    """A CPU for the launching thread: XDNA_VLLM_NPU_CPU, else one whose core no OMP thread uses.

    The thread is created by the OMP master, so it would inherit its CPU and
    take turns with oneDNN's thread 0 there for every launch and sync.
    """
    if os.environ.get("XDNA_VLLM_NPU_CPU"):
        return int(os.environ["XDNA_VLLM_NPU_CPU"])
    omp = {int(c) for c in re.findall(r"\d+", os.environ.get("OMP_PLACES", ""))}
    if not omp:
        return None
    free = [c for c in range(os.cpu_count() or 1) if not (_siblings(c) & omp)]
    return free[-1] if free else None


class _Job:
    """One split call's NPU share: chunks launched in order by the launch thread.

    The caller dequantizes them once its own oneDNN share is done, with all its
    OpenMP threads. (Dequantizing them meanwhile on a spare core was tried: its
    72 MiB a chunk through the L3 slowed the caller's oneDNN share by ~35 %,
    more than it saved.)
    """

    def __init__(self, gemm, R, rows, sx, out):
        self.gemm, self.sx, self.out = gemm, sx, out
        self.chunks = [(r0, min(r0 + rows, R)) for r0 in range(0, R, rows)]
        self.ready = 0  # chunks whose launch has returned
        self.cond = threading.Condition()
        self.error = None
        self.t_submit = time.perf_counter()
        self.t_last = None  # when the last launch returned

    def finish(self):
        """The caller's part: dequantize each chunk as soon as its launch is in."""
        for i, (r0, r1) in enumerate(self.chunks):
            with self.cond:
                while self.ready <= i and self.error is None:
                    self.cond.wait()
                if self.error is not None:
                    raise RuntimeError("NPU GEMM failed") from self.error
            self.gemm.dequant(r0, r1, self.sx[r0:r1], self.out[r0:r1])


class _NpuWorker:
    """The NPU launch thread, on a core no OpenMP thread uses."""

    def __init__(self):
        self.jobs = queue.Queue()
        cpu = _npu_thread_cpu()
        threading.Thread(target=self._loop, args=(cpu,), daemon=True, name="xdna-npu").start()
        if cpu is not None:
            logger.info("xdna_int8: NPU launch thread on CPU %d", cpu)

    def _loop(self, cpu):
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})  # this thread only
        npu_runtime.use_npu()
        while True:
            job = self.jobs.get()
            try:
                for i, (r0, r1) in enumerate(job.chunks):
                    job.gemm.launch_rows(r0, r1 - r0)
                    with job.cond:
                        job.ready = i + 1
                        if job.ready == len(job.chunks):
                            job.t_last = time.perf_counter()
                        job.cond.notify_all()
            except BaseException as e:  # hand the error to the caller
                with job.cond:
                    job.error = e
                    job.cond.notify_all()

    def submit(self, gemm, R, sx, out):
        job = _Job(gemm, R, LAUNCH_ROWS if R >= LAUNCH_ROWS else npu.BLOCK_M, sx, out)
        self.jobs.put(job)
        return job


_WORKER = None


class _SplitTuner:
    """Picks the NPU's share of rows for one (N, K, M // 512).

    At load, every candidate (0, 512, ... rows) is timed a few times and the
    fastest by median is the starting split. 0 (CPU only) is among them, so a
    shape where the NPU does not pay for its quantize/dequantize stays on the
    CPU. From then on the split follows the machine: after each call, if the
    caller had to wait for the NPU's rows it gives the NPU 512 fewer, and if
    the NPU's rows were ready well before the CPU's, 512 more. Load from other
    processes (on the CPU, the DRAM, or the NPU itself) moves the balance a
    timing made at load would not see.
    """

    SAMPLES = 3
    PROBE_EVERY = 64  # calls, at a split of 0: try 512 rows again

    def __init__(self, candidates):
        self.times = {r: [] for r in candidates}
        self.top = max(candidates)
        self.best = None
        self.calls = 0
        self.calibrated = False
        self.lean = 0.0  # smoothed (NPU done - CPU done), seconds

    def exploring(self):
        return any(len(ts) < self.SAMPLES for ts in self.times.values())

    def summary(self):
        return ", ".join(f"{r}: {1e3 * sorted(ts)[len(ts) // 2]:.1f} ms" for r, ts in self.times.items() if ts)

    def pick(self):
        self.calls += 1
        for r, ts in self.times.items():
            if len(ts) < self.SAMPLES:
                return r
        if self.best is None:
            med = {r: sorted(ts)[len(ts) // 2] for r, ts in self.times.items()}
            self.best = min(med, key=med.get)
        if self.best == 0 and self.calibrated and self.calls % self.PROBE_EVERY == 0:
            return min(512, self.top)
        return self.best

    def record(self, r, t):
        if r in self.times and not self.calibrated:
            self.times[r].append(t)

    def balance(self, r, late, per_chunk):
        """After a call with r NPU rows: late = when its NPU rows were all in, minus when the
        CPU finished its own (s); per_chunk = the NPU's time per 512 rows (s)."""
        if not self.calibrated or self.best is None:
            return
        if self.best == 0:  # a probe: keep it if the NPU's rows came back first
            if r > 0 and late < 0:
                self.best, self.lean = r, 0.0
            return
        if r != self.best:
            return
        self.lean = 0.7 * self.lean + 0.3 * late
        if self.lean > 0.5 * per_chunk:
            self.best, self.lean = self.best - 512, 0.0
        elif self.lean < -1.5 * per_chunk and self.best + 512 <= self.top:
            self.best, self.lean = self.best + 512, 0.0


class Int8Linear:
    """One linear's int8 weight, its oneDNN handle, and (if it has one) its NPU twin."""

    def __init__(self, wq, ws, name, allow_npu):
        global _WORKER
        self.name = name
        self.N, self.K = wq.shape
        # oneDNN takes the weight [K, N] and repacks it; dynamic per-token
        # activation quantization, bf16 out. It keeps a pointer to the scales,
        # not a copy, so they must live as long as the handle.
        self.w_scale = ws.float()[:, None].contiguous()
        self.dnnl = ops.create_onednn_scaled_mm(wq.t(), self.w_scale, torch.bfloat16, True, False, 32)
        self.dnnl_handle = self.dnnl.handler_tensor
        self.npu = None
        if allow_npu and USE_NPU and self.K % npu.BLOCK_K == 0 and self.N >= npu.BLOCK_N:
            self.npu = npu.NpuGemm(wq, ws)
            if _WORKER is None:
                _WORKER = _NpuWorker()
        if self.npu is not None:
            self._calibrate()

    def cpu(self, x, out):
        """oneDNN W8A8 on the CPU: x [M, K] contiguous -> out [M, N].

        The ops are called directly: at decode every linear is a few rows, and
        the _custom_ops wrappers' Python is a noticeable share of a token.
        """
        M = x.shape[0]
        xq = torch.empty((M, self.K), dtype=torch.int8)
        xs = torch.empty((M, 1), dtype=torch.float32)
        _QUANT(xq, x, xs, None)
        _ONEDNN_MM(out, xq, xs, None, None, None, self.dnnl_handle)
        return out

    def _tuner(self, M):
        """The split tuner for this shape and call size, shared by all layers of the shape."""
        key = (self.N, self.K, M // npu.BLOCK_M)
        if key not in _TUNERS:
            top = min((M * 3 // 4) // npu.BLOCK_M * npu.BLOCK_M, npu.MAX_ROWS)
            _TUNERS[key] = _SplitTuner(list(range(0, top + 1, npu.BLOCK_M)))
        return _TUNERS[key]

    def _calibrate(self):
        """Time the splits for full prefill chunks once per shape, at load (this also
        compiles the NPU kernel), so serving starts from a measured split."""
        for M in _calibrate_rows():
            t = self._tuner(M)
            if t.calibrated:
                continue
            x = torch.randn(M, self.K).to(torch.bfloat16)
            while t.exploring():
                self(x)
            t.calibrated = True
            logger.info("xdna_int8 [%d, %d] M=%d: NPU rows %d (%s)", self.N, self.K, M, t.pick(), t.summary())

    def npu_rows_for(self, M):
        if self.npu is None or M < NPU_MIN_ROWS or not NPU_ENABLED[0]:
            return 0
        if FIXED_SHARE[0] is not None:
            return min(int(M * FIXED_SHARE[0]) // npu.BLOCK_M * npu.BLOCK_M, npu.MAX_ROWS)
        return self._tuner(M).pick()

    def __call__(self, x):
        t0 = time.perf_counter()
        y, R = self._run(x)
        if TRACE and x.shape[0] >= 256:
            print(f"[xdna] {self.name} M={x.shape[0]} npu_rows={R} {1e3 * (time.perf_counter() - t0):.2f} ms",
                  file=sys.stderr, flush=True)
        return y

    def _run(self, x):
        M = x.shape[0]
        out = torch.empty((M, self.N), dtype=torch.bfloat16)
        R = self.npu_rows_for(M)
        STATS["calls"] += 1
        t0 = time.perf_counter()
        if R == 0:
            STATS["cpu_rows"] += M
            self.cpu(x, out)
            if self.npu is not None and M >= NPU_MIN_ROWS:
                self._tuner(M).record(0, time.perf_counter() - t0)
            return out, 0
        g = self.npu
        sx = g.quantize(x[:R])
        job = _WORKER.submit(g, R, sx, out)
        if M > R:
            self.cpu(x[R:], out[R:])
        t_cpu = time.perf_counter()
        job.finish()
        t_end = time.perf_counter()
        STATS["t_cpu_share"] += t_cpu - t0
        STATS["t_wait"] += t_end - t_cpu
        STATS["t_npu"] += (job.t_last or t_end) - job.t_submit
        tuner = self._tuner(M)
        tuner.record(R, time.perf_counter() - t0)
        if job.t_last is not None:
            tuner.balance(R, job.t_last - t_cpu, (job.t_last - job.t_submit) * npu.BLOCK_M / R)
        STATS["npu_rows"] += R
        STATS["cpu_rows"] += M - R
        return out, R


@torch.library.custom_op("xdna::int8_linear", mutates_args=())
def int8_linear(x: torch.Tensor, key: int) -> torch.Tensor:
    lin = _LAYERS[key]
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if x2.shape[0] < NPU_MIN_ROWS and not TRACE:  # decode, short prompts: straight to oneDNN
        y = lin.cpu(x2, torch.empty((x2.shape[0], lin.N), dtype=torch.bfloat16))
    else:
        y = lin(x2)
    return y.view(*shape[:-1], lin.N)


@int8_linear.register_fake
def _(x, key):
    return x.new_empty((*x.shape[:-1], _LAYERS[key].N))


def _register(layer, wq, ws, name, allow_npu):
    _LAYERS.append(Int8Linear(wq, ws, name, allow_npu))
    layer._xdna_key = len(_LAYERS) - 1
    return _LAYERS[-1]


class XdnaInt8LinearMethod(UnquantizedLinearMethod):
    """int8 from the start: each checkpoint shard is quantized as it loads.

    Per-row (per output channel) scales do not care how the rows are split
    into shards, so q/k/v or gate/up land in the fused weight exactly as if
    the fused bf16 weight had been quantized -- without ever holding it.
    """

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        assert getattr(layer, "tp_size", 1) == 1, "xdna_int8 assumes no tensor parallelism"
        load = extra_weight_attrs.pop("weight_loader")
        N, K = sum(output_partition_sizes), input_size_per_partition
        # The scales load through the layer's own loader too, so they land in
        # the same rows as their shard.
        scale = ModelWeightParameter(data=torch.zeros(N, 1, dtype=torch.float32), input_dim=1, output_dim=0,
                                     weight_loader=load)

        def weight_loader(param, loaded_weight, *args, **kwargs):
            q, sc = _quantize(loaded_weight)
            sc = sc[:, None].contiguous()
            # The loader reads which shard this is (q / k / v, gate / up)
            # off the tensor itself.
            for k, v in vars(loaded_weight).items():
                setattr(q, k, v)
                setattr(sc, k, v)
            load(param, q, *args, **kwargs)
            load(scale, sc, *args, **kwargs)

        weight = ModelWeightParameter(data=torch.zeros(N, K, dtype=torch.int8), input_dim=1, output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.xdna_scale = scale

    def process_weights_after_loading(self, layer):
        if layer.weight.is_meta:
            return
        lin = _register(layer, layer.weight.data, layer.xdna_scale.data[:, 0].clone(), self.prefix,
                        allow_npu=bool(NPU_LAYERS.search(self.prefix)))
        logger.debug("xdna_int8 %s: [%d, %d]%s", self.prefix, lin.N, lin.K, " +NPU" if lin.npu else "")
        # oneDNN and the NPU hold their own copies now.
        layer.weight = Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False)
        del layer.xdna_scale

    def apply(self, layer, x, bias=None):
        y = torch.ops.xdna.int8_linear(x, layer._xdna_key)
        return y + bias if bias is not None else y


class Int8EmbeddingMethod(UnquantizedEmbeddingMethod):
    """An int8 table, quantized row by row as the checkpoint loads; lookups dequantize.

    The scales hang off the weight Parameter (`xdna_scale`), so an LM head
    tied to this table finds them too.
    """

    def __init__(self, group=None):
        self.group = group

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        assert getattr(layer, "tp_size", 1) == 1, "xdna_int8 embeddings assume no tensor parallelism"
        V, K = sum(output_partition_sizes), input_size_per_partition
        group = self.group if self.group and K % self.group == 0 else None
        weight = Parameter(torch.zeros(V, K, dtype=torch.int8), requires_grad=False)
        weight.xdna_scale = torch.zeros(V, K // group if group else 1, dtype=torch.float32)
        extra_weight_attrs.pop("weight_loader", None)

        def weight_loader(param, loaded_weight):
            q, sc = _quantize(loaded_weight, group)
            param.data[: q.shape[0]].copy_(q)
            param.xdna_scale[: q.shape[0]].copy_(sc.view(q.shape[0], -1))

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0, "weight_loader": weight_loader})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        pass

    def embedding(self, layer, input_):
        w = layer.weight
        q = w[input_]                                    # int8 [..., K]
        s = w.xdna_scale[input_]                         # [..., G]
        G = s.shape[-1]
        y = q.view(*q.shape[:-1], G, -1).float() * s[..., None]
        return y.view(q.shape).to(torch.bfloat16)


class XdnaInt8LMHeadMethod(Int8EmbeddingMethod):
    """Logits from the int8 table (tied to the input embedding): oneDNN int8 on the CPU."""

    def process_weights_after_loading(self, layer):
        w = layer.weight
        _register(layer, w.data, w.xdna_scale[:, 0].clone(), "lm_head", allow_npu=False)

    def apply(self, layer, x, bias=None):
        y = torch.ops.xdna.int8_linear(x, layer._xdna_key)
        return y + bias if bias is not None else y


class XdnaInt8Config(QuantizationConfig):
    def get_name(self):
        return "xdna_int8"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        if any(s in prefix for s in _SKIP):
            return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None
        if isinstance(layer, ParallelLMHead):
            return XdnaInt8LMHeadMethod()
        if isinstance(layer, VocabParallelEmbedding):
            # Per row where it may double as an LM head; the per-layer table
            # (35 layers x 256 wide) gets a scale per layer.
            return Int8EmbeddingMethod(group=256 if "per_layer" in prefix else None)
        if isinstance(layer, LinearBase):
            return XdnaInt8LinearMethod(prefix)
        return None
