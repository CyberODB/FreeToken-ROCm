"""Run the CPU-MoE flag-handshake probe on its own (it decides whether HIP decode graphs can be used
in hybrid mode). Set FREETOKEN_CPU_MOE_PROBE_DEBUG=1 to see which step fails."""
import os

os.environ.setdefault("FREETOKEN_CPU_MOE_PROBE_DEBUG", "1")
import torch

import freetoken.kernel._cpu_moe as m

print("torch", torch.__version__, torch.cuda.get_device_name())
print("memops_probe:", m.memops_probe(torch.cuda.current_stream().cuda_stream))
print("kernel_io_probe:", repr(m.kernel_io_probe(torch.cuda.current_stream().cuda_stream)) + " (empty = OK)")
for g, f in ((False, False), (True, False), (True, True)):
    print("kernel_graph_probe:", m.kernel_graph_probe(20, f, g))
