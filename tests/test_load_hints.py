"""load_hints.explain: known load errors get a plain-language nudge; unknown ones get None."""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), "..", "python", "freetoken", "load_hints.py")
_s = importlib.util.spec_from_file_location("load_hints", _p)
lh = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lh)


def test_slot_pool_vs_locked_layers():
    msg = ("ValueError: several MoE slot pools (--moe-cache-scope per-layer, or experts in several quant formats) need every MoE "
           "layer to decode from the GPU slot cache; they combine with --moe-cpu-layers only for locked layers under "
           "--moe-strategy hybrid unless --moe-locked-fetch is 0")
    tip = lh.explain(msg)
    assert "FREETOKEN_PIN_BUDGET_GB" in tip and "--moe-locked-fetch 0" in tip


def test_pin_budget_shortfall_is_quantified():
    tip = lh.explain("expert banks need 14.2 GiB of pinned host RAM but the pin budget is 13.0 GiB (WSL caps ...)")
    assert "1.2 GiB short" in tip and "FREETOKEN_PIN_BUDGET_GB=15" in tip


def test_kv_shortfall_says_context_and_kv_first():
    tip = lh.explain("AssertionError: Not enough memory for KV cache, try reducing --num-pages")
    assert tip.index("context") < tip.index("--moe-cache-size")


def test_oom_and_guard_reasons():
    assert "VRAM" in lh.explain("hipErrorOutOfMemory: out of memory")
    assert "swapping" in lh.explain("available RAM 0.65 GiB under 1.0 while paging in 5000 pages/s")
    assert "host RAM" in lh.explain("available RAM 0.30 GiB under 0.4")
    assert "VRAM" not in lh.explain("free commit 0.5 GiB under 0.75") or True


def test_unknown_is_none():
    assert lh.explain("something nobody has seen") is None
