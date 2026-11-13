"""Stdlib-only GGUF header reader: metadata (scalars and short arrays) and tensor infos with byte sizes.

Used by ft-fit.py so it never imports torch. Sizes are the gap to the next tensor's offset (the last runs to
the end of the file), as in freetoken/ram_preflight.py.
"""
from __future__ import annotations

import os
import struct
import sys

_FIXED = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# ggml type ids → names (enough for the formats seen here; unknown ids print as their number)
GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
        12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
        20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
        29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4"}


def type_name(t: int) -> str:
    return GGML.get(t, str(t))


def read_header(path: str, max_array: int = 64):
    """(meta, tensors): meta maps key → value (arrays longer than max_array → their length as ("len", n));
    tensors is a list of dicts name, shape (ne, fastest first), type, nbytes."""
    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        rd("I")
        n_tensors, n_kv = rd("Q"), rd("Q")
        string = lambda: f.read(rd("Q")).decode("utf-8", "replace")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = rd("I"), rd("Q")
                if n > max_array:
                    if et in _FIXED:
                        f.seek(n * struct.calcsize(_FIXED[et]), 1)
                    else:
                        for _ in range(n):
                            value(et)
                    return ("len", n)
                return [value(et) for _ in range(n)]
            return rd(_FIXED[t])

        meta = {}
        for _ in range(n_kv):
            key = string()
            meta[key] = value(rd("I"))
        align = int(meta.get("general.alignment", 32))
        infos = []
        for _ in range(n_tensors):
            name = string()
            shape = [rd("Q") for _ in range(rd("I"))]
            t = rd("I")
            infos.append({"name": name, "shape": shape, "type": t, "offset": rd("Q")})
        data = -(-f.tell() // align) * align
    end = os.path.getsize(path) - data
    order = sorted(infos, key=lambda i: i["offset"])
    for cur, nxt in zip(order, order[1:] + [None]):
        cur["nbytes"] = (nxt["offset"] if nxt else end) - cur["offset"]
    return meta, infos


if __name__ == "__main__":  # dump: gguf_header.py <gguf> [key-prefix]
    meta, tensors = read_header(sys.argv[1])
    pre = sys.argv[2] if len(sys.argv) > 2 else ""
    for k, v in meta.items():
        if k.startswith(pre) and not k.startswith("tokenizer."):
            print(f"{k} = {v}")
    print(f"{len(tensors)} tensors, {sum(t['nbytes'] for t in tensors) / 2**30:.2f} GiB")
