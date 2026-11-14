"""Hold N GiB of touched RAM until stdin closes. usage: balloon.py GIB"""
import sys

CH = 256 << 20
held = []
for _ in range(int(float(sys.argv[1]) * 4)):
    b = bytearray(CH)
    b[::4096] = b"\x01" * (CH // 4096)  # touch every page
    held.append(b)
print("ready", flush=True)
sys.stdin.read()
