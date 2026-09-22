import socket, sys, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 5201)); s.listen(1)
c, _ = s.accept()
c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
buf = bytearray(4 << 20); mv = memoryview(buf); total = 0; t0 = None
while True:
    n = c.recv_into(mv, len(buf))
    if not n: break
    if t0 is None: t0 = time.monotonic()
    total += n
dt = time.monotonic() - t0
print(f"recv {total/2**30:.2f} GiB in {dt:.2f}s = {total/1e9/dt:.2f} GB/s = {total*8/1e9/dt:.1f} Gb/s")
