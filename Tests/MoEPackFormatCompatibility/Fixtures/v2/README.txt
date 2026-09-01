The same manifest after the container was renamed from GTURBO to MOEPACK
(2026-09-01). It is byte-for-byte the v1 manifest with that one field changed,
which is the whole of the format delta: every other frozen artifact in v1 is
still what the current writers emit.

v1 stays next to this on purpose. It is what every model installed before the
rename carries, and what the published Hugging Face packs still carry, so the
suite reads it back to prove the legacy magic is still accepted.
