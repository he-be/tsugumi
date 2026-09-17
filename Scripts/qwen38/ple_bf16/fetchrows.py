"""ds4 Q2 GGUF の per_layer_token_embd.weight (BF16) から、指定トークン列が引く行だけを 1 本の接続・複数レンジで取り、疎な BF16 PLE GGUF に書く。"""
import sys, os, time, http.client, urllib.parse, resource, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from plerows import rows_for, load
URL = 'https://huggingface.co/antirez/qwen3.8-flash-next-gguf/resolve/main/Qwen3.8-Flash-Next-Q2.gguf'
REMOTE0 = 11025376 + 44795610144   # data_start + tensor offset
DST = os.path.expanduser('~/LLM/Qwen3.8-Flash-Next-DS4-IQ2/ple-bf16-sparse/Qwen3.8-Flash-Next-PLE-BF16-sparse.gguf')
MAN = DST + '.rows.npy'
LOCAL0 = 3744
RB = 320
BATCH = int(os.environ.get('BATCH', '256'))

def resolve():
    p = urllib.parse.urlparse(URL)
    c = http.client.HTTPSConnection(p.netloc, timeout=60)
    c.request('HEAD', p.path); r = c.getresponse(); r.read()
    assert r.status in (301, 302, 307), r.status
    loc = r.getheader('Location'); c.close()
    return urllib.parse.urlparse(urllib.parse.urljoin(URL, loc))

def main():
    want = set()
    for p in sys.argv[1:]: want |= set(rows_for(load(p)))
    have = set(np.load(MAN).tolist()) if os.path.exists(MAN) else set()
    todo = sorted(want - have)
    print('want', len(want), 'have', len(have), 'todo', len(todo), flush=True)
    if not todo: return
    loc = resolve(); conn = http.client.HTTPSConnection(loc.netloc, timeout=120)
    path = loc.path + ('?' + loc.query if loc.query else '')
    fd = os.open(DST, os.O_WRONLY)
    done = []; t0 = time.time(); nbytes = 0
    for bi in range(0, len(todo), BATCH):
        chunk = todo[bi:bi + BATCH]
        rng = ','.join(f'{REMOTE0 + r*RB}-{REMOTE0 + r*RB + RB - 1}' for r in chunk)
        for attempt in range(3):
            try:
                conn.request('GET', path, headers={'Range': 'bytes=' + rng})
                resp = conn.getresponse(); body = resp.read()
                if resp.status == 403:
                    loc = resolve(); conn = http.client.HTTPSConnection(loc.netloc, timeout=120)
                    path = loc.path + ('?' + loc.query if loc.query else ''); continue
                break
            except (http.client.HTTPException, OSError) as e:
                print('retry', e, flush=True); conn = http.client.HTTPSConnection(loc.netloc, timeout=120)
        ct = resp.getheader('Content-Type', '')
        assert resp.status == 206, (resp.status, body[:200])
        parts = {}
        if len(chunk) == 1 and 'multipart' not in ct:
            a = int(re.match(r'bytes (\d+)-', resp.getheader('Content-Range')).group(1)); parts[a] = body
        else:
            bd = re.search(r'boundary=(.+)', ct).group(1).strip('"').encode()
            for seg in body.split(b'--' + bd)[1:]:
                if seg.startswith(b'--'): break
                h, _, data = seg.partition(b'\r\n\r\n')
                m = re.search(rb'Content-Range: bytes (\d+)-(\d+)', h, re.I)
                a, e = int(m.group(1)), int(m.group(2))
                parts[a] = data[:e - a + 1]
        for r in chunk:
            d = parts[REMOTE0 + r*RB]
            assert len(d) == RB, len(d)
            assert any(d), f'row {r} all zero'
            os.pwrite(fd, d, LOCAL0 + r*RB); nbytes += RB
        done.extend(chunk)
        if (bi // BATCH) % 100 == 0 or bi + BATCH >= len(todo):
            np.save(MAN, np.array(sorted(have | set(done)), dtype=np.int64))
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            el = time.time() - t0
            print(f'{len(done)}/{len(todo)} rows {el:.0f}s {len(done)/max(el,1e-9):.0f} rows/s maxrss {rss:.0f}MB', flush=True)
    os.close(fd)
main()
