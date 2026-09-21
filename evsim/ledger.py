"""In-process permissioned ledger: hash-chained blocks, Ed25519 signatures, smart-contract rules and a simulated
PBFT-style commit protocol. Cryptographic costs are measured; network delays between validators are simulated."""
import hashlib, json, time
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

H = lambda b: hashlib.sha256(b).hexdigest()


def canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


class Identity:
    def __init__(self, name):
        self.name = name
        self.sk = Ed25519PrivateKey.generate()
        self.pk = self.sk.public_key()
    def sign(self, msg: bytes) -> bytes:
        return self.sk.sign(msg)


def verify(pk, sig, msg):
    try:
        pk.verify(sig, msg); return True
    except InvalidSignature:
        return False


class Contract:
    """Smart-contract rules enforced before a transaction enters a block."""
    def __init__(self, ra_pk, C=6, m_lo=0.6, m_hi=1.8):
        self.ra_pk, self.C, self.m_lo, self.m_hi = ra_pk, C, m_lo, m_hi
        self.registered, self.nonces = {}, {}

    def check(self, tx, pk_of):
        t, p = tx["type"], tx["payload"]
        if self.nonces.get(tx["sender"], -1) >= tx["nonce"]:
            return False, "replay"
        if t == "register":
            if not verify(self.ra_pk, bytes.fromhex(p["ra_sig"]), canon({"station": p["station"], "pk": p["pk"]})):
                return False, "no_ra_signature"
        elif tx["sender"] not in self.registered and t != "register":
            return False, "unregistered_sender"
        if t == "command":
            if not (self.m_lo <= p["m"] <= self.m_hi and 1 <= p["nact"] <= self.C):
                return False, "out_of_policy"
        if t == "session" and p["energy_kwh"] < 0:
            return False, "negative_energy"
        return True, "ok"

    def apply(self, tx):
        self.nonces[tx["sender"]] = tx["nonce"]
        if tx["type"] == "register":
            self.registered[tx["sender"]] = tx["payload"]["pk"]


class Ledger:
    def __init__(self, n_validators=4, block_size=50, rng=None, hop_ms=(20.0, 0.35), ra=None, C=6):
        self.nv, self.bs = n_validators, block_size
        self.rng = rng or np.random.default_rng(0)
        self.hop_mu, self.hop_sigma = np.log(hop_ms[0]), hop_ms[1]
        self.ra = ra or Identity("RA")
        self.contract = Contract(self.ra.pk, C=C)
        self.pk = {}                # sender -> public key
        self.pool, self.chain = [], []
        self.genesis = {"index": 0, "prev": "0" * 64, "root": H(b""), "txs": [], "ts": 0.0}
        self.genesis["hash"] = H(canon({k: self.genesis[k] for k in ("index", "prev", "root", "ts")}))
        self.chain.append(self.genesis)
        self.stats = dict(rejected={}, sign_s=[], verify_s=[], hash_s=[], commit_ms=[], block_tx=[], bytes=0)
        self.nonce = {}

    # ---- client side
    def make_tx(self, ident, typ, payload):
        n = self.nonce.get(ident.name, 0)
        self.nonce[ident.name] = n + 1
        tx = {"sender": ident.name, "type": typ, "payload": payload, "nonce": n}
        t0 = time.perf_counter()
        tx["sig"] = ident.sign(canon({k: tx[k] for k in ("sender", "type", "payload", "nonce")})).hex()
        self.stats["sign_s"].append(time.perf_counter() - t0)
        return tx

    def register(self, ident):
        p = {"station": ident.name, "pk": ident.pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()}
        p["ra_sig"] = self.ra.sign(canon({"station": p["station"], "pk": p["pk"]})).hex()
        self.pk[ident.name] = ident.pk
        return self.make_tx(ident, "register", p)

    # ---- validator side
    def submit(self, tx):
        pk = self.pk.get(tx["sender"])
        t0 = time.perf_counter()
        ok = pk is not None and verify(pk, bytes.fromhex(tx["sig"]), canon({k: tx[k] for k in ("sender", "type", "payload", "nonce")}))
        self.stats["verify_s"].append(time.perf_counter() - t0)
        if not ok:
            self.stats["rejected"]["bad_signature"] = self.stats["rejected"].get("bad_signature", 0) + 1
            return False
        ok, why = self.contract.check(tx, self.pk)
        if not ok:
            self.stats["rejected"][why] = self.stats["rejected"].get(why, 0) + 1
            return False
        self.contract.apply(tx)
        self.pool.append(tx)
        return True

    @staticmethod
    def merkle(txs):
        layer = [H(canon(t)) for t in txs] or [H(b"")]
        while len(layer) > 1:
            if len(layer) % 2: layer.append(layer[-1])
            layer = [H((layer[i] + layer[i + 1]).encode()) for i in range(0, len(layer), 2)]
        return layer[0]

    def commit(self):
        """Form one block from the pool and run a simulated 3-phase PBFT commit (quorum 2f+1 of n=3f+1)."""
        if not self.pool:
            return None
        txs, self.pool = self.pool[:self.bs], self.pool[self.bs:]
        prev = self.chain[-1]
        t0 = time.perf_counter()
        blk = {"index": prev["index"] + 1, "prev": prev["hash"], "root": self.merkle(txs), "ts": float(prev["ts"] + 1), "txs": txs}
        blk["hash"] = H(canon({k: blk[k] for k in ("index", "prev", "root", "ts")}))
        hash_s = time.perf_counter() - t0
        # real cost of block validation by one validator (re-verify merkle root); other validators run in parallel
        t1 = time.perf_counter(); assert self.merkle(txs) == blk["root"]; val_s = time.perf_counter() - t1
        f = (self.nv - 1) // 3
        quorum = 2 * f + 1
        lat = 0.0
        for _ in range(3):                                   # pre-prepare, prepare, commit
            d = self.rng.lognormal(self.hop_mu, self.hop_sigma, self.nv - 1)
            lat += float(np.sort(d)[min(quorum, len(d)) - 1])   # wait for quorum-th fastest reply
        commit_ms = lat + (hash_s + val_s) * 1000.0
        self.chain.append(blk)
        self.stats["hash_s"].append(hash_s + val_s); self.stats["commit_ms"].append(commit_ms); self.stats["block_tx"].append(len(txs))
        self.stats["bytes"] += len(canon(blk))
        return blk

    def verify_chain(self, chain=None):
        """Full audit: hash links, Merkle roots, per-tx signatures. Returns (ok, reason)."""
        ch = chain or self.chain
        for i in range(1, len(ch)):
            b, p = ch[i], ch[i - 1]
            if b["prev"] != p["hash"]:
                return False, "broken_link"
            if H(canon({k: b[k] for k in ("index", "prev", "root", "ts")})) != b["hash"]:
                return False, "bad_block_hash"
            if self.merkle(b["txs"]) != b["root"]:
                return False, "bad_merkle_root"
            for tx in b["txs"]:
                pk = self.pk.get(tx["sender"])
                if pk is None or not verify(pk, bytes.fromhex(tx["sig"]), canon({k: tx[k] for k in ("sender", "type", "payload", "nonce")})):
                    return False, "bad_signature"
        return True, "ok"
