"""Precompute frozen-bottleneck memory tokens per (sample, variant).

Replicates the Stage-2 offline gather (dataset.py._gather_cosmo_bottleneck offline branch)
for a SPECIFIC variant, runs the frozen Stage-1 bottleneck, and saves (256, mem_dim) fp16
per (episode, step, variant).

Output: <out>/ep{episode}_t{step}.npz  {v{k}: (256,1024) fp16 for each causal variant}
and <out>/wan_chunk_alignment.txt (checked by the Stage-2 loader; see shared/wan_alignment.py).
Loading these at Stage-2 replaces ~18.5MB/sample of cosmo inputs with ~0.5MB tokens.
"""
import argparse, collections, json, os
import numpy as np

from mme_vla_suite.shared.memory_caps import max_recent_tokens, max_step_tokens
from mme_vla_suite.shared.visual_streams import load_dino_manifest, visual_feature_filename


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--stage1-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-samples", type=int, default=0)   # 0 = all; >0 = trial subset
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    p.add_argument("--batch-samples", type=int, default=8)
    args = p.parse_args(argv)
    if args.nshard < 1 or not 0 <= args.shard < args.nshard:
        p.error("--shard must be in [0, --nshard)")
    if args.batch_samples < 1:
        p.error("--batch-samples must be positive")

    import jax, jax.numpy as jnp, flax.nnx as nnx
    import flax.traverse_util as trav
    from omegaconf import OmegaConf
    from mme_vla_suite.training.pretrain_loss import make_stage1_modules
    from mme_vla_suite.shared import wan_alignment

    cfg = OmegaConf.load(args.config)
    cbm = cfg.cosmo_bottleneck_memory
    visual_stream = str(getattr(cbm, "visual_feature_stream", "siglip"))
    feature_store = str(getattr(cbm, "feature_store_path", "")).strip() or args.store
    feature_dir = os.path.join(feature_store, "features")
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")
    if visual_stream == "dinov3":
        load_dino_manifest(feature_store)
    n_variants = int(getattr(cbm, "n_variants", 8))
    POOL_HW = int(getattr(cbm, "pool_hw", 0)) or None
    pool_h = int(getattr(cbm, "pool_h", 0)) or None
    pool_w = int(getattr(cbm, "pool_w", 0)) or None
    wan_h = int(getattr(cbm, "wan_h", 192))
    wan_w = int(getattr(cbm, "wan_w", 256))
    act_enc = str(getattr(cbm, "action_encoding", "raw_delta"))
    pro_enc = str(getattr(cbm, "proprio_encoding", "raw_delta"))
    action_in_dim = int(getattr(cbm, "action_in_dim", 12))
    proprio_in_dim = int(getattr(cbm, "proprio_in_dim", 16))
    lang_dim = int(getattr(cbm, "lang_emb_dim", 2048))
    MAX_RECENT, MAX_STEPS = max_recent_tokens(cbm), max_step_tokens(cbm)
    LATENT_CAP = int(getattr(cbm, "latent_cap", 0)) or 2048
    print(
        f"[precompute] caps latent={LATENT_CAP} recent={MAX_RECENT} "
        f"action/proprio={MAX_STEPS}",
        flush=True,
    )

    # --- load frozen bottleneck (overlay Stage-1 bottleneck/ weights) ---
    bneck = make_stage1_modules(cfg, seed=0)["bottleneck"]
    gd, state = nnx.split(bneck)
    flat = trav.flatten_dict(state.to_pure_dict())
    s1 = np.load(args.stage1_ckpt, allow_pickle=True)
    wan_alignment.require_recorded_alignment(
        wan_alignment.stage1_checkpoint_alignment(s1), cbm, where=args.stage1_ckpt
    )
    nset = 0
    n_ckpt = 0
    missing = []
    for k in s1.files:
        if not k.startswith("bottleneck/"):
            continue
        n_ckpt += 1
        tup = tuple(int(x) if x.isdigit() else x for x in k[len("bottleneck/"):].split("/"))
        for fk in list(flat.keys()):
            if tuple(str(z) for z in fk[:len(tup)]) == tuple(str(z) for z in tup):
                flat[fk] = np.asarray(s1[k], np.float32); nset += 1; break
        else:
            missing.append(k)
    if nset != n_ckpt:
        raise RuntimeError(
            f"Stage-1 bottleneck overlay incomplete: loaded {nset}/{n_ckpt}; "
            f"missing examples={missing[:5]}"
        )
    state.replace_by_pure_dict(trav.unflatten_dict(flat))
    bneck = nnx.merge(gd, state)
    print(f"[precompute] overlaid {nset} bottleneck params", flush=True)

    @nnx.jit
    def run(lat, rec, ac, pr, mm, lg, rm, am, pm):
        mem, _, _ = bneck(lat, rec, ac, pr, mm, lang_emb=lg, recent_mask=rm, action_mask=am, proprio_mask=pm)
        return mem

    def _fixed_pad(arr, L, D):
        out = np.zeros((L, D), np.float32); msk = np.zeros((L,), bool)
        a = arr[-L:] if arr.shape[0] > L else arr
        if a.shape[0]:
            out[:a.shape[0]] = a; msk[:a.shape[0]] = True
        return out, msk

    def load_variant_causal_action(
        epi_dir, *, variant, full_variant_frame_idx, selected_variant_frame_idx,
        global_causal_actions,
    ):
        full_idx = np.asarray(full_variant_frame_idx, dtype=np.int64)
        selected_idx = np.asarray(selected_variant_frame_idx, dtype=np.int64)
        if selected_idx.size > full_idx.size or not np.array_equal(
            selected_idx, full_idx[:selected_idx.size]
        ):
            raise ValueError(f"selected frame indices are not a prefix of variant v{variant}")
        local_path = os.path.join(epi_dir, f"action_causal_v{variant}.npy")
        if os.path.exists(local_path):
            local_actions = np.load(local_path, mmap_mode="r")
            if local_actions.shape[0] != full_idx.size:
                raise ValueError(
                    f"{os.path.basename(local_path)} length {local_actions.shape[0]} "
                    f"does not match variant_index length {full_idx.size}"
                )
            return np.ascontiguousarray(local_actions[:selected_idx.size]).astype(np.float32)
        return np.ascontiguousarray(global_causal_actions[selected_idx]).astype(np.float32)

    def load_episode(epi_dir):
        vi = json.load(open(os.path.join(epi_dir, "variant_index.json")))
        chunk_index = json.load(open(os.path.join(epi_dir, "chunk_index.json")))
        wan_alignment.require_store_alignment(chunk_index, cbm, where=epi_dir)
        latents = [
            np.load(os.path.join(epi_dir, f"latents_v{v}.npy"), mmap_mode="r")
            for v in range(n_variants)
        ]
        recent = np.load(
            os.path.join(epi_dir, visual_feature_filename(visual_stream)),
            mmap_mode="r",
        )
        action_proprio = np.load(
            os.path.join(epi_dir, "action_proprio.npy"), allow_pickle=True
        ).item()
        lang_path = os.path.join(epi_dir, "lang_emb.npy")
        lang = (
            np.load(lang_path).astype(np.float32).reshape(-1)
            if os.path.exists(lang_path)
            else np.zeros((lang_dim,), np.float32)
        )
        return vi, chunk_index, latents, recent, action_proprio, lang

    def gather_variant(epi_dir, episode_data, t, v):
        """Exact replica of the Stage-2 offline gather for a SPECIFIC variant v; returns padded
        inputs + masks, or None if v is not causal at t."""
        vi, chunk_index, latents, recent_mm, ap, lang = episode_data
        fr = vi[f"v{v}"]["frame_idx"]
        if not fr or fr[0] > t:
            return None
        fidx = np.array(fr, np.int64); fidx = fidx[fidx <= t]; n_jit = len(fidx)
        lat = latents[v]
        chunk_frames = int(chunk_index["chunk_frames"])
        nc = max(0, n_jit // chunk_frames); kept = min(lat.shape[0], nc); lat = lat[:kept]
        latent_dim = lat.shape[-1] if lat.ndim == 3 else 48
        if pool_h and pool_w and lat.shape[0] > 0:
            ncur = lat.shape[0]
            gh = wan_h // 16
            gw = wan_w // 16
            ph = int(pool_h)
            pw = int(pool_w)
            pooled_spatial = ph * pw
            if lat.shape[1] % pooled_spatial == 0 and lat.shape[1] < gh * gw:
                pass  # already pooled on disk: (chunks, views*pool_h*pool_w, dim)
            else:
                spatial = gh * gw
                Vv = lat.shape[1] // spatial
                assert lat.shape[1] % spatial == 0, f"unexpected spatial layout {lat.shape} (grid {gh}x{gw})"
                assert gh % ph == 0 and gw % pw == 0, f"grid {gh}x{gw} must divide by pool ({ph},{pw})"
                bh = gh // ph
                bw = gw // pw
                gg = lat.reshape(ncur, Vv, gh, gw, latent_dim)
                gg = gg.reshape(ncur, Vv, ph, bh, pw, bw, latent_dim).mean(axis=(3, 5))
                lat = gg.reshape(ncur, Vv * ph * pw, latent_dim)
        elif POOL_HW and lat.shape[0] > 0:
            ncur = lat.shape[0]; spatial = 64; Vv = lat.shape[1] // spatial
            side = 8; ph = int(POOL_HW); blk = side // ph
            gg = lat.reshape(ncur, Vv, side, side, latent_dim)
            gg = gg.reshape(ncur, Vv, ph, blk, ph, blk, latent_dim).mean(axis=(3, 5))
            lat = gg.reshape(ncur, Vv * ph * ph, latent_dim)
        clat = (np.ascontiguousarray(lat).reshape(-1, latent_dim).astype(np.float32)
                if lat.shape[0] > 0 else np.zeros((0, latent_dim), np.float32))
        sdim = recent_mm.shape[-1]; vb = kept * chunk_frames
        ridx = fidx[vb:] if vb < n_jit else fidx[n_jit:]
        rec = (np.ascontiguousarray(recent_mm[ridx]).reshape(-1, sdim).astype(np.float32)
               if len(ridx) else np.zeros((0, sdim), np.float32))
        action_mem_src = ap.get("action_causal", ap["action"])
        va = load_variant_causal_action(
            epi_dir,
            variant=v,
            full_variant_frame_idx=fr,
            selected_variant_frame_idx=fidx,
            global_causal_actions=action_mem_src,
        )
        vp = np.ascontiguousarray(ap["proprio"][fidx]).astype(np.float32)
        if act_enc == "raw": ae = va
        else:
            da = np.zeros_like(va); da[1:] = va[1:] - va[:-1]; ae = np.concatenate([va, da], -1)
        if pro_enc == "raw": pe = vp
        else:
            dp = np.zeros_like(vp); dp[1:] = vp[1:] - vp[:-1]; pe = np.concatenate([vp, dp], -1)
        rec_p, rec_m = _fixed_pad(rec, MAX_RECENT, sdim)
        act_dim = ae.shape[-1] if ae.size else (action_in_dim if act_enc == "raw" else action_in_dim * 2)
        pro_dim = pe.shape[-1] if pe.size else (proprio_in_dim if pro_enc == "raw" else proprio_in_dim * 2)
        act_p, act_m = _fixed_pad(ae, MAX_STEPS, act_dim)
        pro_p, pro_m = _fixed_pad(pe, MAX_STEPS, pro_dim)
        lat_p = np.zeros((LATENT_CAP, clat.shape[-1] if clat.size else 48), np.float32)
        lat_m = np.zeros((LATENT_CAP,), bool); lc = min(clat.shape[0], LATENT_CAP)
        if lc: lat_p[:lc] = clat[-LATENT_CAP:][:lc]; lat_m[:lc] = True
        return lat_p, lat_m, rec_p, rec_m, act_p, act_m, pro_p, pro_m, lang

    samples = json.load(open(os.path.join(args.store, "meta", "samples.json")))
    if args.max_samples:
        samples = samples[:args.max_samples]
    os.makedirs(args.out, exist_ok=True)
    if any(name.endswith(".npz") for name in os.listdir(args.out)):
        # Existing tokens are kept, so they must use the same alignment (no record: native).
        # Each shard writes the record before its first tokens.
        wan_alignment.require_recorded_alignment(
            wan_alignment.read_alignment_record(args.out), cbm, where=args.out
        )
    wan_alignment.write_alignment_record(
        args.out, wan_alignment.wan_chunk_alignment_from_config(cbm)
    )
    mine = [i for i in range(len(samples)) if i % args.nshard == args.shard]
    print(
        f"[shard {args.shard}/{args.nshard}] {len(mine)} samples, "
        f"batch_samples={args.batch_samples}",
        flush=True,
    )

    def sample_ep_t(sample):
        if isinstance(sample, dict):
            return int(sample["episode_index"]), int(sample["step_index"])
        return int(sample[0]), int(sample[1])

    def flush(batch):
        flat = [
            (outp, variant, gathered)
            for outp, variants in batch
            for variant, gathered in variants
        ]
        columns = list(zip(*(gathered for _, _, gathered in flat)))
        inputs = [jnp.asarray(np.stack(column, axis=0)) for column in columns]
        mem = np.asarray(run(*inputs), np.float16)
        outputs = collections.defaultdict(dict)
        for (outp, variant, _), tokens in zip(flat, mem, strict=True):
            outputs[outp][f"v{variant}"] = tokens
        for outp, tokens in outputs.items():
            tmp = f"{outp}.tmp.npz"
            np.savez(tmp, **tokens)
            os.replace(tmp, outp)

    done = 0
    last_report = 0
    batch = []
    cached_episode = None
    episode_data = None
    for si in mine:
        epi, t = sample_ep_t(samples[si])
        outp = os.path.join(args.out, f"ep{epi}_t{t}.npz")
        if os.path.exists(outp):
            done += 1
            continue
        epi_dir = os.path.join(feature_dir, f"episode_{epi}")
        if cached_episode != epi:
            episode_data = load_episode(epi_dir)
            cached_episode = epi
        variants = []
        for v in range(n_variants):
            gathered = gather_variant(epi_dir, episode_data, t, v)
            if gathered is not None:
                lat_p, lat_m, rec_p, rec_m, act_p, act_m, pro_p, pro_m, lang = gathered
                variants.append(
                    (v, (lat_p, rec_p, act_p, pro_p, lat_m, lang, rec_m, act_m, pro_m))
                )
        if not variants:
            raise RuntimeError(f"No causal variants for episode={epi}, step={t}")
        batch.append((outp, variants))
        if len(batch) >= args.batch_samples:
            flush(batch)
            batch.clear()
        done += 1
        if done - last_report >= 200:
            print(f"[shard {args.shard}] {done}/{len(mine)}", flush=True)
            last_report = done
    if batch:
        flush(batch)
    print(f"[shard {args.shard}] DONE {done} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
