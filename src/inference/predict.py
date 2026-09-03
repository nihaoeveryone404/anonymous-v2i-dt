# -*- coding: utf-8 -*-
r"""
predictnew.py — 合并最终版（Map-PL / KPI-Map + Legacy Samples 推理都支持）

【v6.6 (Scheme-A for S3!=S2)】
在 PL 的 map 推理中新增输出 per-BS 3D LOS/conf（H×W×K），用于决策阶段的 ua-rate-alpha-beta：
- 新增保存：{city_id}_{model_dir}_LOS.npy   shape=[H,W,K]，值域[0,1]
- 3D LOS/conf 的构造（轻量、无需额外模型）：
    conf3d = pr_mw / (max_k pr_mw + eps)
  这样每个像素“最强基站”对应 conf≈1，其它基站 conf<1，S3 不再退化为 S2。
- 策略可视化：ua-rate-alpha-beta 优先使用：
    Step1 的 LOS(若存在且维度匹配) > 本文件生成的 3D LOS > 2D pseudoLOS repeat

【v6.5-hotfix for KPI4】
修复 KPI4（以及其它可能 std 极小的 KPI 维度）反归一化爆炸的问题：
- 原来：y_std[y_std < 1e-6] = 1.0  会把极小 std 强行改成 1，导致 kpi_phys 被拉到 -0.1 这种离谱量级
- 现在：y_std[y_std < KPI_STD_EPS] = KPI_STD_EPS  （默认 1e-6）
"""

import os, csv, json, argparse, re, pickle
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt  # 新增：用于生成 PDF 矢量图

from src.models.cpfl_models import PFL_KPIPredictor, PFL_REMNet
from src.utils.io_utils import parse_args_with_config

# ========================== 默认路径（可用命令行覆盖） ==========================
ROOT_PATH_DEFAULT = "data/RadioMapSeer"
DATASET_PREP_OUTPUT_DIR_DEFAULT = "outputs/dataset"

MODEL_CKPT_DEFAULT_PL = ""
MODEL_CKPT_DEFAULT_KPI = ""

OUTPUTS_PRED_DIR_DEFAULT_PL = "outputs/predictions/pl"
OUTPUTS_PRED_DIR_DEFAULT_KPI = "outputs/predictions/kpi"

# local head 默认目录（你 head-only 拟合输出目录）
LOCAL_HEAD_DIR_DEFAULT_PL = ""
LOCAL_HEAD_DIR_DEFAULT_KPI = ""

# ========================== 常量（与步骤1/2一致/兼容） ==========================
TASK_CHOICES = ["pl", "kpi"]
MODE_CHOICES = ["map", "samples"]

MODEL_DB_RANGES = {
    "IRT2": (True, -160.0, -40.0),
    "carsIRT2": (True, -160.0, -40.0),
    "IRT4": (True, -160.0, -40.0),
    "carsIRT4": (True, -160.0, -40.0),
    "DPM": (True, -160.0, -40.0),
    "carsDPM": (True, -160.0, -40.0),
}

B_HZ = 10_000_000
TX_POWER_DBM = 23.0
NOISE_PSD_DBM_PER_HZ = -174.0
NOISE_FIGURE_DB = 5.0
INTERF_COEFF = 1.0
ALPHA = 1.0
BETA = 1.0
CONF_NLOS = 0.5
SAFETY_EPS = 1e-12

KPI_NUM_OUTPUTS = 15
INPUT_DIM_DEFAULT = 102  # create_feature_matrix 输出维度（x,y,building,road,car + 98 距离特征）

# ===== ✅ KPI 标准差下限（关键修复点） =====
KPI_STD_EPS = 1e-6


# ========================== 工具函数 ==========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pjoin(*x):
    return os.path.join(*x).replace("/", os.sep)


def log(s: str):
    print(s, flush=True)


# ---------- ckpt loader（兼容 weights_only & numpy对象，来自源版思想） ----------
def torch_load_any(path: str, map_location="cpu",
                   trust_ckpt: bool = True,
                   prefer_weights_only: bool = True):
    """
    兼容 torch.load 的 weights_only=True 安全模式与旧 ckpt / numpy 对象。
    - prefer_weights_only=True：优先走 weights_only=True
    - 若遇到 weights_only 的 UnpicklingError：
        * trust_ckpt=True：尝试 allowlist numpy reconstruct，再不行回退 weights_only=False
        * trust_ckpt=False：直接抛出
    """
    if not prefer_weights_only:
        try:
            return torch.load(path, map_location=map_location, weights_only=(not trust_ckpt))
        except TypeError:
            return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        msg = str(e)
        is_wo_fail = ("Weights only load failed" in msg) or isinstance(e, pickle.UnpicklingError)
        if (not is_wo_fail) or (not trust_ckpt):
            raise

        try:
            import numpy.core.multiarray as _np_multiarray
            import torch.serialization as _ts
            _ts.add_safe_globals([_np_multiarray._reconstruct])
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            pass

        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        tensor_like = 0
        for v in ckpt_obj.values():
            if torch.is_tensor(v):
                tensor_like += 1
        if len(ckpt_obj) > 0 and tensor_like >= max(1, int(0.6 * len(ckpt_obj))):
            return ckpt_obj

        for k in ["state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    raise RuntimeError("Cannot extract state_dict from checkpoint object.")


def strip_prefix_if_present(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sd
    out = {}
    for k, v in sd.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


def remap_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        sd = strip_prefix_if_present(sd, "module.")

    out = {}
    for k, v in sd.items():
        kk = k
        kk = kk.replace("backbone.backbone.", "backbone.")
        kk = kk.replace("head.net.", "head.")
        kk = kk.replace("head.head.", "head.")
        kk = kk.replace("module.head.net.", "head.")
        out[kk] = v
    return out


# ========================== 视觉输出小工具 (已修改为生成 PDF 矢量图) ==========================
def save_quick_pdf(arr: np.ndarray, out_pdf: str, vmin=None, vmax=None):
    a = arr.astype(np.float32)
    if vmin is None:
        vmin = float(np.nanpercentile(a, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(a, 99))

    # 构建适合分辨率的画布
    h, w = arr.shape
    plt.figure(figsize=(w / 100, h / 100), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    # interpolation='none' 可以确保放大时每个像素块边缘锐利，不发虚
    plt.imshow(a, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')

    # 保存为 PDF
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def noise_dbm(b_hz: float) -> float:
    return NOISE_PSD_DBM_PER_HZ + 10.0 * np.log10(b_hz) + NOISE_FIGURE_DB


# ========================== manifests / geom / tx ==========================
def load_geom_map(geom_csv_path: str) -> Dict[int, dict]:
    mp: Dict[int, dict] = {}
    with open(geom_csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mp[int(row["city_id"])] = row
    return mp


_TX_COL_PATTERNS = (
    re.compile(r"^tx(_)?\d+$", re.IGNORECASE),
    re.compile(r"^bs(_)?\d+$", re.IGNORECASE),
)


def extract_tx_ids_from_multibs_row(row: pd.Series) -> List[int]:
    cols = []
    for c in row.index:
        s = str(c)
        if any(p.match(s) for p in _TX_COL_PATTERNS):
            cols.append(s)

    def col_key(name: str):
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else 10 ** 9

    cols = sorted(cols, key=col_key)
    tx_ids: List[int] = []
    for c in cols:
        v = row[c]
        if pd.isna(v):
            continue
        try:
            vi = int(v)
        except Exception:
            try:
                vi = int(float(v))
            except Exception:
                continue
        if vi < 0:
            continue
        tx_ids.append(vi)
    return tx_ids


# ========================== norm params（PL） ==========================
def find_norm_params(norm_dir: str, city_id: int, model_dir: str) -> Optional[str]:
    cand1 = pjoin(norm_dir, f"city_{city_id}_model_{model_dir}_mean_std.npz")
    cand2 = pjoin(norm_dir, f"city_{city_id}_mean_std.npz")
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    return None


def load_pl_norm(norm_dir: str, city_id: int, model_dir: str):
    norm_path = find_norm_params(norm_dir, city_id, model_dir)
    if not norm_path:
        return None, None, None
    nd = np.load(norm_path)
    if "y_mean" not in nd or "y_std" not in nd:
        return None, None, None
    y_mean = nd["y_mean"].astype(np.float32)
    y_std = nd["y_std"].astype(np.float32)
    k = int(y_mean.shape[0])
    return norm_path, y_mean, y_std


# ========================== KPI stats（从 users_kpi 反推 mean/std） ==========================
def load_kpi_stats_from_users(prep_out_dir: str, city_id: int, model_dir: str):
    users_kpi_dir = pjoin(prep_out_dir, "users_kpi")
    fname = f"user_{city_id}_{model_dir}_kpi.npz"
    path = pjoin(users_kpi_dir, fname)
    if not os.path.exists(path):
        log(f"[WARN] KPI stats: {path} 不存在，KPI 将保持标准化空间输出。")
        return None, None

    data = np.load(path, allow_pickle=True)
    candidate_arr = None
    for name in data.files:
        arr = data[name]
        if hasattr(arr, "shape") and arr.shape[-1] == KPI_NUM_OUTPUTS:
            candidate_arr = arr.astype(np.float32)
            break
    if candidate_arr is None:
        log(f"[WARN] KPI stats: {path} 中未找到最后一维为 {KPI_NUM_OUTPUTS} 的数组。")
        return None, None

    arr = candidate_arr
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, KPI_NUM_OUTPUTS)

    y_mean = arr.mean(axis=0)
    y_std = arr.std(axis=0)

    # ===== ✅ 关键修复：极小 std 不要置 1.0，而是置为一个很小的 eps =====
    y_std = np.where(np.isfinite(y_std), y_std, KPI_STD_EPS).astype(np.float32)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0).astype(np.float32)
    y_std[y_std < KPI_STD_EPS] = KPI_STD_EPS

    try:
        idx_sorted = np.argsort(y_std)
        small_info = ", ".join([f"k{i + 1}={float(y_std[i]):.3e}" for i in idx_sorted[:5]])
        log(f"[KPI stats] city={city_id} model={model_dir} mean/std 由 users_kpi 反推 | smallest std: {small_info}")
    except Exception:
        log(f"[KPI stats] city={city_id} model={model_dir} mean/std 由 users_kpi 反推。")

    return y_mean.astype(np.float32), y_std.astype(np.float32)


# ========================== 特征构造（map 模式用） ==========================
def create_feature_matrix(city_id: int, geom_row: dict, tx_coords_df: pd.DataFrame,
                          target_hw: Tuple[int, int], root_path: str,
                          tx_ids: List[int],
                          max_tx_slots: int = 98) -> np.ndarray:
    H, W = target_hw
    N = H * W

    buildings_path = pjoin(root_path, geom_row.get("buildings_png", ""))
    roads_path = pjoin(root_path, geom_row.get("roads_png", "")) if geom_row.get("roads_png", "") else ""
    cars_path = pjoin(root_path, geom_row.get("cars_png", "")) if geom_row.get("cars_png", "") else ""

    b_img = Image.open(buildings_path).convert("L").resize((W, H), Image.NEAREST)
    r_img = Image.open(roads_path).convert("L").resize((W, H), Image.NEAREST) if roads_path else Image.new("L", (W, H),
                                                                                                           0)
    c_img = Image.open(cars_path).convert("L").resize((W, H), Image.NEAREST) if cars_path else Image.new("L", (W, H), 0)

    b_arr = np.asarray(b_img, dtype=np.float32)
    r_arr = np.asarray(r_img, dtype=np.float32)
    c_arr = np.asarray(c_img, dtype=np.float32)

    xg, yg = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x_norm = xg / W
    y_norm = yg / H

    X = np.zeros((N, INPUT_DIM_DEFAULT), dtype=np.float32)
    X[:, 0] = x_norm.flatten()
    X[:, 1] = y_norm.flatten()
    X[:, 2] = b_arr.flatten() / 255.0
    X[:, 3] = r_arr.flatten() / 255.0
    X[:, 4] = c_arr.flatten() / 255.0

    use_tx_id = ("tx_id" in tx_coords_df.columns)
    tx_city = tx_coords_df[tx_coords_df["city_id"] == city_id]

    if use_tx_id:
        m = {}
        for _, rr in tx_city.iterrows():
            try:
                m[int(rr["tx_id"])] = (float(rr["tx_x"]), float(rr["tx_y"]))
            except Exception:
                continue
        coords = [m.get(tid, None) for tid in tx_ids]
        for k_idx, xy in enumerate(coords[:max_tx_slots]):
            if xy is None:
                continue
            tx_x, tx_y = xy
            tx_xn = tx_x / W
            tx_yn = tx_y / H
            dist = np.sqrt((x_norm - tx_xn) ** 2 + (y_norm - tx_yn) ** 2)
            X[:, 5 + k_idx] = dist.flatten()
    else:
        tx_x = tx_city["tx_x"].values
        tx_y = tx_city["tx_y"].values
        tx_xn = tx_x / W
        tx_yn = tx_y / H
        for k_idx, (tx_xi, tx_yi) in enumerate(zip(tx_xn, tx_yn)):
            if k_idx >= max_tx_slots:
                break
            dist = np.sqrt((x_norm - tx_xi) ** 2 + (y_norm - tx_yi) ** 2)
            X[:, 5 + k_idx] = dist.flatten()

    return X  # [N,102]


# ========================== 后处理：dB -> pr/sinr/rate & pseudoLOS ==========================
def dB_stack_to_pr_rate(dB_stack: np.ndarray, model_dir: str):
    is_gain = MODEL_DB_RANGES.get(model_dir, (True, -160.0, -40.0))[0]
    pr_dbm = (TX_POWER_DBM + dB_stack) if is_gain else (TX_POWER_DBM - dB_stack)
    pr_mw = np.power(10.0, pr_dbm / 10.0, dtype=np.float32)
    noise_mw = np.power(10.0, (noise_dbm(B_HZ)) / 10.0, dtype=np.float32)
    interf_mw = INTERF_COEFF * (pr_mw.sum(axis=-1, keepdims=True) - pr_mw)
    sinr = pr_mw / (interf_mw + noise_mw + SAFETY_EPS)
    rate = B_HZ * np.log2(1.0 + np.maximum(sinr, 0.0))
    return pr_mw, sinr, rate


def pseudo_los_from_dB(dB_stack: np.ndarray, model_dir: str) -> np.ndarray:
    pr_mw, _, _ = dB_stack_to_pr_rate(dB_stack, model_dir)
    num = np.max(pr_mw, axis=-1)
    den = np.sum(pr_mw, axis=-1) + SAFETY_EPS
    return (num / den).astype(np.float32)


# ===== ✅ 新增：从 pr_mw 构造 3D per-BS LOS/conf（H,W,K）=====
def los3d_from_pr_mw(pr_mw: np.ndarray) -> np.ndarray:
    """
    pr_mw: [H,W,K] 线性功率
    输出:  [H,W,K]，每个像素对每个基站的相对“可置信度”
    采用：pr / max(pr)，使 strongest BS≈1，其它<1，适合做 ua-rate 的 conf 权重。
    """
    mx = np.max(pr_mw, axis=-1, keepdims=True) + SAFETY_EPS
    conf3d = pr_mw / mx
    conf3d = np.clip(conf3d, 0.0, 1.0).astype(np.float32)
    return conf3d


# ========================== 直接加载：global/backbone 与 local head（严格） ==========================
@torch.no_grad()
def load_global_filtered(model: nn.Module, ckpt_state: dict,
                         allow_prefixes=("backbone.", "extractors."),
                         deny_prefixes=("head.",)):
    ckpt_state = remap_known_prefixes(ckpt_state)

    ms = model.state_dict()
    used, skipped = 0, 0
    for k, v in ckpt_state.items():
        if any(str(k).startswith(d) for d in deny_prefixes):
            skipped += 1
            continue
        if allow_prefixes and (not any(str(k).startswith(p) for p in allow_prefixes)):
            skipped += 1
            continue
        if k in ms and ms[k].shape == v.shape:
            ms[k].copy_(v)
            used += 1
        else:
            skipped += 1
    model.load_state_dict(ms, strict=True)
    return used, skipped


def infer_k_from_head_state(head_state: dict) -> Optional[int]:
    for k in ("head.3.weight", "head.1.weight", "head.2.weight", "head.0.weight"):
        if k in head_state and hasattr(head_state[k], "shape"):
            return int(head_state[k].shape[0])
    return None


@torch.no_grad()
def load_local_head_strict(model: nn.Module, head_path: str,
                           expected_k: Optional[int] = None,
                           trust_ckpt: bool = True) -> Tuple[int, int, Optional[int], Optional[dict]]:
    ckpt = torch_load_any(head_path, map_location="cpu", trust_ckpt=trust_ckpt, prefer_weights_only=True)
    meta = None
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        meta = ckpt.get("meta", None)
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    sd = extract_state_dict(sd) if not all(torch.is_tensor(v) for v in sd.values()) else sd
    sd = remap_known_prefixes(sd)

    head_state = {k: v for k, v in sd.items() if str(k).startswith("head.")}
    if len(head_state) == 0:
        tmp = {("head." + k[len("net."):]): v for k, v in sd.items() if str(k).startswith("net.")}
        head_state = tmp

    k_head = infer_k_from_head_state(head_state)

    if expected_k is not None and k_head is not None and int(k_head) != int(expected_k):
        return 0, len(head_state), k_head, meta

    if k_head is not None and hasattr(model, "update_head_for_k"):
        try:
            model.update_head_for_k(int(k_head))
        except Exception as e:
            log(f"[WARN] update_head_for_k({k_head}) failed: {e}")

    ms = model.state_dict()
    used, skipped = 0, 0
    for k, v in head_state.items():
        if k in ms and ms[k].shape == v.shape:
            ms[k].copy_(v)
            used += 1
        else:
            skipped += 1

    if skipped > 0 or used == 0:
        return 0, len(head_state), k_head, meta

    model.load_state_dict(ms, strict=True)
    return used, skipped, k_head, meta


# ========================== 策略（可视化） ==========================
def strategy_weights(sinr_vec: np.ndarray, rate_vec: np.ndarray, los_stack: Optional[np.ndarray], mode: str):
    H, W, K = rate_vec.shape
    if mode == "max-sinr":
        idx = np.argmax(sinr_vec, axis=-1)
        w = np.zeros_like(rate_vec, dtype=np.float32)
        for k in range(K):
            w[..., k] = (idx == k).astype(np.float32)
        return w

    if mode == "prop-rate-alpha":
        base = np.power(np.maximum(rate_vec, 0.0) + SAFETY_EPS, ALPHA)
        denom = base.sum(axis=-1, keepdims=True) + SAFETY_EPS
        return base / denom

    if mode == "ua-rate-alpha-beta":
        base = np.power(np.maximum(rate_vec, 0.0) + SAFETY_EPS, ALPHA)
        if los_stack is not None and los_stack.shape[-1] == K:
            los_f = los_stack.astype(np.float32)
            conf = los_f + (1.0 - los_f) * CONF_NLOS
        else:
            conf = CONF_NLOS * np.ones_like(base, dtype=np.float32)
        base *= np.power(conf, BETA)
        denom = base.sum(axis=-1, keepdims=True) + SAFETY_EPS
        return base / denom

    raise ValueError(f"Unknown strategy: {mode}")


# ========================== map 模式：PL ==========================
def predict_pl_scene_map(model: PFL_REMNet,
                         city_id: int, model_dir: str, geom_row: dict,
                         tx_coords_df: pd.DataFrame, root_path: str,
                         device: torch.device,
                         prep_out_dir: str,
                         norm_dir: str,
                         multibs_row: pd.Series,
                         local_head_dir: str,
                         out_dir_scene: str,
                         trust_ckpt: bool = True,
                         save_visuals: bool = True,
                         also_save_strategies: bool = True,
                         los_dir_from_step1: Optional[str] = None,
                         export_extra: bool = True) -> bool:
    ref_png = pjoin(root_path, geom_row.get("buildings_png", ""))
    if not (ref_png and os.path.exists(ref_png)):
        log(f"[Skip] buildings_png missing for city {city_id}")
        return False
    W, H = Image.open(ref_png).size

    norm_path, y_mean_np, y_std_np = load_pl_norm(norm_dir, city_id, model_dir)
    if norm_path is None:
        log(f"[Skip] norm params not found for {city_id}/{model_dir}")
        return False
    K = int(y_mean_np.shape[0])
    log(f"[Norm] use norm_params | K={K} | norm_path={norm_path}")

    tx_ids = extract_tx_ids_from_multibs_row(multibs_row)
    if len(tx_ids) == 0:
        log(f"[WARN] no tx_ids parsed from multibs row for {city_id}_{model_dir}; fallback tx_coords order")
    if len(tx_ids) > 0 and len(tx_ids) != K:
        log(f"[WARN] tx_ids(len={len(tx_ids)}) != K(norm={K}) for {city_id}_{model_dir}. will truncate/pad to K")
        if len(tx_ids) > K:
            tx_ids = tx_ids[:K]
        else:
            tx_ids = tx_ids + [tx_ids[-1]] * (K - len(tx_ids))

    model.update_head_for_k(K)

    head_path = pjoin(local_head_dir, f"user_{city_id}_{model_dir}.pth")
    if not os.path.exists(head_path):
        log(f"[Skip] local head missing: {head_path}")
        return False

    used, skipped, k_head, head_meta = load_local_head_strict(model, head_path, expected_k=None, trust_ckpt=trust_ckpt)
    if used == 0:
        log(f"[Skip] local head NOT usable (used=0). file={head_path}")
        return False
    if isinstance(head_meta, dict) and ("y_mean" in head_meta) and ("y_std" in head_meta):
        y_mean_np = np.asarray(head_meta["y_mean"], dtype=np.float32).reshape(-1)
        y_std_np = np.asarray(head_meta["y_std"], dtype=np.float32).reshape(-1)

    K = int(getattr(model, "current_k", K))
    if y_mean_np.shape[0] != K:
        if y_mean_np.shape[0] > K:
            y_mean_np = y_mean_np[:K]
            y_std_np = y_std_np[:K]
        else:
            pad = K - y_mean_np.shape[0]
            y_mean_np = np.pad(y_mean_np, (0, pad), constant_values=0.0)
            y_std_np = np.pad(y_std_np, (0, pad), constant_values=1.0)
    if len(tx_ids) > 0 and len(tx_ids) != K:
        tx_ids = tx_ids[:K] if len(tx_ids) > K else (tx_ids + [tx_ids[-1]] * (K - len(tx_ids)))

    log(f"  -> local head loaded (strict): {head_path} (used={used}, skipped={skipped}, K={K})")

    X = create_feature_matrix(city_id, geom_row, tx_coords_df, (H, W), root_path, tx_ids=tx_ids)
    X_t = torch.from_numpy(X).float().to(device)

    y_mean = torch.from_numpy(y_mean_np).float().to(device)
    y_std = torch.from_numpy(y_std_np).float().to(device)

    model.eval()
    with torch.no_grad():
        pred_n = model(X_t)

    dB_stack = ((pred_n * y_std) + y_mean).detach().cpu().numpy().reshape(H, W, K).astype(np.float32)

    ensure_dir(out_dir_scene)
    out_npy = pjoin(out_dir_scene, f"{city_id}_{model_dir}_pred_dB.npy")
    np.save(out_npy, dB_stack)
    log(f"  -> dB stack saved: {out_npy} shape={dB_stack.shape}")

    # 已替换：保存为 pdf
    if save_visuals:
        for k in range(K):
            save_quick_pdf(dB_stack[..., k], pjoin(out_dir_scene, f"{city_id}_{model_dir}_tx{k:02d}.pdf"))

    # 2D pseudoLOS
    los_proxy = pseudo_los_from_dB(dB_stack, model_dir)  # [H,W]
    pseudo_los_npy = pjoin(out_dir_scene, f"{city_id}_{model_dir}_pseudoLOS.npy")
    np.save(pseudo_los_npy, los_proxy)
    if save_visuals:
        save_quick_pdf(los_proxy, pjoin(out_dir_scene, f"{city_id}_{model_dir}_pseudoLOS.pdf"), vmin=0.0, vmax=1.0)

    # ===== ✅ 新增：3D per-BS LOS/conf =====
    pr_mw, sinr, rate = dB_stack_to_pr_rate(dB_stack, model_dir)
    los3d = los3d_from_pr_mw(pr_mw)  # [H,W,K]
    los3d_npy = pjoin(out_dir_scene, f"{city_id}_{model_dir}_LOS.npy")
    np.save(los3d_npy, los3d)
    if save_visuals:
        los3d_vis = np.max(los3d, axis=-1).astype(np.float32)
        save_quick_pdf(los3d_vis, pjoin(out_dir_scene, f"{city_id}_{model_dir}_LOS.pdf"), vmin=0.0, vmax=1.0)

    los_stack_for_strat: Optional[np.ndarray] = None
    if los_dir_from_step1:
        cand = pjoin(los_dir_from_step1, model_dir, f"{city_id}_LOS.npy")
        if os.path.exists(cand):
            try:
                los_ref = np.load(cand).astype(np.float32)
                los_ref_vis = None
                if los_ref.ndim == 3 and los_ref.shape[:2] == (H, W) and los_ref.shape[-1] == K:
                    los_stack_for_strat = los_ref
                    los_ref_vis = np.clip(los_ref.max(axis=-1), 0, 1).astype(np.float32)
                elif los_ref.ndim == 2 and los_ref.shape == (H, W):
                    los_stack_for_strat = np.repeat(los_ref[..., None], K, axis=-1)
                    los_ref_vis = np.clip(los_ref, 0, 1).astype(np.float32)
                if los_ref_vis is not None and save_visuals:
                    save_quick_pdf(los_ref_vis, pjoin(out_dir_scene, f"{city_id}_{model_dir}_LOS_ref.pdf"), vmin=0.0,
                                   vmax=1.0)
            except Exception as e:
                log(f"[WARN] load LOS ref failed: {e}")

    if los_stack_for_strat is None and (los3d.ndim == 3 and los3d.shape[-1] == K):
        los_stack_for_strat = los3d

    if export_extra:
        np.save(pjoin(out_dir_scene, f"{city_id}_{model_dir}_pr_mw.npy"), pr_mw.astype(np.float32))
        np.save(pjoin(out_dir_scene, f"{city_id}_{model_dir}_sinr.npy"), sinr.astype(np.float32))
        np.save(pjoin(out_dir_scene, f"{city_id}_{model_dir}_rate.npy"), rate.astype(np.float32))
        if save_visuals:
            top1 = np.argmax(sinr, axis=-1).astype(np.float32)
            denom = max(1, K - 1)
            save_quick_pdf(top1 / denom, pjoin(out_dir_scene, f"{city_id}_{model_dir}_top1_idx.pdf"), vmin=0.0,
                           vmax=1.0)

    if also_save_strategies and save_visuals:
        for strat in ["max-sinr", "prop-rate-alpha", "ua-rate-alpha-beta"]:
            los_stack = None
            if strat == "ua-rate-alpha-beta":
                los_stack = los_stack_for_strat if los_stack_for_strat is not None else np.repeat(los_proxy[..., None],
                                                                                                  K, axis=-1)
            Wk = strategy_weights(sinr, rate, los_stack, strat)
            idx = np.argmax(Wk, axis=-1).astype(np.float32) / max(1, Wk.shape[-1] - 1)
            save_quick_pdf(idx, pjoin(out_dir_scene, f"{city_id}_{model_dir}_strategy_{strat}.pdf"), vmin=0.0, vmax=1.0)

    with open(pjoin(out_dir_scene, f"{city_id}_{model_dir}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "city_id": city_id,
            "model_dir": model_dir,
            "H": H, "W": W, "K": K,
            "norm_path": norm_path,
            "local_head": head_path,
            "files": {
                "dB_stack": out_npy,
                "pseudoLOS_2D": pseudo_los_npy,
                "LOS_3D": los3d_npy,
                "pr_mw": pjoin(out_dir_scene, f"{city_id}_{model_dir}_pr_mw.npy"),
                "sinr": pjoin(out_dir_scene, f"{city_id}_{model_dir}_sinr.npy"),
                "rate": pjoin(out_dir_scene, f"{city_id}_{model_dir}_rate.npy"),
            }
        }, f, ensure_ascii=False, indent=2)

    return True


# ========================== map 模式：KPI ==========================
def predict_kpi_scene_map(model: PFL_KPIPredictor,
                          city_id: int, model_dir: str, geom_row: dict,
                          tx_coords_df: pd.DataFrame, root_path: str,
                          device: torch.device, prep_out_dir: str,
                          multibs_row: pd.Series,
                          local_head_dir: str,
                          out_dir_scene: str,
                          trust_ckpt: bool = True) -> bool:
    ref_png = pjoin(root_path, geom_row.get("buildings_png", ""))
    if not (ref_png and os.path.exists(ref_png)):
        log(f"[Skip] buildings_png missing for city {city_id}")
        return False
    W, H = Image.open(ref_png).size

    tx_ids = extract_tx_ids_from_multibs_row(multibs_row)
    if len(tx_ids) == 0:
        log(f"[WARN] no tx_ids parsed for {city_id}_{model_dir}; fallback tx_coords order")

    head_path = pjoin(local_head_dir, f"user_{city_id}_{model_dir}_kpi.pth")
    if not os.path.exists(head_path):
        log(f"[Skip] local head missing: {head_path}")
        return False
    used, skipped, _, _ = load_local_head_strict(model, head_path, expected_k=KPI_NUM_OUTPUTS, trust_ckpt=trust_ckpt)
    if used == 0:
        log(f"[Skip] local head NOT usable: {head_path}")
        return False
    log(f"  -> local head loaded (strict): {head_path} (used={used}, skipped={skipped})")

    X = create_feature_matrix(city_id, geom_row, tx_coords_df, (H, W), root_path, tx_ids=tx_ids)
    X_t = torch.from_numpy(X).float().to(device)

    model.eval()
    with torch.no_grad():
        kpi_norm = model(X_t).squeeze(0).detach().cpu().numpy().astype(np.float32)  # [15]

    y_mean_kpi, y_std_kpi = load_kpi_stats_from_users(prep_out_dir, city_id, model_dir)
    if y_mean_kpi is not None and y_std_kpi is not None and y_mean_kpi.shape[0] == kpi_norm.shape[0]:
        kpi_phys = (kpi_norm * y_std_kpi) + y_mean_kpi
    else:
        kpi_phys = kpi_norm
        log(f"[KPI] city={city_id} model={model_dir} 未能反推 mean/std，pred_kpi 将保持标准化空间。")

    ensure_dir(out_dir_scene)
    out_npy_norm = pjoin(out_dir_scene, f"{city_id}_{model_dir}_pred_kpi_norm.npy")
    out_npy = pjoin(out_dir_scene, f"{city_id}_{model_dir}_pred_kpi.npy")

    np.save(out_npy_norm, kpi_norm)
    np.save(out_npy, kpi_phys)

    with open(pjoin(out_dir_scene, f"{city_id}_{model_dir}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "city_id": city_id,
            "model_dir": model_dir,
            "kpi_dim": int(kpi_norm.shape[0]),
            "local_head": head_path,
            "kpi_norm_file": out_npy_norm,
            "kpi_file": out_npy,
        }, f, ensure_ascii=False, indent=2)

    log(f"  -> KPI (norm/phys) saved: {out_npy_norm}, {out_npy} shape={kpi_norm.shape}")
    return True


# ========================== samples 模式（源版能力整合） ==========================
def compute_mean_std_from_train(Ytr: np.ndarray, eps: float = 1e-6):
    mean = Ytr.mean(axis=0)
    std = Ytr.std(axis=0)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


def auto_choose_domain(pred_raw: np.ndarray, y_gt: np.ndarray, mean: np.ndarray, std: np.ndarray):
    err_raw = pred_raw - y_gt
    mae_raw = float(np.mean(np.abs(err_raw)))
    rmse_raw = float(np.sqrt(np.mean(err_raw ** 2)))

    pred_den = pred_raw * std + mean
    err_den = pred_den - y_gt
    mae_den = float(np.mean(np.abs(err_den)))
    rmse_den = float(np.sqrt(np.mean(err_den ** 2)))

    if mae_den < mae_raw:
        return pred_den, True, mae_den, rmse_den, pred_raw
    return pred_raw, False, mae_raw, rmse_raw, pred_den


# ---- 兜底：从 ckpt 重建 MLP（只重建 Linear 串），来自源版 ----
def detect_linear_stack_prefix(sd: dict, want_root: str):
    keys = list(sd.keys())
    candidates = [
        f"{want_root}.",
        f"{want_root}.{want_root}.",
        f"{want_root}.net.",
        f"module.{want_root}.",
        f"module.{want_root}.net.",
    ]

    best = None
    best_cnt = -1
    best_maxidx = -1

    for pref in candidates:
        idxs = set()
        for k in keys:
            m = re.match(re.escape(pref) + r"(\d+)\.(weight|bias)$", k)
            if m:
                idxs.add(int(m.group(1)))
        if len(idxs) > best_cnt:
            best_cnt = len(idxs)
            best_maxidx = max(idxs) if idxs else -1
            best = pref

    if best_cnt <= 0:
        idxs = set()
        pref_cnt = {}
        for k in keys:
            m = re.match(rf"(.*{re.escape(want_root)}.*?)(\d+)\.(weight|bias)$", k)
            if m:
                pref = m.group(1)
                idxs.add(int(m.group(2)))
                pref_cnt[pref] = pref_cnt.get(pref, 0) + 1
        if pref_cnt:
            best = max(pref_cnt.items(), key=lambda x: x[1])[0]
            best_cnt = pref_cnt[best]
            best_maxidx = max(idxs) if idxs else -1

    return best, best_cnt, best_maxidx


def rebuild_mlp_from_state_dict(sd: dict, prefix: str):
    idx_to_w = {}
    for k, v in sd.items():
        m = re.match(re.escape(prefix) + r"(\d+)\.weight$", k)
        if not m:
            continue
        idx = int(m.group(1))
        if torch.is_tensor(v):
            idx_to_w[idx] = v

    if not idx_to_w:
        raise RuntimeError(f"Cannot rebuild: no keys like '{prefix}#.weight' found.")

    linear_idxs = []
    skipped = []
    for idx in sorted(idx_to_w.keys()):
        w = idx_to_w[idx]
        if w.ndim == 2:
            linear_idxs.append(idx)
        else:
            skipped.append((idx, tuple(w.shape)))

    if len(linear_idxs) == 0:
        msg = f"No 2D linear weights under prefix='{prefix}'. Found non-linear sample: {skipped[:10]}"
        raise RuntimeError(msg)

    linears = []
    for idx in linear_idxs:
        w = sd[f"{prefix}{idx}.weight"]
        b = sd.get(f"{prefix}{idx}.bias", None)
        out_dim, in_dim = w.shape
        lin = nn.Linear(in_dim, out_dim, bias=(b is not None))
        linears.append((idx, lin))

    layers = []
    for i, (idx, lin) in enumerate(linears):
        layers.append(lin)
        if i != len(linears) - 1:
            layers.append(nn.ReLU(inplace=True))

    mlp = nn.Sequential(*layers)

    with torch.no_grad():
        for idx, lin in linears:
            lin.weight.copy_(sd[f"{prefix}{idx}.weight"])
            if lin.bias is not None:
                lin.bias.copy_(sd[f"{prefix}{idx}.bias"])

    return mlp, max(linear_idxs)


class MLPModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model_from_ckpts_fallback(global_ckpt_path: str, local_head_path: str, device: str, trust_ckpt: bool):
    g_obj = torch_load_any(global_ckpt_path, map_location=device, trust_ckpt=trust_ckpt, prefer_weights_only=True)
    g_sd = remap_known_prefixes(extract_state_dict(g_obj))

    h_obj = torch_load_any(local_head_path, map_location=device, trust_ckpt=trust_ckpt, prefer_weights_only=True)
    h_sd = remap_known_prefixes(extract_state_dict(h_obj))

    b_pref, b_cnt, b_max = detect_linear_stack_prefix(g_sd, "backbone")
    if b_cnt <= 0 or b_pref is None:
        raise RuntimeError(f"Cannot rebuild backbone from ckpt: {global_ckpt_path}")
    backbone, _ = rebuild_mlp_from_state_dict(g_sd, b_pref)

    h_pref, h_cnt, h_max = detect_linear_stack_prefix(h_sd, "head")
    if h_cnt <= 0 or h_pref is None:
        h_pref, h_cnt, h_max = detect_linear_stack_prefix(h_sd, "net")
    if h_cnt <= 0 or h_pref is None:
        raise RuntimeError(f"Cannot rebuild head from ckpt: {local_head_path}")
    head, _ = rebuild_mlp_from_state_dict(h_sd, h_pref)

    model = MLPModel(backbone, head).to(device)
    model.eval()
    log(f"[Fallback-Rebuild] backbone prefix='{b_pref}'  head prefix='{h_pref}'")
    return model


def try_direct_model_samples(task: str, input_dim: int, out_dim: int,
                             global_ckpt: str, local_head: str,
                             device: torch.device, trust_ckpt: bool):
    if task == "pl":
        m = PFL_REMNet(input_dim=input_dim, initial_k=out_dim, two_layer_head=True, head_dropout=0.10).to(device)
        g_obj = torch_load_any(global_ckpt, map_location="cpu", trust_ckpt=trust_ckpt, prefer_weights_only=True)
        g_sd = remap_known_prefixes(extract_state_dict(g_obj))
        used, skipped = load_global_filtered(m, g_sd, allow_prefixes=("backbone.", "extractors."),
                                             deny_prefixes=("head.",))
        m.update_head_for_k(out_dim)
        u2, s2, _, _ = load_local_head_strict(m, local_head, expected_k=None, trust_ckpt=trust_ckpt)
        if used < 4 or u2 == 0:
            return None
        m.eval()
        return m

    if task == "kpi":
        m = PFL_KPIPredictor(input_dim=input_dim, hidden_dim=384, dropout=0.1207568, out_dim=out_dim).to(device)
        g_obj = torch_load_any(global_ckpt, map_location="cpu", trust_ckpt=trust_ckpt, prefer_weights_only=True)
        g_sd = remap_known_prefixes(extract_state_dict(g_obj))
        used, skipped = load_global_filtered(m, g_sd, allow_prefixes=("backbone.",), deny_prefixes=("head.",))
        u2, s2, _, _ = load_local_head_strict(m, local_head, expected_k=out_dim, trust_ckpt=trust_ckpt)
        if used < 4 or u2 == 0:
            return None
        m.eval()
        return m

    return None


# ===== 下面 main / samples / locate_manifest 等保持你原逻辑不变 =====

# ========================== manifests 定位（map 模式用，更鲁棒） ==========================
def locate_dataset_prep_root(prep_dir_hint: str, root_path_hint: str):
    checked = []

    def check(base: str):
        if not base:
            return None
        mani = pjoin(base, "manifests")
        multibs = pjoin(mani, "multibs_manifest.csv")
        geom = pjoin(mani, "geom_manifest.csv")
        tx = pjoin(mani, "tx_coords.csv")
        checked.append(mani)
        if os.path.exists(multibs) and os.path.exists(geom) and os.path.exists(tx):
            norm_dir = pjoin(base, "normalization_params")
            los_root = pjoin(base, "los")
            return base, mani, multibs, geom, tx, norm_dir, los_root
        return None

    res = check(prep_dir_hint)
    if res:
        return res, checked

    return None, checked


def read_manifest_pairs(csv_path: str):
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                city_id = int(r["city_id"])
                model_dir = str(r["model_dir"])
            except Exception:
                continue
            pairs.append((city_id, model_dir))
    seen = set()
    uniq = []
    for p in pairs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/predict_pl.yaml")
    ap.add_argument("--task", type=str, default="pl", choices=TASK_CHOICES)
    ap.add_argument("--mode", type=str, default="map", choices=MODE_CHOICES)

    ap.add_argument("--root_path", type=str, default=ROOT_PATH_DEFAULT)
    ap.add_argument("--prep_out_dir", type=str, default=DATASET_PREP_OUTPUT_DIR_DEFAULT)

    ap.add_argument("--model_ckpt", type=str, default="")
    ap.add_argument("--local_head_dir", type=str, default="")
    ap.add_argument("--outputs_dir", type=str, default="")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--save_visuals", action="store_true", default=True)
    ap.add_argument("--save_strategies", action="store_true", default=True)
    ap.add_argument("--export_extra", action="store_true", default=True)

    ap.add_argument("--city_filter", type=str, default="")
    ap.add_argument("--model_filter", type=str, default="")

    ap.add_argument("--do_kpi", action="store_true", default=False)
    ap.add_argument("--do_pl", action="store_true", default=False)

    ap.add_argument("--trust_ckpt", action="store_true", default=True,
                    help="本地训练出的 ckpt：允许 weights_only 失败后回退 weights_only=False（默认开启）")

    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--conf_nlos", type=float, default=CONF_NLOS)

    return parse_args_with_config(ap)


def main():
    global ALPHA, BETA, CONF_NLOS
    args = parse_args()
    ALPHA = args.alpha
    BETA = args.beta
    CONF_NLOS = args.conf_nlos
    device = torch.device(args.device)
    log(f"[Config] task={args.task} mode={args.mode}")
    log(f"[Config] device={device} trust_ckpt={args.trust_ckpt}")

    # samples 模式（原样保留）
    if args.mode == "samples":
        do_kpi = args.do_kpi or ((not args.do_kpi) and (not args.do_pl))
        do_pl = args.do_pl or ((not args.do_kpi) and (not args.do_pl))

        mani_csv = pjoin(args.prep_out_dir, "manifests", "multibs_manifest.csv")
        if not os.path.exists(mani_csv):
            located, _ = locate_dataset_prep_root(args.prep_out_dir, args.root_path)
            if located is not None:
                prep_dir, mani_dir, multibs_csv, geom_csv, tx_csv, norm_dir, los_root = located
                mani_csv = multibs_csv
                args.prep_out_dir = prep_dir

        if not os.path.exists(mani_csv):
            raise FileNotFoundError(f"[samples] cannot find multibs_manifest.csv under: {args.prep_out_dir}")

        pairs = read_manifest_pairs(mani_csv)

        if args.city_filter.strip():
            keep = set(int(x) for x in args.city_filter.split(",") if x.strip().isdigit())
            pairs = [p for p in pairs if p[0] in keep]
        if args.model_filter.strip():
            keepm = set(x.strip() for x in args.model_filter.split(",") if x.strip())
            pairs = [p for p in pairs if p[1] in keepm]

        out_kpi = args.outputs_dir or OUTPUTS_PRED_DIR_DEFAULT_KPI
        out_pl = args.outputs_dir or OUTPUTS_PRED_DIR_DEFAULT_PL
        ensure_dir(out_kpi);
        ensure_dir(out_pl)

        kpi_global = args.model_ckpt or MODEL_CKPT_DEFAULT_KPI
        pl_global = args.model_ckpt or MODEL_CKPT_DEFAULT_PL
        kpi_head_dir = args.local_head_dir or LOCAL_HEAD_DIR_DEFAULT_KPI
        pl_head_dir = args.local_head_dir or LOCAL_HEAD_DIR_DEFAULT_PL

        sum_rows = []
        for (city_id, model_dir) in pairs:
            if do_kpi:
                ok = run_kpi_one_samples(args.prep_out_dir, out_kpi, kpi_global, kpi_head_dir, city_id, model_dir,
                                         device, args.trust_ckpt)
                sum_rows.append({"mode": "samples", "task": "kpi", "city_id": city_id, "model_dir": model_dir,
                                 "status": "ok" if ok else "skip"})
            if do_pl:
                ok = run_pl_one_samples(args.prep_out_dir, out_pl, pl_global, pl_head_dir, city_id, model_dir, device,
                                        args.trust_ckpt)
                sum_rows.append({"mode": "samples", "task": "pl", "city_id": city_id, "model_dir": model_dir,
                                 "status": "ok" if ok else "skip"})

        out_sum = pjoin(args.outputs_dir or os.path.dirname(out_kpi), "summary_samples.csv")
        pd.DataFrame(sum_rows).to_csv(out_sum, index=False, encoding="utf-8-sig")
        log(f"[DONE] samples summary -> {out_sum}")
        return

    # map 模式（原样保留）
    located, checked = locate_dataset_prep_root(args.prep_out_dir, args.root_path)
    if located is None:
        msg = (
                "[ERROR] 找不到 manifests（multibs_manifest.csv / geom_manifest.csv / tx_coords.csv）。\n"
                f"你当前给的 prep_out_dir: {args.prep_out_dir}\n"
                "我检查过以下路径下的 manifests：\n  - " + "\n  - ".join(checked[:8]) + (
                    "\n  - ..." if len(checked) > 8 else "") + "\n\n"
                                                               "解决方法：\n"
                                                               "1) 确认你运行过 dataset_preparation.py，并生成了 dataset_preparation_output/manifests\n"
                                                               "2) 把 --prep_out_dir 指向那个包含 manifests 的 dataset_preparation_output 目录。"
        )
        raise FileNotFoundError(msg)

    prep_dir, mani_dir, multibs_csv, geom_csv, tx_csv, norm_dir, los_root = located
    log(f"[Manifests] use prep_dir={prep_dir}")

    multibs_df = pd.read_csv(multibs_csv)
    geom_map = load_geom_map(geom_csv)
    tx_df = pd.read_csv(tx_csv)

    if args.city_filter.strip():
        keep = set(int(x) for x in args.city_filter.split(",") if x.strip().isdigit())
        multibs_df = multibs_df[multibs_df["city_id"].isin(list(keep))]
    if args.model_filter.strip():
        keepm = set(x.strip() for x in args.model_filter.split(",") if x.strip())
        multibs_df = multibs_df[multibs_df["model_dir"].isin(list(keepm))]

    out_root = args.outputs_dir or (OUTPUTS_PRED_DIR_DEFAULT_PL if args.task == "pl" else OUTPUTS_PRED_DIR_DEFAULT_KPI)
    ensure_dir(out_root)

    model_ckpt = args.model_ckpt or (MODEL_CKPT_DEFAULT_PL if args.task == "pl" else MODEL_CKPT_DEFAULT_KPI)
    local_head_dir = args.local_head_dir or (
        LOCAL_HEAD_DIR_DEFAULT_PL if args.task == "pl" else LOCAL_HEAD_DIR_DEFAULT_KPI)

    log(f"[Load] ckpt: {model_ckpt}")
    log(f"[LocalHead] dir: {local_head_dir}")

    if args.task == "pl":
        model = PFL_REMNet(input_dim=INPUT_DIM_DEFAULT, initial_k=32, two_layer_head=True, head_dropout=0.10).to(device)
    else:
        model = PFL_KPIPredictor(input_dim=INPUT_DIM_DEFAULT, hidden_dim=384, dropout=0.1207568,
                                 out_dim=KPI_NUM_OUTPUTS).to(device)

    sd_obj = torch_load_any(model_ckpt, map_location="cpu", trust_ckpt=args.trust_ckpt, prefer_weights_only=True)
    sd = extract_state_dict(sd_obj)
    used, skipped = load_global_filtered(model, sd,
                                         allow_prefixes=("backbone.", "extractors.") if args.task == "pl" else (
                                         "backbone.",), deny_prefixes=("head.",))
    log(f"[OK] global weights loaded (filtered): used={used}, skipped={skipped} | deny=head.*")

    los_root = pjoin(prep_dir, "los")
    sum_rows = []
    for _, row in multibs_df.iterrows():
        city = int(row["city_id"])
        mdir = str(row["model_dir"])
        geom_row = geom_map.get(city, None)
        if geom_row is None:
            log(f"[Skip] no geom for city {city}")
            sum_rows.append({"city_id": city, "model_dir": mdir, "status": "skip", "reason": "no_geom"})
            continue

        out_scene_dir = pjoin(out_root, f"{city}_{mdir}")
        ensure_dir(out_scene_dir)
        log(f"[Run] {city}_{mdir} ...")

        if args.task == "pl":
            ok = predict_pl_scene_map(model, city, mdir, geom_row, tx_df, args.root_path, device,
                                      prep_dir, norm_dir, row, local_head_dir, out_scene_dir,
                                      trust_ckpt=args.trust_ckpt,
                                      save_visuals=args.save_visuals,
                                      also_save_strategies=args.save_strategies,
                                      los_dir_from_step1=los_root,
                                      export_extra=args.export_extra)
        else:
            ok = predict_kpi_scene_map(model, city, mdir, geom_row, tx_df, args.root_path, device,
                                       prep_dir, row, local_head_dir, out_scene_dir,
                                       trust_ckpt=args.trust_ckpt)

        sum_rows.append(
            {"mode": "map", "task": args.task, "city_id": city, "model_dir": mdir, "status": "ok" if ok else "skip",
             "out_dir": out_scene_dir})

    sum_csv = pjoin(out_root, f"summary_{args.task}_{args.mode}.csv")
    pd.DataFrame(sum_rows).to_csv(sum_csv, index=False, encoding="utf-8-sig")
    log(f"[DONE] Summary -> {sum_csv}")


if __name__ == "__main__":
    main()
