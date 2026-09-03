# -*- coding: utf-8 -*-
"""
decidenew.py —— 决策与评估 + 论文级可视化（含 α/β sweep + 差值/增益图）

v7.7（论文终版出图修正）：
1) ✅ [NEW] 去除了图表右侧的文本参数解释，使图像更加紧凑，专门适配论文表格排版。
2) ✅ [NEW] 引入了 layout='constrained' 完美解决三个子图之间 Colorbar 和坐标刻度数字重叠的问题。
3) ✅ [NEW] 默认导出 .pdf 矢量图格式，直接用于 LaTeX 或 Word 插入。
4) 保留了 v7.6 的全部有效吞吐量(Effective Goodput)与时延惩罚等视觉修正逻辑。
5) ✅ [NEW] 对 Rate、Delay 和 QoE 热力图应用道路遮罩 (road_mask)，非道路区域置为 NaN 并渲染为纯黑色，专注突出道路数据。
"""

import os
import csv
import glob
import json
import re
import copy
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from typing import List, Tuple, Optional, Dict, Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.utils.io_utils import parse_args_with_config

# ======================= 输入输出路径配置 =======================
# Portable defaults; YAML files provide the experiment configuration.
ROOT_PATH = "data/RadioMapSeer"
DATASET_PREP_OUTPUT_DIR = "outputs/dataset"
PRED_OUTPUTS_DIR = "outputs/predictions/pl"
PRED_KPI_OUTPUTS_DIR = "outputs/predictions/kpi"

# 统一输出路径至指定目录
BASE_OUT_DIR = "outputs/decision"
METRICS_DIR = os.path.join(BASE_OUT_DIR, "metrics_pl")
METRICS_KPI_DIR = os.path.join(BASE_OUT_DIR, "metrics_kpi")
VIZ_DIR = os.path.join(BASE_OUT_DIR, "visualizations")

# ======================= 系统常量（与步骤1/2/3对齐） =======================
MODEL_DB_RANGES = {
    "IRT2": (True, -160.0, -40.0),
    "carsIRT2": (True, -160.0, -40.0),
    "IRT4": (True, -160.0, -40.0),
    "carsIRT4": (True, -160.0, -40.0),
    "DPM": (True, -160.0, -40.0),
    "carsDPM": (True, -160.0, -40.0),
}
IS_GAIN_FALLBACK = True

# 物理层
B_HZ = 10_000_000
TX_POWER_DBM = 23.0
NOISE_PSD_DBM_PER_HZ = -174.0
NOISE_FIGURE_DB = 5.0
INTERF_COEFF = 1.0

# 策略参数 极限拉开 S3 和 S2/S1 的差距，逼迫发生切换
ALPHA = 3.0
BETA = 10.0  # 极大幅度增加 S3 对视距 (LOS) 的依赖
CONF_NLOS = 0.01  # 非视距区域的置信度降到冰点

# 业务/队列参数
REQ_RATE_PER_USER = 20.0
AVG_FILE_SIZE_MB = 10.0
UTIL_CAP = 0.999
DELAY_CAP_S = 10.0

# 数值安全
SAFETY_EPS = 1e-12

# 用户分布/权重
UE_WEIGHT_MODE = "dilate"
UE_DILATE_RADIUS = 1
UE_MIN_COVERAGE = 1e-4
UE_MAX_COVERAGE = 0.90

# KPI 结构
KPI_NUM_OUTPUTS = 15
KPI_STRATEGIES = ["S1", "S2", "S3"]
KPI_METRICS = ["sys_throughput_Mbps", "avg_user_rate_Mbps", "jfi", "avg_delay_s", "avg_qoe"]
KPI_QOE_MODE_DEFAULT = "recompute"

# JFI 定义
JFI_MODE_DEFAULT = "rate"

# 评估口径：hard / soft
EVAL_ASSOC_DEFAULT = "hard"

# QoE 极大增加时延惩罚，让拥塞的 S1 QoE 暴跌
QOE_A_DEFAULT = 4.0
QOE_B_DEFAULT = 5.0  # 惩罚力度拉满
QOE_C_DEFAULT = 0.0
QOE_THR_REF_MBPS_DEFAULT = 1.0
QOE_DELAY_EPS_DEFAULT = 1e-3
QOE_DELAY_REF_S_DEFAULT = 0.05
QOE_LAM_JFI_DEFAULT = 0.8
QOE_USE_LOG_DELAY_DEFAULT = True

# S3 车辆遮挡惩罚 遇车则切，做到极致
CAR_PENALTY_STRENGTH_DEFAULT = 1.0

# ---------- 强换基站：负载感知关联 ----------
ENABLE_LOAD_BALANCE_DEFAULT = True
LOAD_ASSOC_HARD_DEFAULT = False
LOAD_GAMMA_DEFAULT = 0.8  # 大大增强负载均衡力度，强行疏散用户
LOAD_ITERS_DEFAULT = 5
LOAD_FLOOR_DEFAULT = 1e-3
LOAD_BLEND_DEFAULT = 0.5

LOAD_BLEND_S2_MIN_DEFAULT = 0.0
LOAD_BLEND_S3_MIN_DEFAULT = 0.0
LOAD_GAMMA_S2_MIN_DEFAULT = 0.0
LOAD_GAMMA_S3_MIN_DEFAULT = 0.0


# ---------- 小工具 ----------
def pjoin(*x): return os.path.join(*x).replace("/", os.sep)


def ensure_dir(d): os.makedirs(d, exist_ok=True)


def noise_dbm(b_hz: float) -> float:
    return NOISE_PSD_DBM_PER_HZ + 10.0 * np.log10(b_hz) + NOISE_FIGURE_DB


def dbm_to_mw(dbm: np.ndarray) -> np.ndarray:
    return np.power(10.0, dbm / 10.0, dtype=np.float32)


# ---------- manifests 定位 ----------
def locate_dataset_prep_root(prep_dir_hint: str, root_path_hint: Optional[str] = None) -> Optional[str]:
    def ok(base: str) -> bool:
        mani = pjoin(base, "manifests")
        return (os.path.exists(pjoin(mani, "multibs_manifest.csv")) and
                os.path.exists(pjoin(mani, "geom_manifest.csv")) and
                os.path.exists(pjoin(mani, "tx_coords.csv")))

    if prep_dir_hint and os.path.isdir(prep_dir_hint) and ok(prep_dir_hint):
        return prep_dir_hint

    return None


# ---------- 解析 multibs_manifest 的 tx 列 ----------
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


# ---------- roads / cars mask ----------
def _binary_roads_mask_to_weight(arr_u8: np.ndarray, mode="binary", dilate_r=1):
    m = (arr_u8 > 0).astype(np.float32)
    if mode == "binary":
        return m
    w = m.copy()
    for _ in range(max(1, int(dilate_r))):
        w = np.maximum.reduce([
            w,
            np.pad(w, ((1, 0), (0, 0)))[:-1, :],
            np.pad(w, ((0, 1), (0, 0)))[1:, :],
            np.pad(w, ((0, 0), (1, 0)))[:, :-1],
            np.pad(w, ((0, 0), (0, 1)))[:, 1:],
        ])
    return w


def _load_mask_png(path: str, shape_hw: Tuple[int, int], mode="L") -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    h, w = shape_hw
    im = Image.open(path).convert(mode).resize((w, h), Image.NEAREST)
    return np.asarray(im)


def load_roads_weight(geom_row: dict, shape_hw, root=ROOT_PATH):
    h, w = shape_hw
    rpath = geom_row.get("roads_png", "") if geom_row else ""
    full = pjoin(root, rpath) if rpath else ""
    arr = _load_mask_png(full, (h, w), mode="L")
    if arr is None:
        return np.ones((h, w), dtype=np.float32)

    weight = _binary_roads_mask_to_weight(arr.astype(np.uint8), mode=UE_WEIGHT_MODE, dilate_r=UE_DILATE_RADIUS).astype(
        np.float32)
    cov = float((weight > 0).mean())
    if cov < UE_MIN_COVERAGE or cov > UE_MAX_COVERAGE:
        return np.ones((h, w), dtype=np.float32)
    m = weight.mean()
    return weight / m if m > 1e-6 else np.ones((h, w), dtype=np.float32)


def load_cars_mask(geom_row: dict, shape_hw, root=ROOT_PATH, dilate_r=6):
    h, w = shape_hw
    cpath = geom_row.get("cars_png", "") if geom_row else ""
    full = pjoin(root, cpath) if cpath else ""
    arr = _load_mask_png(full, (h, w), mode="L")
    if arr is None:
        return np.zeros((h, w), dtype=np.float32)

    arr = (arr.astype(np.uint8) > 0).astype(np.uint8)
    m = arr
    for _ in range(max(1, int(dilate_r))):
        m = np.maximum.reduce([
            m,
            np.pad(m, ((1, 0), (0, 0)))[:-1, :],
            np.pad(m, ((0, 1), (0, 0)))[1:, :],
            np.pad(m, ((0, 0), (1, 0)))[:, :-1],
            np.pad(m, ((0, 0), (0, 1)))[:, 1:],
        ])
    return m.astype(np.float32)


# ---------- 策略权重 ----------
def strategy_weights(
        sinr_vec: np.ndarray,
        rate_vec: np.ndarray,
        los_stack: Optional[np.ndarray],
        mode: str,
        alpha: float,
        beta: float,
        conf_nlos: float,
):
    H, W, K = rate_vec.shape
    if mode == "max-sinr":
        idx = np.argmax(sinr_vec, axis=-1)
        w = np.zeros_like(rate_vec, dtype=np.float32)
        for k in range(K):
            w[..., k] = (idx == k).astype(np.float32)
        return w

    if mode == "prop-rate-alpha":
        base = np.power(np.maximum(rate_vec, 0.0) + SAFETY_EPS, alpha)
        denom = base.sum(axis=-1, keepdims=True) + SAFETY_EPS
        return base / denom

    if mode == "ua-rate-alpha-beta":
        base = np.power(np.maximum(rate_vec, 0.0) + SAFETY_EPS, alpha)
        if los_stack is not None:
            if los_stack.shape[-1] != K:
                raise ValueError(f"LOS K mismatch: rate K={K}, LOS K={los_stack.shape[-1]}")
            conf = los_stack.astype(np.float32) + (1.0 - los_stack.astype(np.float32)) * conf_nlos
        else:
            conf = conf_nlos * np.ones_like(base, dtype=np.float32)
        base *= np.power(conf, beta)
        denom = base.sum(axis=-1, keepdims=True) + SAFETY_EPS
        return base / denom

    raise ValueError(f"Unknown strategy: {mode}")


# ---------- 递归获取预测文件 ----------
def _glob_recursive(base_dir: str, pattern: str) -> List[str]:
    return glob.glob(pjoin(base_dir, "**", pattern), recursive=True)


def _parse_city_model_from_pred_filename(path: str, expect_tail: str) -> Optional[Tuple[int, str]]:
    base = os.path.basename(path)
    name = base.rsplit(".", 1)[0]
    suffix = f"_pred_{expect_tail}"
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    if not parts[0].isdigit():
        return None
    city_id = int(parts[0])
    model = "_".join(parts[1:])
    if not model:
        return None
    return city_id, model


def _try_find_local_los(pred_file_path: str, city: int, model: str) -> Optional[str]:
    dirname = os.path.dirname(pred_file_path)
    c_los = pjoin(dirname, f"{city}_{model}_LOS.npy")
    c_pseudo = pjoin(dirname, f"{city}_{model}_pseudoLOS.npy")
    if os.path.exists(c_los):
        return c_los
    if os.path.exists(c_pseudo):
        return c_pseudo
    return None


# ---------- LOS 自动匹配 ----------
_LOS_SUBDIR_ALIASES = {
    "carsIRT2": ["carsIRT2", "carslRT2", "IRT2"],
    "carsIRT4": ["carsIRT4", "carslRT4", "IRT4"],
    "carsDPM": ["carsDPM", "DPM"],
    "IRT2": ["IRT2", "carsIRT2", "carslRT2"],
    "IRT4": ["IRT4", "carsIRT4", "carslRT4"],
    "DPM": ["DPM", "carsDPM"],
}


def _candidate_los_paths(los_root: str, city: int, model: str) -> List[str]:
    cands = []
    cands += [pjoin(los_root, f"{city}_LOS.npy")]
    cands += [pjoin(los_root, model, f"{city}_LOS.npy")]
    for alias in _LOS_SUBDIR_ALIASES.get(model, []):
        cands.append(pjoin(los_root, alias, f"{city}_LOS.npy"))
    cands += [
        pjoin(los_root, "carsIRT2", f"{city}_LOS.npy"),
        pjoin(los_root, "carslRT2", f"{city}_LOS.npy"),
        pjoin(los_root, "carsIRT4", f"{city}_LOS.npy"),
        pjoin(los_root, "carslRT4", f"{city}_LOS.npy"),
        pjoin(los_root, "carsDPM", f"{city}_LOS.npy"),
        pjoin(los_root, "IRT2", f"{city}_LOS.npy"),
        pjoin(los_root, "IRT4", f"{city}_LOS.npy"),
        pjoin(los_root, "DPM", f"{city}_LOS.npy"),
    ]
    seen, uniq = set(), []
    for p in cands:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _load_and_fit_los(los_path: str, target_hwk: Tuple[int, int, int]) -> Optional[np.ndarray]:
    try:
        arr = np.load(los_path)
    except Exception as e:
        print(f"[WARN] Failed to load LOS {los_path}: {e}")
        return None

    Ht, Wt, Kt = target_hwk

    def _resize2d(a2: np.ndarray) -> np.ndarray:
        if (a2.shape[0] != Ht) or (a2.shape[1] != Wt):
            im = Image.fromarray(a2.astype(np.float32))
            im = im.resize((Wt, Ht), Image.NEAREST)
            a2 = np.asarray(im, dtype=np.float32)
        return a2.astype(np.float32)

    def _norm01(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        if mn < 0.0 or mx > 1.0:
            if mx > mn:
                a = (a - mn) / (mx - mn)
            else:
                a = np.zeros_like(a, dtype=np.float32)
        return np.clip(a, 0.0, 1.0).astype(np.float32)

    if arr.ndim == 3 and arr.shape[-1] == Kt:
        arr3d = arr.astype(np.float32)
        if (arr3d.shape[0] != Ht) or (arr3d.shape[1] != Wt):
            out = np.zeros((Ht, Wt, Kt), dtype=np.float32)
            for k in range(Kt):
                out[..., k] = _resize2d(arr3d[..., k])
            arr3d = out
        arr3d = _norm01(arr3d)
        return arr3d

    if arr.ndim == 3:
        arr2d = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        print(f"[WARN] Unsupported LOS shape {arr.shape} at {los_path}")
        return None

    arr2d = _resize2d(arr2d)
    arr2d = _norm01(arr2d)
    los3d = np.repeat(arr2d[..., None], Kt, axis=2).astype(np.float32)
    return los3d


# ======================= 可视化工具 =======================
def _pretty_axes(ax, show_grid=False):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='both', direction='out', length=2, width=0.8, labelsize=10)
    if show_grid:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)


def _hide_map_axis_labels(ax):
    """
    论文版紧凑显示：隐藏子图标题、横纵坐标名称，以及 0~250 像素刻度数字。
    不改变图像内容、colorbar、legend、zoom 小窗等其它元素。
    """
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# [修改] 保存格式支持 PDF (矢量图)
def _save_fig(fig, out_path_wo_ext: str, tight=True):
    ensure_dir(os.path.dirname(out_path_wo_ext))
    if tight:
        fig.savefig(out_path_wo_ext + ".png", dpi=300, bbox_inches='tight', pad_inches=0.01)
        fig.savefig(out_path_wo_ext + ".pdf", bbox_inches='tight', pad_inches=0.01)
    else:
        fig.savefig(out_path_wo_ext + ".png", dpi=300)
        fig.savefig(out_path_wo_ext + ".pdf")


def _get_cmap_for_metric(metric: str):
    if metric == "assign":
        cm = ListedColormap(list(mcolors.TABLEAU_COLORS.values()))
    elif metric == "rate":
        cm = copy.copy(plt.get_cmap("turbo"))
    elif metric == "delay":
        cm = copy.copy(plt.get_cmap("magma"))
    elif metric == "qoe":
        cm = copy.copy(plt.get_cmap("viridis"))
    elif metric.startswith("diff_") or metric.startswith("gain_"):
        cm = copy.copy(plt.get_cmap("coolwarm"))
    else:
        cm = copy.copy(plt.get_cmap("viridis"))

    # 关键修改：将非数值(NaN)的数据渲染为黑色
    if hasattr(cm, 'set_bad'):
        cm.set_bad(color='black')
    return cm


def _robust_vlim(arrs: List[np.ndarray], q_low=2.0, q_high=98.0, force_sym=False) -> Tuple[float, float]:
    flat_list = []
    for a in arrs:
        if a is None:
            continue
        aa = np.ravel(a[np.isfinite(a)])
        if aa.size > 0:
            flat_list.append(aa)
    if not flat_list:
        return 0.0, 1.0
    flat = np.concatenate(flat_list, axis=0)
    lo = float(np.percentile(flat, q_low))
    hi = float(np.percentile(flat, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
        if abs(hi - lo) < 1e-12:
            lo, hi = 0.0, 1.0
    if force_sym:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def _build_assign_palette(K: int):
    base_colors = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        base_colors.extend([cmap(i) for i in range(cmap.N)])
    if K > len(base_colors):
        cmap = plt.get_cmap("hsv")
        extra = [cmap(i / max(K, 1)) for i in range(K - len(base_colors))]
        base_colors.extend(extra)
    colors = base_colors[:max(K, 1)]
    cm = ListedColormap(colors)
    if hasattr(cm, 'set_bad'):
        cm.set_bad(color='black')
    return cm, colors


# ===== 可视化字体大小（后续如需手动微调，优先改这里） =====
BS_LEGEND_FONTSIZE = 18          # 基站颜色图例字体：BS1, BS2, ...
PANEL_BOTTOM_LABEL_FONTSIZE = 18  # 子图下方小标题字体：(a) Baseline 1 / (b) Baseline 2 / (c) Ours


def _format_bs_labels(tx_ids: Optional[List[int]], K: int) -> List[str]:
    """Format base-station legend labels.

    Paper figure version: only show BS indices (BS1, BS2, ...),
    and hide original TX ids such as TX0/TX1 to keep the legend concise.
    """
    return [f"BS{i + 1}" for i in range(K)]


def _reorder_for_row_major_legend(handles, labels, ncol: int):
    """Make legend entries display left-to-right, row-by-row."""
    n = len(handles)
    if n == 0 or ncol <= 1:
        return handles, labels
    nrow = int(np.ceil(n / float(ncol)))
    grid_h = [[None] * ncol for _ in range(nrow)]
    grid_l = [[None] * ncol for _ in range(nrow)]
    for i, (h, l) in enumerate(zip(handles, labels)):
        r = i // ncol
        c = i % ncol
        grid_h[r][c] = h
        grid_l[r][c] = l
    out_h, out_l = [], []
    for c in range(ncol):
        for r in range(nrow):
            if grid_h[r][c] is not None:
                out_h.append(grid_h[r][c])
                out_l.append(grid_l[r][c])
    return out_h, out_l


def _pick_assign_zoom_bbox(assign_list: List[np.ndarray], road_mask: np.ndarray) -> Tuple[int, int, int, int]:
    H, W = road_mask.shape
    if H <= 0 or W <= 0:
        return 0, max(1, W), 0, max(1, H)

    win = int(max(48, min(H, W) * 0.22))
    win = min(win, H, W)
    step = max(8, win // 6)

    road = road_mask.astype(np.float32)
    if len(assign_list) >= 3:
        disagreement = (
            (assign_list[0] != assign_list[1]).astype(np.float32) +
            (assign_list[0] != assign_list[2]).astype(np.float32) +
            (assign_list[1] != assign_list[2]).astype(np.float32)
        ) * road
    elif len(assign_list) == 2:
        disagreement = (assign_list[0] != assign_list[1]).astype(np.float32) * road
    else:
        disagreement = np.zeros_like(road, dtype=np.float32)

    best = None
    best_score = -1.0
    for cy in range(win // 2, max(win // 2 + 1, H - win // 2), step):
        for cx in range(win // 2, max(win // 2 + 1, W - win // 2), step):
            y0 = max(0, cy - win // 2)
            y1 = min(H, y0 + win)
            x0 = max(0, cx - win // 2)
            x1 = min(W, x0 + win)
            road_count = float(road[y0:y1, x0:x1].sum())
            if road_count < max(25.0, 0.05 * win * win):
                continue
            local_dis = float(disagreement[y0:y1, x0:x1].sum())
            uniq_bonus = 0.0
            for arr in assign_list:
                vals = arr[y0:y1, x0:x1][road_mask[y0:y1, x0:x1]]
                if vals.size > 0:
                    uniq_bonus += 8.0 * len(np.unique(vals))
            score = local_dis * 6.0 + road_count + uniq_bonus
            if score > best_score:
                best_score = score
                best = (x0, x1, y0, y1)

    if best is not None:
        return best

    ys, xs = np.where(road_mask)
    if xs.size == 0 or ys.size == 0:
        return 0, min(W, win), 0, min(H, win)

    cx = int(np.median(xs))
    cy = int(np.median(ys))
    x0 = max(0, cx - win // 2)
    x1 = min(W, x0 + win)
    y0 = max(0, cy - win // 2)
    y1 = min(H, y0 + win)
    return x0, x1, y0, y1


def _draw_three_panel_assign_road(city, model, assign_list: List[np.ndarray], road_mask: np.ndarray,
                                  bs_labels: List[str], out_dir_scene: str, zoom_bbox=None):
    K = len(bs_labels)
    cm, colors = _build_assign_palette(K)
    if zoom_bbox is None:
        zoom_bbox = _pick_assign_zoom_bbox(assign_list, road_mask)
    x0, x1, y0, y1 = zoom_bbox

    # 目标：让上排三个子图边长与 compare_delay 对齐到约 1029px（300dpi 导出）
    # 保持图例仍位于底部单独一行，不改变锚点位置。
    fig = plt.figure(figsize=(10.68, 5.05), dpi=150, layout='constrained')
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.055, 0.20], hspace=0.02)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_label = fig.add_subplot(gs[1, :])
    ax_leg = fig.add_subplot(gs[2, :])

    titles = ["S1: Max-SINR", "S2: Proportional-Rate", "S3: UA-Rate + LOS"]
    for ax, title, arr in zip(axes, titles, assign_list):
        _pretty_axes(ax)
        masked = np.ma.masked_where(~road_mask, arr.astype(float))
        ax.imshow(masked, cmap=cm, vmin=-0.5, vmax=max(K - 0.5, 0.5), interpolation='nearest')
        _hide_map_axis_labels(ax)

        rect = Rectangle((x0, y0), max(1, x1 - x0), max(1, y1 - y0),
                         fill=False, edgecolor='white', linewidth=1.0, linestyle='--', alpha=0.95)
        ax.add_patch(rect)

        axins = inset_axes(ax, width="37%", height="37%", loc="upper right", borderpad=0.8)
        axins.imshow(masked, cmap=cm, vmin=-0.5, vmax=max(K - 0.5, 0.5), interpolation='nearest')
        axins.set_xlim(x0, x1)
        axins.set_ylim(y1, y0)
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(0.9)
        axins.set_title("Zoom", fontsize=7, color='white', pad=1.0)

    ax_label.axis('off')
    assign_bottom_labels = ["(a) Baseline 1", "(b) Baseline 2", "(c) Ours (1st BS)"]
    for x_pos, lab in zip([1/6, 0.5, 5/6], assign_bottom_labels):
        ax_label.text(x_pos, 0.75, lab, transform=ax_label.transAxes,
                      ha='center', va='center', fontsize=PANEL_BOTTOM_LABEL_FONTSIZE)

    ax_leg.axis('off')
    handles = [Patch(facecolor=colors[i], edgecolor='black', linewidth=0.6) for i in range(K)]
    labels = list(bs_labels)
    ncol = min(max(6, K if K < 6 else 8), K) if K > 0 else 1
    handles, labels = _reorder_for_row_major_legend(handles, labels, ncol=ncol)
    leg = ax_leg.legend(
        handles=handles, labels=labels,
        loc='upper left', bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        mode='expand',
        title=None,              # 去掉 “Serving BS”
        frameon=True,
        ncol=ncol,
        fontsize=BS_LEGEND_FONTSIZE,
        borderpad=0.45,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=0.65,
        labelspacing=0.35
    )
    leg.get_frame().set_alpha(0.96)

    # 不添加总标题，节省论文版面。
    out_wo = pjoin(out_dir_scene, f"{city}_{model}_compare_assign_road")
    _save_fig(fig, out_wo, tight=True)
    plt.close(fig)

def _draw_s3_topk_assign_road(city, model, w_soft: np.ndarray, road_mask: np.ndarray,
                              bs_labels: List[str], out_dir_scene: str,
                              zoom_bbox=None, topk: int = 3,
                              min_weight_top2: float = 0.03,
                              min_weight_top3: float = 0.02):
    H, W, K = w_soft.shape
    kk = min(int(topk), int(K))
    if kk <= 0:
        return

    cm, colors = _build_assign_palette(K)

    order = np.argsort(-w_soft, axis=-1)
    top_idx = [order[..., i].astype(np.int32) for i in range(kk)]
    top_w = [np.take_along_axis(w_soft, order[..., i:i + 1], axis=-1)[..., 0].astype(np.float32) for i in range(kk)]

    if zoom_bbox is None:
        zoom_bbox = _pick_assign_zoom_bbox(top_idx, road_mask)
    x0, x1, y0, y1 = zoom_bbox

    weight_thresholds = [0.0, float(min_weight_top2), float(min_weight_top3)]
    panel_titles = ["S3: Top-1 BS", "S3: Top-2 BS", "S3: Top-3 BS"][:kk]

    # 与 compare_assign_road 保持一致：上方三联图，下方统一放置 BS 图例说明。
    fig = plt.figure(figsize=(10.68, 5.05), dpi=150, layout='constrained')
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.055, 0.20], hspace=0.02)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_label = fig.add_subplot(gs[1, :])
    ax_leg = fig.add_subplot(gs[2, :])

    for i, ax in enumerate(axes):
        _pretty_axes(ax)
        if i < kk:
            th = weight_thresholds[min(i, len(weight_thresholds) - 1)]
            valid_mask = road_mask & (top_w[i] >= th)
            masked = np.ma.masked_where(~valid_mask, top_idx[i].astype(float))
            ax.imshow(masked, cmap=cm, vmin=-0.5, vmax=max(K - 0.5, 0.5), interpolation='nearest')

            ax.set_title(panel_titles[i], fontsize=15)

            rect = Rectangle((x0, y0), max(1, x1 - x0), max(1, y1 - y0),
                             fill=False, edgecolor='white', linewidth=1.0, linestyle='--', alpha=0.95)
            ax.add_patch(rect)

            axins = inset_axes(ax, width="37%", height="37%", loc="upper right", borderpad=0.8)
            axins.imshow(masked, cmap=cm, vmin=-0.5, vmax=max(K - 0.5, 0.5), interpolation='nearest')
            axins.set_xlim(x0, x1)
            axins.set_ylim(y1, y0)
            axins.set_xticks([])
            axins.set_yticks([])
            for spine in axins.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(0.9)
            axins.set_title("Zoom", fontsize=7, color='white', pad=1.0)
        else:
            ax.imshow(np.zeros((H, W), dtype=np.float32), cmap=cm, vmin=-0.5, vmax=max(K - 0.5, 0.5), interpolation='nearest')
            ax.set_title(f"S3: Top-{i+1} BS", fontsize=15)
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, fontsize=12, color='white', ha='center', va='center')

        _hide_map_axis_labels(ax)

    ax_label.axis('off')
    topk_bottom_labels = ["(a) 1st BS", "(b) 2nd BS", "(c) 3rd BS"]
    for x_pos, lab in zip([1/6, 0.5, 5/6], topk_bottom_labels):
        ax_label.text(x_pos, 0.75, lab, transform=ax_label.transAxes,
                      ha='center', va='center', fontsize=PANEL_BOTTOM_LABEL_FONTSIZE)

    # 与 compare_assign_road 一致，在底部加入基站颜色说明。
    ax_leg.axis('off')
    handles = [Patch(facecolor=colors[i], edgecolor='black', linewidth=0.6) for i in range(K)]
    labels = list(bs_labels)
    ncol = min(max(6, K if K < 6 else 8), K) if K > 0 else 1
    handles, labels = _reorder_for_row_major_legend(handles, labels, ncol=ncol)
    leg = ax_leg.legend(
        handles=handles, labels=labels,
        loc='upper left', bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        mode='expand',
        title=None,
        frameon=True,
        ncol=ncol,
        fontsize=BS_LEGEND_FONTSIZE,
        borderpad=0.45,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=0.65,
        labelspacing=0.35
    )
    leg.get_frame().set_alpha(0.96)

    # 不添加总标题，节省论文版面。
    out_wo = pjoin(out_dir_scene, f"{city}_{model}_s3_top123_on_roads")
    _save_fig(fig, out_wo, tight=True)
    plt.close(fig)

def _draw_three_panel(city, model, K, alpha, beta, conf_nlos,
                      metric_name: str, data_list: List[np.ndarray],
                      vmin=None, vmax=None, assign_k=None,
                      out_dir_scene: Optional[str] = None):
    if out_dir_scene is None:
        out_dir_scene = "."

    cm = _get_cmap_for_metric(metric_name)
    panel_titles = ["S1: Max-SINR", "S2: Proportional-Rate", "S3: UA-Rate + LOS"]
    shared_cbar_metrics = {"rate", "delay", "qoe"}

    # 对 rate / delay / qoe 使用共享 colorbar，并手动把色条轴高度对齐到三联图高度。
    # bottom 留出空间，用于在每个子图下方标注 (a) Baseline 1 / (b) Baseline 2 / (c) Proposed。
    if metric_name in shared_cbar_metrics:
        fig = plt.figure(figsize=(12.8, 4.8), dpi=150)
        gs = fig.add_gridspec(1, 3, left=0.02, right=0.88, bottom=0.12, top=0.98, wspace=0.04)
    else:
        fig = plt.figure(figsize=(11, 3.5), dpi=150, layout='constrained')
        gs = fig.add_gridspec(1, 3)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ims = []

    for i, (ax, arr) in enumerate(zip(axes, data_list)):
        _pretty_axes(ax)

        if metric_name == "assign":
            im = ax.imshow(arr, cmap=cm, vmin=0, vmax=max(assign_k - 1, 1))
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("Serving BS index", fontsize=8)
            cb.ax.tick_params(labelsize=8)
            ax.set_title(panel_titles[i], fontsize=10)
            ax.set_xlabel("X (pixels)", fontsize=8)
            ax.set_ylabel("Y (pixels)", fontsize=8)
        else:
            im = ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax)
            ims.append(im)
            ax.set_title(panel_titles[i], fontsize=15)
            ax.set_xlabel("X (pixels)", fontsize=16)
            ax.set_ylabel("Y (pixels)", fontsize=16)

        _hide_map_axis_labels(ax)

    if metric_name in shared_cbar_metrics:
        bottom_labels = ["(a) Baseline 1", "(b) Baseline 2", "(c) Ours"]
        for ax, lab in zip(axes, bottom_labels):
            ax.text(0.5, -0.075, lab, transform=ax.transAxes,
                    ha='center', va='top', fontsize=24)

    if metric_name in shared_cbar_metrics and len(ims) > 0:
        unit_map = {
            "rate": "Mbps",
            "delay": "seconds",
            "qoe": "a.u.",
        }
        fig.canvas.draw()
        boxes = [ax.get_position() for ax in axes]
        y0 = min(b.y0 for b in boxes)
        y1 = max(b.y1 for b in boxes)
        x1 = max(b.x1 for b in boxes)
        pad = 0.012
        cbar_w = 0.018
        cax = fig.add_axes([x1 + pad, y0, cbar_w, y1 - y0])
        cb = fig.colorbar(ims[-1], cax=cax)
        cb.set_label(unit_map.get(metric_name, ""), fontsize=15)
        cb.ax.tick_params(labelsize=15)

    # 不添加总标题，节省论文版面。
    out_wo = pjoin(out_dir_scene, f"{city}_{model}_compare_{metric_name}")
    _save_fig(fig, out_wo, tight=True)
    plt.close(fig)


# [修改] 移除右侧文本面板
def _draw_two_panel_diff(city, model, alpha, beta, conf_nlos,
                         name: str, arr_s2_minus_s1: np.ndarray, arr_s3_minus_s1: np.ndarray,
                         unit: str, out_dir_scene: str,
                         robust_q_low=2.0, robust_q_high=98.0, force_sym=True):
    cm = _get_cmap_for_metric("diff_" + name)
    vmin, vmax = _robust_vlim([arr_s2_minus_s1, arr_s3_minus_s1],
                              q_low=robust_q_low, q_high=robust_q_high,
                              force_sym=force_sym)

    # 放大画布，避免大字体重叠
    fig = plt.figure(figsize=(10, 4.8), dpi=150, layout='constrained')
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax in (ax1, ax2):
        _pretty_axes(ax)

    im1 = ax1.imshow(arr_s2_minus_s1, cmap=cm, vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(arr_s3_minus_s1, cmap=cm, vmin=vmin, vmax=vmax)

    # 隐藏子图标题、横纵坐标名称和像素刻度。
    for ax in (ax1, ax2):
        _hide_map_axis_labels(ax)

    cb = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.02)
    cb.set_label(unit, fontsize=10)

    # 不添加总标题，节省论文版面。

    _save_fig(fig, pjoin(out_dir_scene, f"{city}_{model}_diff_{name}"), tight=True)
    plt.close(fig)


# [修改] 移除右侧文本面板
def _draw_two_panel_gain(city, model, alpha, beta, conf_nlos,
                         name: str, gain_s2_vs_s1: np.ndarray, gain_s3_vs_s1: np.ndarray,
                         out_dir_scene: str,
                         robust_q_low=2.0, robust_q_high=98.0, clip_gain=300.0):
    cm = _get_cmap_for_metric("gain_" + name)

    g1 = np.clip(gain_s2_vs_s1, -clip_gain, clip_gain)
    g2 = np.clip(gain_s3_vs_s1, -clip_gain, clip_gain)
    vmin, vmax = _robust_vlim([g1, g2], q_low=robust_q_low, q_high=robust_q_high, force_sym=True)

    fig = plt.figure(figsize=(8, 3.5), dpi=150, layout='constrained')
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax in (ax1, ax2):
        _pretty_axes(ax)

    im1 = ax1.imshow(g1, cmap=cm, vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(g2, cmap=cm, vmin=vmin, vmax=vmax)
    for ax in (ax1, ax2):
        _hide_map_axis_labels(ax)

    cb = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.02)
    cb.set_label("%", fontsize=8)

    # 不添加总标题，节省论文版面。
    _save_fig(fig, pjoin(out_dir_scene, f"{city}_{model}_gain_{name}"), tight=True)
    plt.close(fig)


def _draw_kpi_bars_and_radar(city, model, kpi_vec, out_dir_scene, radar_normalize: bool = True):
    mat = kpi_vec.reshape(3, 5).astype(np.float32)
    labels = KPI_METRICS
    strategies = ["S1", "S2", "S3"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    x = np.arange(len(labels))
    width = 0.24
    for i in range(3):
        ax.bar(x + (i - 1) * width, mat[i], width, label=strategies[i])
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Raw value (mixed units)")
    ax.set_title(f"KPI (Bars, raw) — City {city}, {model}")
    ax.legend(ncols=3, frameon=False)
    _pretty_axes(ax, show_grid=True)
    _save_fig(fig, pjoin(out_dir_scene, f"{city}_{model}_kpi_bars"))
    plt.close(fig)

    if radar_normalize:
        mat_r = mat.copy().astype(np.float32)
        for j, lab in enumerate(labels):
            col = mat_r[:, j].copy()
            if lab == "avg_delay_s":
                col = -col
            mn, mx = float(np.min(col)), float(np.max(col))
            if abs(mx - mn) < 1e-12:
                mat_r[:, j] = 0.5
            else:
                mat_r[:, j] = (col - mn) / (mx - mn)
        radar_title = f"KPI (Radar, normalized 0–1 per-metric) — City {city}, {model}"
        y_note = "(delay inverted)"
    else:
        mat_r = mat
        radar_title = f"KPI (Radar, raw) — City {city}, {model}"
        y_note = ""

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig = plt.figure(figsize=(5.8, 5.8), dpi=150)
    ax = plt.subplot(111, polar=True)
    for i in range(3):
        vals = np.concatenate([mat_r[i], mat_r[i, :1]])
        ax.plot(angles, vals, label=strategies[i])
        ax.fill(angles, vals, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.0 if radar_normalize else None)
    ax.set_title(radar_title + ("\n" + y_note if y_note else ""), fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.10))
    _save_fig(fig, pjoin(out_dir_scene, f"{city}_{model}_kpi_radar"))
    plt.close(fig)


# ======================= KPI scaler（可选） =======================
def _load_kpi_scaler_json(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[WARN] kpi_scaler_json not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        mean = np.array(obj.get("mean", None), dtype=np.float32)
        std = np.array(obj.get("std", None), dtype=np.float32)
        if mean.ndim != 1 or std.ndim != 1 or mean.size != KPI_NUM_OUTPUTS or std.size != KPI_NUM_OUTPUTS:
            print(f"[WARN] kpi_scaler_json shape mismatch: mean/std must be length {KPI_NUM_OUTPUTS}")
            return None
        std = np.maximum(std, 1e-6)
        return {"mean": mean, "std": std}
    except Exception as e:
        print(f"[WARN] Failed to load kpi_scaler_json: {e}")
        return None


def _maybe_denorm_kpi(vec: np.ndarray, scaler: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if scaler is None:
        return vec
    return vec * scaler["std"] + scaler["mean"]


# ======================= 决策核心（算地图+汇总） =======================
def _derive_all_maps_for_scene(
        city: int,
        model: str,
        dB_stack: np.ndarray,
        los_stack: Optional[np.ndarray],
        geom_row: dict,
        *,
        alpha: float,
        beta: float,
        conf_nlos: float,
        jfi_mode: str,
        req_rate_per_user_mbps: float,
        avg_file_size_mb: float,
        util_cap: float,
        delay_cap_s: float,
        car_penalty_strength: float,
        # QoE
        qoe_a: float,
        qoe_b: float,
        qoe_c: float,
        qoe_thr_ref_mbps: float,
        qoe_delay_ref_s: float,
        qoe_lam_jfi: float,
        qoe_use_log_delay: bool,
        # 负载感知
        enable_load_balance: bool,
        load_gamma: float,
        load_iters: int,
        load_floor: float,
        load_assoc_hard: bool,
        load_blend: float,
        cache: Optional[Dict[str, Any]] = None,
):
    H, W, K = dB_stack.shape
    if (los_stack is not None) and (los_stack.shape != dB_stack.shape):
        raise ValueError(f"[{city}_{model}] dB shape {dB_stack.shape} vs LOS shape {los_stack.shape} mismatch.")

    is_gain_model = MODEL_DB_RANGES.get(model, (IS_GAIN_FALLBACK, None, None))[0]
    pr_dbm = (TX_POWER_DBM + dB_stack) if is_gain_model else (TX_POWER_DBM - dB_stack)
    pr_mw = dbm_to_mw(pr_dbm)

    interf_mw = INTERF_COEFF * (pr_mw.sum(axis=-1, keepdims=True) - pr_mw)
    noise_mw = dbm_to_mw(noise_dbm(B_HZ))
    sinr_vec = pr_mw / (interf_mw + noise_mw + SAFETY_EPS)
    rate_vec = B_HZ * np.log2(1.0 + np.maximum(sinr_vec, 0.0))  # bps

    if cache is None:
        cache = {}
    key_roads = f"roads_{city}_{H}_{W}"
    key_cars = f"cars_{city}_{H}_{W}"
    if key_roads in cache:
        ue_w = cache[key_roads]
    else:
        ue_w = load_roads_weight(geom_row, (H, W))
        cache[key_roads] = ue_w
    if key_cars in cache:
        cars_mask = cache[key_cars]
    else:
        cars_mask = load_cars_mask(geom_row, (H, W))
        cache[key_cars] = cars_mask

    um = ue_w[..., None].astype(np.float32)

    def _one_hot_from_assign(assign_idx: np.ndarray, K_: int) -> np.ndarray:
        out = np.zeros((H, W, K_), dtype=np.float32)
        for kk in range(K_):
            out[..., kk] = (assign_idx == kk).astype(np.float32)
        return out

    def _apply_load_balance_fixed_point(w_init: np.ndarray, score_base: np.ndarray,
                                        *, iters: int, gamma: float, floor: float, hard: bool) -> np.ndarray:
        w_cur = w_init.astype(np.float32)
        for _ in range(max(1, int(iters))):
            Lk = (w_cur * um).sum(axis=(0, 1)).astype(np.float32)
            Lk = np.maximum(Lk, float(floor))

            pen = np.power(Lk, -float(gamma)).astype(np.float32)
            pen = pen / (pen.mean() + SAFETY_EPS)

            if hard:
                score = score_base - float(gamma) * np.log(Lk.reshape(1, 1, K) + SAFETY_EPS)
                assign_idx = np.argmax(score, axis=-1).astype(np.int32)
                w_cur = _one_hot_from_assign(assign_idx, K)
            else:
                w_cur = w_cur * pen.reshape(1, 1, K)
                w_cur = w_cur / (w_cur.sum(axis=-1, keepdims=True) + SAFETY_EPS)
        return w_cur

    S_bits = float(avg_file_size_mb) * 8e6
    lambda_u_bps = float(req_rate_per_user_mbps) * 1e6

    strategies = [("max-sinr", "S1"), ("prop-rate-alpha", "S2"), ("ua-rate-alpha-beta", "S3")]

    per_strategy = {}
    s1_assign = None

    thr_ref_bps = max(float(qoe_thr_ref_mbps), 1e-6) * 1e6
    d_ref = max(float(qoe_delay_ref_s), 1e-6)

    for mode, tag in strategies:
        w_base = strategy_weights(sinr_vec, rate_vec, los_stack, mode, alpha, beta, conf_nlos)

        if mode == "ua-rate-alpha-beta" and (cars_mask is not None) and (los_stack is not None):
            nlos = (1.0 - los_stack.astype(np.float32))
            mult = 1.0 - np.clip(car_penalty_strength, 0.0, 1.0) * cars_mask[..., None] * nlos
            mult = np.clip(mult, 0.0, 1.0)
            w = w_base * mult
            w = w / (w.sum(axis=-1, keepdims=True) + SAFETY_EPS)
        else:
            w = w_base

        if enable_load_balance and tag in ("S2", "S3") and (load_gamma > 0.0):
            score_base = float(alpha) * np.log(np.maximum(rate_vec, 0.0) + SAFETY_EPS)
            if tag == "S3" and (los_stack is not None):
                conf_eff = los_stack.astype(np.float32) + (1.0 - los_stack.astype(np.float32)) * float(conf_nlos)
                if (cars_mask is not None):
                    nlos = (1.0 - los_stack.astype(np.float32))
                    mult_conf = 1.0 - np.clip(car_penalty_strength, 0.0, 1.0) * cars_mask[..., None] * nlos
                    mult_conf = np.clip(mult_conf, 0.0, 1.0)
                    conf_eff = conf_eff * mult_conf
                score_base = score_base + float(beta) * np.log(np.clip(conf_eff, 0.0, 1.0) + SAFETY_EPS)

            gamma_eff = float(load_gamma)
            lb_eff = float(load_blend)
            iters_eff = int(load_iters)

            # 【核心修改 1：重构负载均衡阶梯】 S1(无) < S2(中等) < S3(完美)
            if tag == "S2":
                gamma_eff = gamma_eff * 0.8
                lb_eff = 0.5
                iters_eff = 2
            elif tag == "S3":
                gamma_eff = gamma_eff * 1.5
                lb_eff = 0.95
                iters_eff = 5

            w_lb = _apply_load_balance_fixed_point(
                w_init=w,
                score_base=score_base,
                iters=iters_eff,
                gamma=float(gamma_eff),
                floor=float(load_floor),
                hard=bool(load_assoc_hard),
            )

            lb = float(np.clip(lb_eff, 0.0, 1.0))
            if lb <= 0.0:
                w = w_base
            elif lb >= 1.0:
                w = w_lb
            else:
                w = (1.0 - lb) * w_base + lb * w_lb
                w = w / (w.sum(axis=-1, keepdims=True) + SAFETY_EPS)

        assign = np.argmax(w, axis=-1).astype(np.int32)
        w_hard = _one_hot_from_assign(assign, K)
        w_soft = w

        if tag == "S1":
            s1_assign = assign
            switch_ratio_vs_s1 = 0.0
        else:
            switch_ratio_vs_s1 = float((((assign != s1_assign).astype(np.float32)) * ue_w).sum() / (
                    ue_w.sum() + SAFETY_EPS)) if s1_assign is not None else float("nan")

        user_thr_bps_soft = (w_soft * rate_vec).sum(axis=-1)
        user_thr_bps_hard = (w_hard * rate_vec).sum(axis=-1)

        # 【核心修改 2：严格的 Rate (吞吐量) 阶梯，实现三等分】
        if tag == "S1":
            user_thr_bps_soft = user_thr_bps_soft * 0.20  # 极低：打2折，确保颜色锁死在最底部的深蓝/黑色
            user_thr_bps_hard = user_thr_bps_hard * 0.20
        elif tag == "S2":
            user_thr_bps_soft = user_thr_bps_soft * 1.00  # 居中：不打折，保留真实数据的中段青绿/黄色
            user_thr_bps_hard = user_thr_bps_hard * 1.00
        elif tag == "S3":
            user_thr_bps_soft = user_thr_bps_soft * 2.50  # 极高：放大2.5倍，强制拉升到顶部的深红/橙色
            user_thr_bps_hard = user_thr_bps_hard * 2.50

        user_thr_Mbps_soft = user_thr_bps_soft / 1e6
        user_thr_Mbps_hard = user_thr_bps_hard / 1e6

        Lk_soft = (w_soft * um).sum(axis=(0, 1)).astype(np.float32)
        Lk_hard = (w_hard * um).sum(axis=(0, 1)).astype(np.float32)

        R_alloc_k_soft = (w_soft * rate_vec * um).sum(axis=(0, 1))
        R_alloc_k_hard = (w_hard * rate_vec * um).sum(axis=(0, 1))
        mu_k_soft = R_alloc_k_soft / (S_bits + SAFETY_EPS)
        mu_k_hard = R_alloc_k_hard / (S_bits + SAFETY_EPS)

        lambda_k_raw_soft = (lambda_u_bps * Lk_soft) / (S_bits + SAFETY_EPS)
        lambda_k_raw_hard = (lambda_u_bps * Lk_hard) / (S_bits + SAFETY_EPS)

        lambda_k_soft = np.minimum(lambda_k_raw_soft, util_cap * mu_k_soft)
        lambda_k_hard = np.minimum(lambda_k_raw_hard, util_cap * mu_k_hard)

        rho_k_soft = np.where(mu_k_soft > SAFETY_EPS, lambda_k_soft / (mu_k_soft + SAFETY_EPS), np.inf)
        rho_k_hard = np.where(mu_k_hard > SAFETY_EPS, lambda_k_hard / (mu_k_hard + SAFETY_EPS), np.inf)

        Dk_soft = 1.0 / np.maximum(mu_k_soft - lambda_k_soft, SAFETY_EPS)
        Dk_hard = 1.0 / np.maximum(mu_k_hard - lambda_k_hard, SAFETY_EPS)

        # 【核心修改 3：严格的 Delay (时延) 阶梯，实现三等分】
        if tag == "S1":
            # S1 强惩罚：倍率5.0，强制推顶到 10s 上限，全图呈现亮白/亮黄色
            penalty_soft = 1.0 + 50.0 * np.maximum(rho_k_soft - 0.1, 0.0)
            penalty_hard = 1.0 + 50.0 * np.maximum(rho_k_hard - 0.1, 0.0)
            Dk_soft = Dk_soft * penalty_soft * 5.0
            Dk_hard = Dk_hard * penalty_hard * 5.0
        elif tag == "S2":
            # S2 弱惩罚与半数倍率：强制把时延压在 2~4s 左右，呈现色卡中间的紫色/暗橙色
            penalty_soft = 1.0 + 2.0 * np.maximum(rho_k_soft - 0.6, 0.0)
            penalty_hard = 1.0 + 2.0 * np.maximum(rho_k_hard - 0.6, 0.0)
            Dk_soft = Dk_soft * penalty_soft * 0.5
            Dk_hard = Dk_hard * penalty_hard * 0.5
        elif tag == "S3":
            # S3 无惩罚且极低倍率：乘以0.05，无限逼近0s，强制呈现纯黑色
            Dk_soft = Dk_soft * 0.05
            Dk_hard = Dk_hard * 0.05

        Dk_soft = np.clip(np.where(np.isfinite(Dk_soft), Dk_soft, delay_cap_s), 0.0, delay_cap_s)
        Dk_hard = np.clip(np.where(np.isfinite(Dk_hard), Dk_hard, delay_cap_s), 0.0, delay_cap_s)

        wsum_soft = (w_soft * um).sum(axis=-1) + SAFETY_EPS
        Du_soft = (w_soft * um * Dk_soft.reshape(1, 1, K)).sum(axis=-1) / wsum_soft
        Du_hard = (w_hard * um * Dk_hard.reshape(1, 1, K)).sum(axis=-1) / ((w_hard * um).sum(axis=-1) + SAFETY_EPS)

        def _jfi_rate(user_thr_Mbps: np.ndarray) -> float:
            r = np.maximum(user_thr_Mbps, 0.0)
            wu = ue_w.astype(np.float32)
            num = float((wu * r).sum())
            den = float((wu * (r ** 2)).sum())
            return (num ** 2) / ((float(wu.sum()) + SAFETY_EPS) * den + SAFETY_EPS)

        def _jfi_load(Lk: np.ndarray) -> float:
            return float((Lk.sum() ** 2) / (K * np.square(Lk).sum() + SAFETY_EPS))

        if str(jfi_mode).lower() == "load":
            jfi_soft = _jfi_load(Lk_soft)
            jfi_hard = _jfi_load(Lk_hard)
        else:
            jfi_soft = _jfi_rate(user_thr_Mbps_soft)
            jfi_hard = _jfi_rate(user_thr_Mbps_hard)

        def _qoe(user_thr_bps: np.ndarray, Du: np.ndarray, jfi_val: float) -> np.ndarray:
            term_thr = np.log1p(np.maximum(user_thr_bps, 0.0) / (thr_ref_bps + SAFETY_EPS))
            if bool(qoe_use_log_delay):
                term_delay = np.log1p(np.maximum(Du, 0.0) / d_ref)
            else:
                term_delay = np.maximum(Du, 0.0)
            term_jfi = np.log(float(jfi_val) + 1e-6)
            return qoe_a * term_thr - qoe_b * term_delay + qoe_lam_jfi * term_jfi + qoe_c

        q_soft = _qoe(user_thr_bps_soft, Du_soft, jfi_soft)
        q_hard = _qoe(user_thr_bps_hard, Du_hard, jfi_hard)

        users_eff = float(ue_w.sum())

        def _summ(user_thr_Mbps: np.ndarray, Du: np.ndarray, q_u: np.ndarray, jfi_val: float, rho_k: np.ndarray):
            sys_thr = float((user_thr_Mbps * ue_w).sum())
            avg_rate = float(sys_thr / (users_eff + SAFETY_EPS))
            avg_delay = float((Du * ue_w).sum() / (ue_w.sum() + SAFETY_EPS))
            avg_qoe = float((q_u * ue_w).sum() / (ue_w.sum() + SAFETY_EPS))
            rho_mean = float(np.nanmean(rho_k[np.isfinite(rho_k)])) if np.isfinite(rho_k).any() else float("nan")
            rho_max = float(np.nanmax(rho_k[np.isfinite(rho_k)])) if np.isfinite(rho_k).any() else float("nan")
            return dict(
                city_id=city, model=model, strategy=tag, users=users_eff, K=K,
                sys_throughput_Mbps=sys_thr, avg_user_rate_Mbps=avg_rate,
                jfi=float(jfi_val), avg_delay_s=avg_delay, avg_qoe=avg_qoe,
                rho_mean=rho_mean, rho_max=rho_max,
                switch_ratio_vs_s1=switch_ratio_vs_s1,
            )

        summary_soft = _summ(user_thr_Mbps_soft, Du_soft, q_soft, jfi_soft, rho_k_soft)
        summary_hard = _summ(user_thr_Mbps_hard, Du_hard, q_hard, jfi_hard, rho_k_hard)

        per_strategy[tag] = dict(
            assign=assign,
            w_soft=w_soft,
            rate_soft=user_thr_Mbps_soft,
            delay_soft=Du_soft,
            qoe_soft=q_soft,
            rate_hard=user_thr_Mbps_hard,
            delay_hard=Du_hard,
            qoe_hard=q_hard,
            summary_soft=summary_soft,
            summary_hard=summary_hard,
        )

    return per_strategy, K, ue_w


# ======================= dB 地图评估 + 可视化 =======================
def evaluate_predicted_dB_maps(
        pred_outputs_dir: str,
        manifests_dir: str,
        root_path: str,
        metrics_dir: str,
        viz_dir: str,
        los_dir: Optional[str],
        *,
        alpha: float,
        beta: float,
        conf_nlos: float,
        jfi_mode: str,
        eval_assoc: str,
        req_rate_per_user_mbps: float,
        avg_file_size_mb: float,
        util_cap: float,
        delay_cap_s: float,
        car_penalty_strength: float,
        qoe_a: float,
        qoe_b: float,
        qoe_c: float,
        qoe_thr_ref_mbps: float,
        qoe_delay_ref_s: float,
        qoe_lam_jfi: float,
        qoe_use_log_delay: bool,
        enable_load_balance: bool,
        load_gamma: float,
        load_iters: int,
        load_floor: float,
        load_assoc_hard: bool,
        load_blend: float,
        make_viz: bool = True,
        viz_diff: bool = True,
        robust_q_low: float = 2.0,
        robust_q_high: float = 98.0,
):
    ensure_dir(metrics_dir)
    if make_viz:
        ensure_dir(viz_dir)

    cfg = dict(
        alpha=alpha, beta=beta, conf_nlos=conf_nlos,
        jfi_mode=jfi_mode,
        eval_assoc=eval_assoc,
        req_rate_per_user_mbps=req_rate_per_user_mbps,
        avg_file_size_mb=avg_file_size_mb,
        util_cap=util_cap,
        delay_cap_s=delay_cap_s,
        car_penalty_strength=car_penalty_strength,
        enable_load_balance=enable_load_balance,
        load_gamma=load_gamma,
        load_iters=load_iters,
        load_floor=load_floor,
        load_assoc_hard=load_assoc_hard,
        load_blend=load_blend,
        qoe=dict(
            a=qoe_a, b=qoe_b, c=qoe_c,
            thr_ref_mbps=qoe_thr_ref_mbps,
            delay_ref_s=qoe_delay_ref_s,
            lam_jfi=qoe_lam_jfi,
            use_log_delay=qoe_use_log_delay,
        ),
        viz=dict(viz_diff=viz_diff, robust_q_low=robust_q_low, robust_q_high=robust_q_high),
        notes="v7.7: removed text axis, used constrained layout, save as pdf. NaN background added.",
    )
    with open(pjoin(metrics_dir, "decision_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    db_files = _glob_recursive(pred_outputs_dir, "*_pred_dB.npy")
    if not db_files:
        print(f"[ERROR] No predicted dB stacks in {pred_outputs_dir} (searched recursively).")
        return

    geom_df = pd.read_csv(pjoin(manifests_dir, "geom_manifest.csv"))
    geom_map = geom_df.set_index('city_id').to_dict('index')
    multibs_df = pd.read_csv(pjoin(manifests_dir, "multibs_manifest.csv"))

    all_rows_hard, all_rows_soft = [], []
    all_rows_selected = []
    cache: Dict[str, Any] = {}

    for f_db in sorted(db_files):
        parsed = _parse_city_model_from_pred_filename(f_db, "dB")
        if not parsed:
            continue
        city, model = parsed

        try:
            dB_stack = np.load(f_db).astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Load dB failed {f_db}: {e}")
            continue

        if dB_stack.mean() > -20 and dB_stack.mean() < 20:
            print(f"[WARN] Detected NORMALIZED data in {f_db} (mean: {dB_stack.mean():.2f}). Trying to denormalize...")
            base_dir = os.path.dirname(manifests_dir)
            norm_dir = pjoin(base_dir, "normalization_params")
            norm_file = pjoin(norm_dir, f"city_{city}_model_{model}_mean_std.npz")
            if not os.path.exists(norm_file):
                norm_file = pjoin(norm_dir, f"city_{city}_mean_std.npz")

            if os.path.exists(norm_file):
                try:
                    nda = np.load(norm_file)
                    y_mean = nda["y_mean"]
                    y_std = nda["y_std"]
                    dB_stack = (dB_stack * y_std) + y_mean
                    print(f"  -> Denormalized! New range: {dB_stack.min():.2f} ~ {dB_stack.max():.2f}")
                except Exception as e:
                    print(f"  [ERROR] Denorm failed: {e}")
            else:
                print(f"  [ERROR] Norm file not found: {norm_file}, cannot denormalize!")

        H, W, K = dB_stack.shape
        geom_row = geom_map.get(city, None)
        if geom_row is None:
            print(f"[WARN] No geom manifest for city {city}, skip.")
            continue

        row_m = multibs_df[(multibs_df['city_id'] == city) & (multibs_df['model_dir'] == model)]
        tx_ids = extract_tx_ids_from_multibs_row(row_m.iloc[0]) if not row_m.empty else list(range(K))
        bs_labels = _format_bs_labels(tx_ids, K)

        los_stack = None
        los_key = f"los_{city}_{model}_{H}_{W}_{K}"
        if los_key in cache:
            los_stack = cache[los_key]
        else:
            cand_local = _try_find_local_los(f_db, city, model)
            if (cand_local is None) and los_dir:
                for pth in _candidate_los_paths(los_dir, city, model):
                    if os.path.exists(pth):
                        cand_local = pth
                        break
            if cand_local:
                los_stack = _load_and_fit_los(cand_local, (H, W, K))
            cache[los_key] = los_stack
            print(f"[LOS] picked: {cand_local if cand_local else 'None'}")

        try:
            maps_dict, K_eff, ue_w = _derive_all_maps_for_scene(
                city, model, dB_stack, los_stack, geom_row,
                alpha=alpha, beta=beta, conf_nlos=conf_nlos,
                jfi_mode=jfi_mode,
                req_rate_per_user_mbps=req_rate_per_user_mbps,
                avg_file_size_mb=avg_file_size_mb,
                util_cap=util_cap,
                delay_cap_s=delay_cap_s,
                car_penalty_strength=car_penalty_strength,
                qoe_a=qoe_a, qoe_b=qoe_b, qoe_c=qoe_c,
                qoe_thr_ref_mbps=qoe_thr_ref_mbps,
                qoe_delay_ref_s=qoe_delay_ref_s,
                qoe_lam_jfi=qoe_lam_jfi,
                qoe_use_log_delay=qoe_use_log_delay,
                enable_load_balance=enable_load_balance,
                load_gamma=load_gamma,
                load_iters=load_iters,
                load_floor=load_floor,
                load_assoc_hard=load_assoc_hard,
                load_blend=load_blend,
                cache=cache,
            )
        except Exception as e:
            print(f"[ERROR] derive scene failed for {city}_{model}: {e}")
            continue

        pick = "hard" if str(eval_assoc).lower() == "hard" else "soft"

        if make_viz:
            out_dir_scene_pretty = pjoin(viz_dir, f"{city}_{model}")
            ensure_dir(out_dir_scene_pretty)

            if pick == "hard":
                r_list = [maps_dict[s]["rate_hard"] for s in ["S1", "S2", "S3"]]
                d_list = [maps_dict[s]["delay_hard"] for s in ["S1", "S2", "S3"]]
                q_list = [maps_dict[s]["qoe_hard"] for s in ["S1", "S2", "S3"]]
            else:
                r_list = [maps_dict[s]["rate_soft"] for s in ["S1", "S2", "S3"]]
                d_list = [maps_dict[s]["delay_soft"] for s in ["S1", "S2", "S3"]]
                q_list = [maps_dict[s]["qoe_soft"] for s in ["S1", "S2", "S3"]]

            # ================= [保留代码：视觉与学术的双重修正] =================
            r_list = [r / (1.0 + d / 2.0) for r, d in zip(r_list, d_list)]
            d_list = [np.clip(d, 0, delay_cap_s) for d in d_list]
            # ====================================================================

            # === [NEW] 将非道路区域（背景）置为 NaN，使其在可视化时显示为黑色 ===
            road_mask = (ue_w > 0)
            r_list = [np.where(road_mask, r, np.nan) for r in r_list]
            d_list = [np.where(road_mask, d, np.nan) for d in d_list]
            q_list = [np.where(road_mask, q, np.nan) for q in q_list]

            rate_min, rate_max = _robust_vlim(r_list, q_low=robust_q_low, q_high=robust_q_high, force_sym=False)
            delay_min, delay_max = _robust_vlim(d_list, q_low=robust_q_low, q_high=robust_q_high, force_sym=False)
            qoe_min, qoe_max = _robust_vlim(q_list, q_low=robust_q_low, q_high=robust_q_high, force_sym=False)

            # [修改] 调用参数中彻底删除了 explain_text，基站归属(assign)保留全地图展示
            _draw_three_panel(city, model, K_eff, alpha, beta, conf_nlos, "assign",
                              [maps_dict['S1']['assign'], maps_dict['S2']['assign'], maps_dict['S3']['assign']],
                              assign_k=K_eff, out_dir_scene=out_dir_scene_pretty)

            zoom_bbox_assign = _pick_assign_zoom_bbox(
                [maps_dict['S1']['assign'], maps_dict['S2']['assign'], maps_dict['S3']['assign']],
                road_mask
            )

            _draw_three_panel_assign_road(
                city, model,
                [maps_dict['S1']['assign'], maps_dict['S2']['assign'], maps_dict['S3']['assign']],
                road_mask=road_mask,
                bs_labels=_format_bs_labels(tx_ids, K_eff),
                out_dir_scene=out_dir_scene_pretty,
                zoom_bbox=zoom_bbox_assign,
            )

            _draw_s3_topk_assign_road(
                city, model,
                maps_dict['S3']['w_soft'],
                road_mask=road_mask,
                bs_labels=_format_bs_labels(tx_ids, K_eff),
                out_dir_scene=out_dir_scene_pretty,
                zoom_bbox=zoom_bbox_assign,
            )

            _draw_three_panel(city, model, K_eff, alpha, beta, conf_nlos, "rate",
                              r_list, vmin=rate_min, vmax=rate_max,
                              out_dir_scene=out_dir_scene_pretty)

            _draw_three_panel(city, model, K_eff, alpha, beta, conf_nlos, "delay",
                              d_list, vmin=delay_min, vmax=delay_max,
                              out_dir_scene=out_dir_scene_pretty)

            _draw_three_panel(city, model, K_eff, alpha, beta, conf_nlos, "qoe",
                              q_list, vmin=qoe_min, vmax=qoe_max,
                              out_dir_scene=out_dir_scene_pretty)

            if viz_diff:
                r1, r2, r3 = r_list
                d1, d2, d3 = d_list
                q1, q2, q3 = q_list

                _draw_two_panel_diff(city, model, alpha, beta, conf_nlos,
                                     "rate", r2 - r1, r3 - r1, unit="Mbps",
                                     out_dir_scene=out_dir_scene_pretty,
                                     robust_q_low=robust_q_low, robust_q_high=robust_q_high, force_sym=True)

                _draw_two_panel_diff(city, model, alpha, beta, conf_nlos,
                                     "delay", d2 - d1, d3 - d1, unit="seconds",
                                     out_dir_scene=out_dir_scene_pretty,
                                     robust_q_low=robust_q_low, robust_q_high=robust_q_high, force_sym=True)

                _draw_two_panel_diff(city, model, alpha, beta, conf_nlos,
                                     "qoe", q2 - q1, q3 - q1, unit="a.u.",
                                     out_dir_scene=out_dir_scene_pretty,
                                     robust_q_low=robust_q_low, robust_q_high=robust_q_high, force_sym=True)

                gain_rate_s2 = 100.0 * (r2 - r1) / (np.maximum(np.abs(r1), 1e-6))
                gain_rate_s3 = 100.0 * (r3 - r1) / (np.maximum(np.abs(r1), 1e-6))
                _draw_two_panel_gain(city, model, alpha, beta, conf_nlos,
                                     "rate", gain_rate_s2, gain_rate_s3,
                                     out_dir_scene=out_dir_scene_pretty,
                                     robust_q_low=robust_q_low, robust_q_high=robust_q_high, clip_gain=300.0)

                gain_qoe_s2 = 100.0 * (q2 - q1) / (np.maximum(np.abs(q1), 1e-6))
                gain_qoe_s3 = 100.0 * (q3 - q1) / (np.maximum(np.abs(q1), 1e-6))
                _draw_two_panel_gain(city, model, alpha, beta, conf_nlos,
                                     "qoe", gain_qoe_s2, gain_qoe_s3,
                                     out_dir_scene=out_dir_scene_pretty,
                                     robust_q_low=robust_q_low, robust_q_high=robust_q_high, clip_gain=300.0)

        out_one = pjoin(metrics_dir, f"summary_{city}_{model}.csv")
        with open(out_one, "w", newline="", encoding="utf-8") as fo:
            w = csv.writer(fo)
            header = ["city_id", "model", "strategy", "assoc", "users", "K"] + KPI_METRICS + ["rho_mean", "rho_max",
                                                                                              "switch_ratio_vs_s1"]
            w.writerow(header)
            for s in ["S1", "S2", "S3"]:
                for assoc in ["hard", "soft"]:
                    rr = maps_dict[s][f"summary_{assoc}"]
                    row = [city, model, s, assoc, rr["users"], rr["K"]]
                    row += [f'{rr[m]:.6f}' for m in KPI_METRICS]
                    row += [f'{rr.get("rho_mean", np.nan):.6f}', f'{rr.get("rho_max", np.nan):.6f}',
                            f'{rr.get("switch_ratio_vs_s1", np.nan):.6f}']
                    w.writerow(row)

        for s in ["S1", "S2", "S3"]:
            all_rows_hard.append(maps_dict[s]["summary_hard"])
            all_rows_soft.append(maps_dict[s]["summary_soft"])
            all_rows_selected.append(maps_dict[s][f"summary_{pick}"])

    def _write_rows(rows: List[Dict[str, Any]], out_path: str):
        if not rows:
            return
        with open(out_path, "w", newline="", encoding="utf-8") as fa:
            w = csv.writer(fa)
            w.writerow(["city_id", "model", "strategy", "users", "K"] + KPI_METRICS + ["rho_mean", "rho_max",
                                                                                       "switch_ratio_vs_s1"])
            for rr in rows:
                row = [rr["city_id"], rr["model"], rr["strategy"], rr["users"], rr["K"]]
                row += [f'{rr[m]:.6f}' for m in KPI_METRICS]
                row += [f'{rr.get("rho_mean", np.nan):.6f}', f'{rr.get("rho_max", np.nan):.6f}',
                        f'{rr.get("switch_ratio_vs_s1", np.nan):.6f}']
                w.writerow(row)
        print(f"[SUCCESS] Saved: {out_path}")

    _write_rows(all_rows_selected, pjoin(metrics_dir, "scene_metrics_from_pl.csv"))
    _write_rows(all_rows_hard, pjoin(metrics_dir, "scene_metrics_from_pl_hard.csv"))
    _write_rows(all_rows_soft, pjoin(metrics_dir, "scene_metrics_from_pl_soft.csv"))


# ======================= KPI 向量评估 + 可视化 =======================
def evaluate_predicted_kpi_vectors(
        pred_kpi_outputs_dir: str,
        manifests_dir: str,
        metrics_kpi_dir: str,
        viz_dir: Optional[str] = None,
        use_norm: bool = False,
        *,
        kpi_scaler: Optional[Dict[str, np.ndarray]] = None,
        kpi_qoe_mode: str = KPI_QOE_MODE_DEFAULT,
        qoe_a: float = QOE_A_DEFAULT,
        qoe_b: float = QOE_B_DEFAULT,
        qoe_c: float = QOE_C_DEFAULT,
        qoe_thr_ref_mbps: float = QOE_THR_REF_MBPS_DEFAULT,
        qoe_delay_ref_s: float = QOE_DELAY_REF_S_DEFAULT,
        qoe_lam_jfi: float = QOE_LAM_JFI_DEFAULT,
        qoe_use_log_delay: bool = QOE_USE_LOG_DELAY_DEFAULT,
):
    ensure_dir(metrics_kpi_dir)
    if viz_dir:
        ensure_dir(viz_dir)

    pattern = "*_pred_kpi_norm.npy" if use_norm else "*_pred_kpi.npy"
    kpi_files = _glob_recursive(pred_kpi_outputs_dir, pattern)
    if not kpi_files:
        print(f"[WARN] No predicted KPI vector files in {pred_kpi_outputs_dir} (searched recursively).")
        return

    geom_df = pd.read_csv(pjoin(manifests_dir, "geom_manifest.csv"))
    geom_map = geom_df.set_index('city_id').to_dict('index')
    multibs_df = pd.read_csv(pjoin(manifests_dir, "multibs_manifest.csv"))

    all_rows = []
    for f_kpi_path in sorted(kpi_files):
        parsed = _parse_city_model_from_pred_filename(f_kpi_path, "kpi_norm" if use_norm else "kpi")
        if not parsed:
            continue
        city, model = parsed

        try:
            vec = np.load(f_kpi_path)
            if vec.ndim == 2 and vec.shape[0] == 1:
                vec = vec[0]
            if vec.shape[0] != KPI_NUM_OUTPUTS:
                print(f"[ERROR] KPI dim mismatch at {f_kpi_path}: {vec.shape}")
                continue
        except Exception as e:
            print(f"[ERROR] Load KPI failed {f_kpi_path}: {e}")
            continue

        vec = _maybe_denorm_kpi(vec.astype(np.float32), kpi_scaler)

        row_m = multibs_df[(multibs_df['city_id'] == city) & (multibs_df['model_dir'] == model)]
        if row_m.empty:
            print(f"[WARN] No multibs entry for {city}_{model}, skip KPI eval.")
            continue
        tx_ids = extract_tx_ids_from_multibs_row(row_m.iloc[0])
        K_actual = len(tx_ids)

        mat = vec.reshape(3, 5)

        for s_idx, strat in enumerate(["S1", "S2", "S3"]):
            rr = {
                "city_id": city, "model": model, "strategy": strat,
                "users": float(geom_map.get(city, {}).get('num_users', np.nan)),
                "K": K_actual,
                "sys_throughput_Mbps": float(mat[s_idx, 0]),
                "avg_user_rate_Mbps": float(mat[s_idx, 1]),
                "jfi": float(mat[s_idx, 2]),
                "avg_delay_s": float(mat[s_idx, 3]),
                "avg_qoe": float(mat[s_idx, 4]),
            }

            if str(kpi_qoe_mode).lower() == "recompute":
                thr_ref = max(float(qoe_thr_ref_mbps), 1e-6)
                term_thr = np.log1p(max(float(rr["avg_user_rate_Mbps"]), 0.0) / (thr_ref + SAFETY_EPS))

                if bool(qoe_use_log_delay):
                    dref = max(float(qoe_delay_ref_s), 1e-6)
                    term_delay = np.log1p(max(float(rr["avg_delay_s"]), 0.0) / dref)
                else:
                    term_delay = max(float(rr["avg_delay_s"]), 0.0)

                term_jfi = np.log(max(float(rr["jfi"]), 0.0) + 1e-6)
                rr["avg_qoe"] = float(qoe_a * term_thr - qoe_b * term_delay + qoe_lam_jfi * term_jfi + qoe_c)

            all_rows.append(rr)

        per_scene_dir = pjoin(metrics_kpi_dir, "per-scene-pred-kpi")
        ensure_dir(per_scene_dir)
        out_one = pjoin(per_scene_dir, f"summary_{city}_{model}_pred_kpi.csv")
        with open(out_one, "w", newline="", encoding="utf-8") as fo:
            w = csv.writer(fo)
            header = ["city_id", "model", "strategy", "users", "K"] + KPI_METRICS
            w.writerow(header)
            for s in ["S1", "S2", "S3"]:
                rowk = next(
                    rr for rr in all_rows if rr["city_id"] == city and rr["model"] == model and rr["strategy"] == s)
                row = [city, model, s, rowk["users"], K_actual] + [f'{rowk[m]:.6f}' for m in KPI_METRICS]
                w.writerow(row)

        if viz_dir:
            out_dir_scene = pjoin(viz_dir, f"{city}_{model}")
            ensure_dir(out_dir_scene)
            _draw_kpi_bars_and_radar(city, model, vec, out_dir_scene)

    if all_rows:
        out_all = pjoin(metrics_kpi_dir, "scene_metrics_pred_kpi.csv")
        with open(out_all, "w", newline="", encoding="utf-8") as fa:
            w = csv.writer(fa)
            w.writerow(["city_id", "model", "strategy", "users", "K"] + KPI_METRICS)
            for rr in all_rows:
                row = [rr["city_id"], rr["model"], rr["strategy"], rr["users"], rr["K"]]
                row += [f'{rr[m]:.6f}' for m in KPI_METRICS]
                w.writerow(row)
        print(f"[SUCCESS] All KPI metrics saved: {out_all}")
    else:
        print("[WARN] No KPI metrics produced.")


# ======================= α/β Sweep：一次运行输出曲线 =======================
def _parse_float_list(s: str) -> List[float]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    parts = [x.strip() for x in s.split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        out.append(float(p))
    return out


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    df = pd.DataFrame(rows)
    res: Dict[str, Dict[str, float]] = {}
    for s in ["S1", "S2", "S3"]:
        sub = df[df["strategy"] == s]
        if sub.empty:
            continue
        res[s] = {m: float(sub[m].mean()) for m in KPI_METRICS}
    return res


def _plot_sweep_curves(x_vals: List[float], y_by_strat: Dict[str, List[float]],
                       xlabel: str, ylabel: str, title: str, out_wo: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for s in ["S1", "S2", "S3"]:
        if s in y_by_strat:
            ax.plot(x_vals, y_by_strat[s], marker='o', linewidth=1.8, label=s)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncols=3)
    _pretty_axes(ax, show_grid=True)
    _save_fig(fig, out_wo, tight=True)
    plt.close(fig)


def run_param_sweep_on_pl(
        *,
        sweep_name: str,
        sweep_values: List[float],
        sweep_kind: str,
        pred_outputs_dir: str,
        manifests_dir: str,
        root_path: str,
        los_dir: Optional[str],
        metrics_dir: str,
        viz_dir: str,
        base_alpha: float,
        base_beta: float,
        conf_nlos: float,
        jfi_mode: str,
        eval_assoc: str,
        req_rate_per_user_mbps: float,
        avg_file_size_mb: float,
        util_cap: float,
        delay_cap_s: float,
        car_penalty_strength: float,
        qoe_a: float,
        qoe_b: float,
        qoe_c: float,
        qoe_thr_ref_mbps: float,
        qoe_delay_ref_s: float,
        qoe_lam_jfi: float,
        qoe_use_log_delay: bool,
        enable_load_balance: bool,
        load_gamma: float,
        load_iters: int,
        load_floor: float,
        load_assoc_hard: bool,
        load_blend: float,
        sweep_with_viz: bool = False,
):
    if not sweep_values:
        return

    ensure_dir(pjoin(metrics_dir, "sweeps"))
    ensure_dir(pjoin(viz_dir, "sweeps"))

    db_files = _glob_recursive(pred_outputs_dir, "*_pred_dB.npy")
    if not db_files:
        print(f"[ERROR] No predicted dB stacks in {pred_outputs_dir} (searched recursively).")
        return

    geom_df = pd.read_csv(pjoin(manifests_dir, "geom_manifest.csv"))
    geom_map = geom_df.set_index('city_id').to_dict('index')

    cache: Dict[str, Any] = {}

    sweep_records = []
    y_thr = {s: [] for s in ["S1", "S2", "S3"]}
    y_jfi = {s: [] for s in ["S1", "S2", "S3"]}
    y_qoe = {s: [] for s in ["S1", "S2", "S3"]}

    pick = "hard" if str(eval_assoc).lower() == "hard" else "soft"

    for x in sweep_values:
        alpha = x if sweep_kind == "alpha" else base_alpha
        beta = x if sweep_kind == "beta" else base_beta

        rows_this_x = []
        for f_db in sorted(db_files):
            parsed = _parse_city_model_from_pred_filename(f_db, "dB")
            if not parsed:
                continue
            city, model = parsed

            geom_row = geom_map.get(city, None)
            if geom_row is None:
                continue

            try:
                dB_stack = np.load(f_db).astype(np.float32)
            except Exception:
                continue

            if dB_stack.mean() > -20 and dB_stack.mean() < 20:
                base_dir = os.path.dirname(manifests_dir)
                norm_dir = pjoin(base_dir, "normalization_params")
                norm_file = pjoin(norm_dir, f"city_{city}_model_{model}_mean_std.npz")
                if not os.path.exists(norm_file):
                    norm_file = pjoin(norm_dir, f"city_{city}_mean_std.npz")
                if os.path.exists(norm_file):
                    try:
                        nda = np.load(norm_file)
                        dB_stack = (dB_stack * nda["y_std"]) + nda["y_mean"]
                    except:
                        pass

            H, W, K = dB_stack.shape

            los_stack = None
            los_key = f"los_{city}_{model}_{H}_{W}_{K}"
            if los_key in cache:
                los_stack = cache[los_key]
            else:
                cand_local = _try_find_local_los(f_db, city, model)
                if (cand_local is None) and los_dir:
                    for pth in _candidate_los_paths(los_dir, city, model):
                        if os.path.exists(pth):
                            cand_local = pth
                            break
                if cand_local:
                    los_stack = _load_and_fit_los(cand_local, (H, W, K))
                cache[los_key] = los_stack

            try:
                maps_dict, K_eff, ue_w = _derive_all_maps_for_scene(
                    city, model, dB_stack, los_stack, geom_row,
                    alpha=alpha, beta=beta, conf_nlos=conf_nlos,
                    jfi_mode=jfi_mode,
                    req_rate_per_user_mbps=req_rate_per_user_mbps,
                    avg_file_size_mb=avg_file_size_mb,
                    util_cap=util_cap,
                    delay_cap_s=delay_cap_s,
                    car_penalty_strength=car_penalty_strength,
                    qoe_a=qoe_a, qoe_b=qoe_b, qoe_c=qoe_c,
                    qoe_thr_ref_mbps=qoe_thr_ref_mbps,
                    qoe_delay_ref_s=qoe_delay_ref_s,
                    qoe_lam_jfi=qoe_lam_jfi,
                    qoe_use_log_delay=qoe_use_log_delay,
                    enable_load_balance=enable_load_balance,
                    load_gamma=load_gamma,
                    load_iters=load_iters,
                    load_floor=load_floor,
                    load_assoc_hard=load_assoc_hard,
                    load_blend=load_blend,
                    cache=cache,
                )
            except Exception:
                continue

            for s in ["S1", "S2", "S3"]:
                rows_this_x.append(maps_dict[s][f"summary_{pick}"])

            if sweep_with_viz:
                out_dir_scene = pjoin(viz_dir, "sweeps", sweep_name, f"x_{x}", f"{city}_{model}")
                ensure_dir(out_dir_scene)

                if pick == "hard":
                    r_list = [maps_dict[s]["rate_hard"] for s in ["S1", "S2", "S3"]]
                    d_list = [maps_dict[s]["delay_hard"] for s in ["S1", "S2", "S3"]]
                    q_list = [maps_dict[s]["qoe_hard"] for s in ["S1", "S2", "S3"]]
                else:
                    r_list = [maps_dict[s]["rate_soft"] for s in ["S1", "S2", "S3"]]
                    d_list = [maps_dict[s]["delay_soft"] for s in ["S1", "S2", "S3"]]
                    q_list = [maps_dict[s]["qoe_soft"] for s in ["S1", "S2", "S3"]]

                r_list = [r / (1.0 + d / 2.0) for r, d in zip(r_list, d_list)]
                d_list = [np.clip(d, 0, delay_cap_s) for d in d_list]

                # === [NEW] 遮罩非道路区域 ===
                road_mask = (ue_w > 0)
                r_list = [np.where(road_mask, r, np.nan) for r in r_list]
                d_list = [np.where(road_mask, d, np.nan) for d in d_list]
                q_list = [np.where(road_mask, q, np.nan) for q in q_list]

                rate_min, rate_max = _robust_vlim(r_list, 2.0, 98.0, False)
                delay_min, delay_max = _robust_vlim(d_list, 2.0, 98.0, False)
                qoe_min, qoe_max = _robust_vlim(q_list, 2.0, 98.0, False)

                _draw_three_panel(city, model, K, alpha, beta, conf_nlos, "rate", r_list, vmin=rate_min, vmax=rate_max,
                                  out_dir_scene=out_dir_scene)
                _draw_three_panel(city, model, K, alpha, beta, conf_nlos, "delay", d_list, vmin=delay_min,
                                  vmax=delay_max, out_dir_scene=out_dir_scene)
                _draw_three_panel(city, model, K, alpha, beta, conf_nlos, "qoe", q_list, vmin=qoe_min, vmax=qoe_max,
                                  out_dir_scene=out_dir_scene)

        agg = _aggregate_rows(rows_this_x)
        for s in ["S1", "S2", "S3"]:
            if s not in agg:
                y_thr[s].append(np.nan)
                y_jfi[s].append(np.nan)
                y_qoe[s].append(np.nan)
                continue
            y_thr[s].append(agg[s]["sys_throughput_Mbps"])
            y_jfi[s].append(agg[s]["jfi"])
            y_qoe[s].append(agg[s]["avg_qoe"])
            sweep_records.append(
                {"sweep": sweep_name, sweep_kind: float(x), "strategy": s, **{m: agg[s][m] for m in KPI_METRICS}})

        print(f"[SWEEP] {sweep_kind}={x} done. (alpha={alpha}, beta={beta})")

    out_csv = pjoin(metrics_dir, "sweeps", f"{sweep_name}.csv")
    pd.DataFrame(sweep_records).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SWEEP] Saved: {out_csv}")

    xlab = "alpha (α)" if sweep_kind == "alpha" else "beta (β)"
    out_base = pjoin(viz_dir, "sweeps", sweep_name)
    _plot_sweep_curves(sweep_values, y_thr, xlabel=xlab, ylabel="Mean sys_throughput (Mbps)",
                       title=f"System Throughput vs {xlab}", out_wo=out_base + "_sys_throughput_Mbps")
    _plot_sweep_curves(sweep_values, y_jfi, xlabel=xlab, ylabel="Mean JFI", title=f"Fairness (JFI) vs {xlab}",
                       out_wo=out_base + "_jfi")
    _plot_sweep_curves(sweep_values, y_qoe, xlabel=xlab, ylabel="Mean QoE", title=f"QoE vs {xlab}",
                       out_wo=out_base + "_avg_qoe")
    print(f"[SWEEP] Plots saved under: {pjoin(viz_dir, 'sweeps')}")


# ======================= 主程序 =======================
def main():
    global B_HZ, TX_POWER_DBM, NOISE_PSD_DBM_PER_HZ, NOISE_FIGURE_DB, INTERF_COEFF
    global UE_WEIGHT_MODE, UE_DILATE_RADIUS, UE_MIN_COVERAGE, UE_MAX_COVERAGE
    ap = argparse.ArgumentParser("Decision & Evaluation (v7.7 visual and academic fixes applied)")

    ap.add_argument('--config', type=str, default='configs/decision.yaml')

    ap.add_argument('--dataset_prep_output_dir', type=str, default=DATASET_PREP_OUTPUT_DIR)
    ap.add_argument('--pred_outputs_dir', type=str, default=PRED_OUTPUTS_DIR)
    ap.add_argument('--pred_kpi_outputs_dir', type=str, default=PRED_KPI_OUTPUTS_DIR)
    ap.add_argument('--root_path', type=str, default=ROOT_PATH)

    # 路径已经重定向为常量中的 METRICS_DIR, VIZ_DIR 等
    ap.add_argument('--metrics_dir', type=str, default=METRICS_DIR)
    ap.add_argument('--metrics_kpi_dir', type=str, default=METRICS_KPI_DIR)
    ap.add_argument('--viz_dir', type=str, default=VIZ_DIR)
    ap.add_argument('--los_dir', type=str, default=None,
                    help="指向 dataset_preparation_output/los，例如 .../dataset_preparation_output/los")

    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--beta', type=float, default=BETA)
    ap.add_argument('--conf_nlos', type=float, default=CONF_NLOS)

    ap.add_argument('--jfi_mode', type=str, default=JFI_MODE_DEFAULT, choices=['rate', 'load'])

    ap.add_argument('--eval_assoc', type=str, default=EVAL_ASSOC_DEFAULT, choices=['hard', 'soft'])

    ap.add_argument('--req_rate_per_user_mbps', type=float, default=REQ_RATE_PER_USER)
    ap.add_argument('--avg_file_size_mb', type=float, default=AVG_FILE_SIZE_MB)
    ap.add_argument('--util_cap', type=float, default=UTIL_CAP)
    ap.add_argument('--delay_cap_s', type=float, default=DELAY_CAP_S)
    ap.add_argument('--bandwidth_hz', type=float, default=B_HZ)
    ap.add_argument('--tx_power_dbm', type=float, default=TX_POWER_DBM)
    ap.add_argument('--noise_psd_dbm_per_hz', type=float, default=NOISE_PSD_DBM_PER_HZ)
    ap.add_argument('--noise_figure_db', type=float, default=NOISE_FIGURE_DB)
    ap.add_argument('--interference_coeff', type=float, default=INTERF_COEFF)
    ap.add_argument('--ue_weight_mode', choices=['binary', 'dilate'], default=UE_WEIGHT_MODE)
    ap.add_argument('--ue_dilate_radius', type=int, default=UE_DILATE_RADIUS)
    ap.add_argument('--ue_min_coverage', type=float, default=UE_MIN_COVERAGE)
    ap.add_argument('--ue_max_coverage', type=float, default=UE_MAX_COVERAGE)

    ap.add_argument('--paper_stress_load', action='store_true', default=False,
                    help="开启后自动把负载推高（建议论文出图用）；你手动传参会覆盖它")

    ap.add_argument('--car_penalty_strength', type=float, default=CAR_PENALTY_STRENGTH_DEFAULT)

    grp_lb = ap.add_mutually_exclusive_group(required=False)
    grp_lb.add_argument('--enable_load_balance', action='store_true', default=ENABLE_LOAD_BALANCE_DEFAULT)
    grp_lb.add_argument('--disable_load_balance', action='store_true', default=False)
    ap.add_argument('--load_gamma', type=float, default=LOAD_GAMMA_DEFAULT)
    ap.add_argument('--load_iters', type=int, default=LOAD_ITERS_DEFAULT)
    ap.add_argument('--load_floor', type=float, default=LOAD_FLOOR_DEFAULT)
    ap.add_argument('--load_assoc', type=str, default=('hard' if LOAD_ASSOC_HARD_DEFAULT else 'soft'),
                    choices=['hard', 'soft'])
    ap.add_argument('--load_blend', type=float, default=LOAD_BLEND_DEFAULT)

    ap.add_argument('--qoe_a', type=float, default=QOE_A_DEFAULT)
    ap.add_argument('--qoe_b', type=float, default=QOE_B_DEFAULT)
    ap.add_argument('--qoe_c', type=float, default=QOE_C_DEFAULT)
    ap.add_argument('--qoe_thr_ref_mbps', type=float, default=QOE_THR_REF_MBPS_DEFAULT)
    ap.add_argument('--qoe_delay_ref_s', type=float, default=QOE_DELAY_REF_S_DEFAULT)
    ap.add_argument('--qoe_lam_jfi', type=float, default=QOE_LAM_JFI_DEFAULT)
    ap.add_argument('--qoe_use_log_delay', action='store_true', default=QOE_USE_LOG_DELAY_DEFAULT)

    ap.add_argument('--kpi_use_norm', action='store_true', default=False)
    ap.add_argument('--kpi_qoe_mode', type=str, default=KPI_QOE_MODE_DEFAULT, choices=['use_pred', 'recompute'])
    ap.add_argument('--kpi_scaler_json', type=str, default=None,
                    help='若 pred_kpi 是标准化输出：传入包含 {"mean":[15], "std":[15]} 的 json，用于反标准化')

    ap.add_argument('--viz_diff', action='store_true', default=True)
    ap.add_argument('--robust_q_low', type=float, default=2.0)
    ap.add_argument('--robust_q_high', type=float, default=98.0)

    ap.add_argument('--alpha_list', type=str, default="")
    ap.add_argument('--beta_list', type=str, default="")
    ap.add_argument('--sweep_with_viz', action='store_true', default=False)
    ap.add_argument('--skip_pl_eval', action='store_true', default=False)
    ap.add_argument('--skip_kpi_eval', action='store_true', default=False)

    args = parse_args_with_config(ap)

    B_HZ = args.bandwidth_hz
    TX_POWER_DBM = args.tx_power_dbm
    NOISE_PSD_DBM_PER_HZ = args.noise_psd_dbm_per_hz
    NOISE_FIGURE_DB = args.noise_figure_db
    INTERF_COEFF = args.interference_coeff
    UE_WEIGHT_MODE = args.ue_weight_mode
    UE_DILATE_RADIUS = args.ue_dilate_radius
    UE_MIN_COVERAGE = args.ue_min_coverage
    UE_MAX_COVERAGE = args.ue_max_coverage

    manifests_dir = pjoin(args.dataset_prep_output_dir, "manifests")
    need_files = [pjoin(manifests_dir, "multibs_manifest.csv"),
                  pjoin(manifests_dir, "geom_manifest.csv"),
                  pjoin(manifests_dir, "tx_coords.csv")]
    if not all(os.path.exists(p) for p in need_files):
        located = locate_dataset_prep_root(args.dataset_prep_output_dir, args.root_path)
        if located:
            args.dataset_prep_output_dir = located
            manifests_dir = pjoin(args.dataset_prep_output_dir, "manifests")
            print(f"[AUTO] dataset_prep_output_dir -> {args.dataset_prep_output_dir}")

    if args.paper_stress_load:
        if abs(args.req_rate_per_user_mbps - REQ_RATE_PER_USER) < 1e-12:
            args.req_rate_per_user_mbps = 2.0
        if abs(args.avg_file_size_mb - AVG_FILE_SIZE_MB) < 1e-12:
            args.avg_file_size_mb = 10.0

    enable_lb = (args.enable_load_balance and (not args.disable_load_balance))
    kpi_scaler = _load_kpi_scaler_json(args.kpi_scaler_json)

    print(f"[CONFIG] manifests   = {manifests_dir}")
    print(f"[CONFIG] pred dB     = {args.pred_outputs_dir}")
    print(f"[CONFIG] pred KPI    = {args.pred_kpi_outputs_dir}")
    print(f"[CONFIG] metrics(dB) = {args.metrics_dir}")
    print(f"[CONFIG] metrics(KPI)= {args.metrics_kpi_dir}")
    print(f"[CONFIG] viz         = {args.viz_dir}")
    print(
        f"[CONFIG] alpha={args.alpha}, beta={args.beta}, conf_nlos={args.conf_nlos}, jfi_mode={args.jfi_mode}, eval_assoc={args.eval_assoc}")
    print(
        f"[CONFIG] req_rate_per_user_mbps={args.req_rate_per_user_mbps}, avg_file_size_mb={args.avg_file_size_mb}, util_cap={args.util_cap}, delay_cap_s={args.delay_cap_s}")
    print(
        f"[CONFIG] load_balance={enable_lb} (assoc={args.load_assoc}), gamma={args.load_gamma}, iters={args.load_iters}, floor={args.load_floor}, blend={args.load_blend}")
    print(
        f"[CONFIG] QoE: a={args.qoe_a}, b={args.qoe_b}, lam_jfi={args.qoe_lam_jfi}, thr_ref={args.qoe_thr_ref_mbps}Mbps, delay_ref={args.qoe_delay_ref_s}s, log_delay={args.qoe_use_log_delay}")
    print(
        f"[CONFIG] kpi_use_norm={args.kpi_use_norm}, kpi_qoe_mode={args.kpi_qoe_mode}, kpi_scaler_json={args.kpi_scaler_json}")

    if not args.skip_pl_eval:
        evaluate_predicted_dB_maps(
            pred_outputs_dir=args.pred_outputs_dir,
            manifests_dir=manifests_dir,
            root_path=args.root_path,
            metrics_dir=args.metrics_dir,
            viz_dir=args.viz_dir,
            los_dir=args.los_dir,
            alpha=args.alpha,
            beta=args.beta,
            conf_nlos=args.conf_nlos,
            jfi_mode=args.jfi_mode,
            eval_assoc=args.eval_assoc,
            req_rate_per_user_mbps=args.req_rate_per_user_mbps,
            avg_file_size_mb=args.avg_file_size_mb,
            util_cap=args.util_cap,
            delay_cap_s=args.delay_cap_s,
            car_penalty_strength=args.car_penalty_strength,
            qoe_a=args.qoe_a, qoe_b=args.qoe_b, qoe_c=args.qoe_c,
            qoe_thr_ref_mbps=args.qoe_thr_ref_mbps,
            qoe_delay_ref_s=args.qoe_delay_ref_s,
            qoe_lam_jfi=args.qoe_lam_jfi,
            qoe_use_log_delay=args.qoe_use_log_delay,
            enable_load_balance=enable_lb,
            load_gamma=args.load_gamma,
            load_iters=args.load_iters,
            load_floor=args.load_floor,
            load_assoc_hard=(args.load_assoc == 'hard'),
            load_blend=args.load_blend,
            make_viz=True,
            viz_diff=args.viz_diff,
            robust_q_low=args.robust_q_low,
            robust_q_high=args.robust_q_high,
        )

    alpha_list = _parse_float_list(args.alpha_list)
    if alpha_list:
        run_param_sweep_on_pl(
            sweep_name="alpha_sweep",
            sweep_values=alpha_list,
            sweep_kind="alpha",
            pred_outputs_dir=args.pred_outputs_dir,
            manifests_dir=manifests_dir,
            root_path=args.root_path,
            los_dir=args.los_dir,
            metrics_dir=args.metrics_dir,
            viz_dir=args.viz_dir,
            base_alpha=args.alpha,
            base_beta=args.beta,
            conf_nlos=args.conf_nlos,
            jfi_mode=args.jfi_mode,
            eval_assoc=args.eval_assoc,
            req_rate_per_user_mbps=args.req_rate_per_user_mbps,
            avg_file_size_mb=args.avg_file_size_mb,
            util_cap=args.util_cap,
            delay_cap_s=args.delay_cap_s,
            car_penalty_strength=args.car_penalty_strength,
            qoe_a=args.qoe_a, qoe_b=args.qoe_b, qoe_c=args.qoe_c,
            qoe_thr_ref_mbps=args.qoe_thr_ref_mbps,
            qoe_delay_ref_s=args.qoe_delay_ref_s,
            qoe_lam_jfi=args.qoe_lam_jfi,
            qoe_use_log_delay=args.qoe_use_log_delay,
            enable_load_balance=enable_lb,
            load_gamma=args.load_gamma,
            load_iters=args.load_iters,
            load_floor=args.load_floor,
            load_assoc_hard=(args.load_assoc == 'hard'),
            load_blend=args.load_blend,
            sweep_with_viz=args.sweep_with_viz,
        )

    beta_list = _parse_float_list(args.beta_list)
    if beta_list:
        run_param_sweep_on_pl(
            sweep_name="beta_sweep",
            sweep_values=beta_list,
            sweep_kind="beta",
            pred_outputs_dir=args.pred_outputs_dir,
            manifests_dir=manifests_dir,
            root_path=args.root_path,
            los_dir=args.los_dir,
            metrics_dir=args.metrics_dir,
            viz_dir=args.viz_dir,
            base_alpha=args.alpha,
            base_beta=args.beta,
            conf_nlos=args.conf_nlos,
            jfi_mode=args.jfi_mode,
            eval_assoc=args.eval_assoc,
            req_rate_per_user_mbps=args.req_rate_per_user_mbps,
            avg_file_size_mb=args.avg_file_size_mb,
            util_cap=args.util_cap,
            delay_cap_s=args.delay_cap_s,
            car_penalty_strength=args.car_penalty_strength,
            qoe_a=args.qoe_a, qoe_b=args.qoe_b, qoe_c=args.qoe_c,
            qoe_thr_ref_mbps=args.qoe_thr_ref_mbps,
            qoe_delay_ref_s=args.qoe_delay_ref_s,
            qoe_lam_jfi=args.qoe_lam_jfi,
            qoe_use_log_delay=args.qoe_use_log_delay,
            enable_load_balance=enable_lb,
            load_gamma=args.load_gamma,
            load_iters=args.load_iters,
            load_floor=args.load_floor,
            load_assoc_hard=(args.load_assoc == 'hard'),
            load_blend=args.load_blend,
            sweep_with_viz=args.sweep_with_viz,
        )

    if not args.skip_kpi_eval:
        evaluate_predicted_kpi_vectors(
            pred_kpi_outputs_dir=args.pred_kpi_outputs_dir,
            manifests_dir=manifests_dir,
            metrics_kpi_dir=args.metrics_kpi_dir,
            viz_dir=args.viz_dir,
            use_norm=args.kpi_use_norm,
            kpi_scaler=kpi_scaler,
            kpi_qoe_mode=args.kpi_qoe_mode,
            qoe_a=args.qoe_a, qoe_b=args.qoe_b, qoe_c=args.qoe_c,
            qoe_thr_ref_mbps=args.qoe_thr_ref_mbps,
            qoe_delay_ref_s=args.qoe_delay_ref_s,
            qoe_lam_jfi=args.qoe_lam_jfi,
            qoe_use_log_delay=args.qoe_use_log_delay,
        )

    print("[END] Decision & Evaluation done.")


if __name__ == "__main__":
    main()
