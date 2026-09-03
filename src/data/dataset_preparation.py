# -*- coding: utf-8 -*-
"""
dataset_preparation.py —— RadioMapSeer 数据准备 + 统一特征Schema + 几何LOS重建(必生成) + KPI标签生成(必生成)

新增/强化要点（与源代码相比）：
1) Non-IID 场景标注：基于城市“信道丰度”近似指标自动划分 light / medium / heavy，并写出 manifests/noniid_manifest.csv 与 subset_plan.json 内部字段。
2) geom_manifest 增补 num_users（基于 roads 权重的像素近似计数），避免后续评估缺列。
3) LOS 目录“别名镜像”：保存到 canonical 目录（如 carsIRT2）同时**镜像**到 carslRT2 / carslRT4 等别名子目录，确保 decision.py 的别名容错一定能“found”。
4) 多模型一致性：core/ablation 模型均可出数，支持 K=32 城市子集（按参数控制）。
5) 统一 102 维 schema，K<=97 用 tx_mask 标识；输出 schema_v1.json。
6) KPI 15 维标签（3策略×5指标）始终生成到 users_kpi/。
7) 末尾校验：los/ 下至少出现一个 .npy，否则抛出明确排查建议。
"""

import os
import re
import csv
import json
import random
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import argparse

from src.utils.io_utils import parse_args_with_config

# ===================== Portable defaults (overridden by YAML) =====================
ROOT_PATH = "data/RadioMapSeer"
OUTPUT_DIR = "outputs/dataset"

# 计划参数
TOTAL_CITIES = 12
RANDOM_SEED = 0
CORE_MODELS = ["carsIRT2", "carsDPM"]
ABLATION_MODELS = ["carsIRT4"]
K_CORE = 16
EXTRA_K32_CITIES = 2  # 有些城市对 core 模型升为 K=32

# ===================== 模型范围/别名/数值范围 =====================
# dB 解码范围：True -> 读到的是 pathGAIN（pr = tx_power + dB），False -> 读 pathLOSS（pr = tx_power - dB）
MODEL_DB_RANGES = {
    "IRT2": (True, -160.0, -40.0),
    "carsIRT2": (True, -160.0, -40.0),
    "IRT4": (True, -160.0, -40.0),
    "carsIRT4": (True, -160.0, -40.0),
    "DPM": (True, -160.0, -40.0),
    "carsDPM": (True, -160.0, -40.0),
}

# 磁盘目录别名映射：逻辑名 -> 实际可能出现的目录名（优先级从左到右）
MODEL_ALIASES = {
    "carsIRT2": ["carsIRT2", "carslRT2"],  # 注意：磁盘可能是 carslRT2（小写L）
    "carsIRT4": ["carsIRT4", "carslRT4"],
    "IRT2":     ["IRT2", "lRT2"],
    "IRT4":     ["IRT4", "lRT4"],
    "DPM":      ["DPM"],
    "carsDPM":  ["carsDPM"],
}
CORE_MODEL_SET = set(CORE_MODELS)
ABLATION_MODEL_SET = set(ABLATION_MODELS)

# 用于 LOS 镜像保存的别名表（写出到 canonical 与 alias 两处，保证能“found”）
LOS_ALIAS_MIRRORS = {
    "carsIRT2": ["carslRT2"],
    "carsIRT4": ["carslRT4"],
    "carsDPM":  [],          # 无额外别名
    "IRT2":     ["lRT2"],
    "IRT4":     ["lRT4"],
    "DPM":      []
}

# ===================== 系统/策略参数 =====================
B_HZ               = 10_000_000
TX_POWER_DBM       = 23.0
NOISE_PSD_DBM_PER_HZ = -174.0
NOISE_FIGURE_DB    = 5.0
INTERF_COEFF       = 1.0
ALPHA              = 1.0
BETA               = 1.0
CONF_NLOS          = 0.5
REQ_RATE_PER_USER  = 0.2
AVG_FILE_SIZE_MB   = 2.0
UTIL_CAP           = 0.98
DELAY_CAP_S        = 10.0
SAFETY_EPS         = 1e-12
IS_GAIN_FALLBACK   = True
GAIN_DB_MIN, GAIN_DB_MAX = -160.0, -40.0

UE_WEIGHT_MODE     = "dilate"  # "binary" | "dilate"
UE_DILATE_RADIUS   = 1
UE_MIN_COVERAGE    = 1e-4
UE_MAX_COVERAGE    = 0.90

# ===================== 特征 Schema =====================
FEATURE_VERSION = "v1.0"
SCHEMA_IN_DIM = 102           # 固定 102 维
K_MAX = 97                    # 102 - 5 基础通道
BASE_CHANNELS = ["x_norm", "y_norm", "buildings", "roads", "cars"]

# ===================== 工具函数 =====================
def ensure_dir(d: str): os.makedirs(d, exist_ok=True)
def pjoin(*x): return os.path.join(*x).replace("/", os.sep)
def file_exists(p: str): return os.path.exists(p)

def resolve_model_dir_name(root_path: str, logical_model: str) -> Optional[str]:
    """把逻辑模型名映射为磁盘真实目录名（gain/<model>）。"""
    candidates = MODEL_ALIASES.get(logical_model, [logical_model])
    for name in candidates:
        if os.path.isdir(pjoin(root_path, "gain", name)):
            return name
    if os.path.isdir(pjoin(root_path, "gain", logical_model)):
        return logical_model
    return None

def canonical_model_for_ranges(model_dir_on_disk: str) -> str:
    """把磁盘目录名映射回用于查 dB 范围的规范键。"""
    for canon, aliases in MODEL_ALIASES.items():
        if model_dir_on_disk in aliases or model_dir_on_disk == canon:
            return canon
    return model_dir_on_disk

def noise_dbm(b_hz: float) -> float:
    return NOISE_PSD_DBM_PER_HZ + 10.0 * np.log10(b_hz) + NOISE_FIGURE_DB

def read_gray_to_db_with_model(img_path: str, model_dir_on_disk: str) -> Tuple[np.ndarray, bool]:
    img = Image.open(img_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    canon = canonical_model_for_ranges(model_dir_on_disk)
    if canon in MODEL_DB_RANGES:
        is_gain_m, dmin, dmax = MODEL_DB_RANGES[canon]
    else:
        is_gain_m, dmin, dmax = IS_GAIN_FALLBACK, GAIN_DB_MIN, GAIN_DB_MAX
    db = dmin + (arr / 255.0) * (dmax - dmin)
    return db, is_gain_m

# ===================== 子集选择与 TX 统计 =====================
def get_available_tx_count_per_model_and_city(root_path: str, models: List[str], cities: Set[int]) -> Dict[str, Dict[int, int]]:
    available_tx_counts = {}
    for logical_model in models:
        disk_model = resolve_model_dir_name(root_path, logical_model)
        if disk_model is None:
            print(f"[WARN] gain/<{logical_model}> not found under {pjoin(root_path,'gain')}")
            continue
        model_dir_path = pjoin(root_path, "gain", disk_model)
        available_tx_counts[logical_model] = {}
        for city_id in cities:
            count = 0
            for i in range(1000):  # 最多扫1000个发射机
                if os.path.exists(pjoin(model_dir_path, f"{city_id}_{i}.png")):
                    count = i + 1
                else:
                    break
            available_tx_counts[logical_model][city_id] = count
        print(f"[Info] Model {logical_model} -> disk '{disk_model}': TX counts for {len(available_tx_counts[logical_model])} cities.")
    return available_tx_counts

def farthest_point_sample(coords_xy, K):
    N = len(coords_xy)
    if K >= N: return list(range(N))
    xs = np.array([p[0] for p in coords_xy], dtype=float)
    ys = np.array([p[1] for p in coords_xy], dtype=float)
    cx, cy = xs.mean(), ys.mean()
    d0 = (xs - cx)**2 + (ys - cy)**2
    start = int(np.argmin(d0))
    selected = [start]
    dist = np.full(N, np.inf)
    for _ in range(1, K):
        last = selected[-1]
        dx = xs - xs[last]; dy = ys - ys[last]
        dist = np.minimum(dist, dx*dx + dy*dy)
        dist[selected] = -1
        nxt = int(np.argmax(dist))
        selected.append(nxt)
    return selected

def select_subset_and_write_plan(total_cities: int, k_core: int, extra_k32: int,
                                 core_models: List[str], ablation_models: List[str],
                                 root_path: str, output_dir: str, random_seed: int = 0):
    random.seed(random_seed); np.random.seed(random_seed)
    all_models = core_models + ablation_models

    # 用 core 模型的 tx0 图均值作为“信道丰度”近似指标（轻量，避免遍历全部K）
    los_proxy_by_city = {}
    representative_model = core_models[0] if core_models else "carsIRT2"
    disk_rep = resolve_model_dir_name(root_path, representative_model) or representative_model
    gain_dir = pjoin(root_path, "gain", disk_rep)
    if os.path.isdir(gain_dir):
        for fn in os.listdir(gain_dir):
            m = re.match(r"^(\d+)_(\d+)\.png$", fn)
            if m and int(m.group(2)) == 0:
                city = int(m.group(1))
                db_path = pjoin(gain_dir, f"{city}_0.png")
                try:
                    db_arr, _ = read_gray_to_db_with_model(db_path, disk_rep)
                    los_proxy_by_city[city] = float(db_arr.mean())
                except Exception as e:
                    print(f"[WARN] LOS-proxy city {city} failed: {e}")
    if not los_proxy_by_city:
        raise RuntimeError("未找到任何城市（gain 目录为空或命名不匹配）。")

    # 基于 proxy 值划分 Non-IID 场景：light / medium / heavy
    vals = np.array(list(los_proxy_by_city.values()), dtype=float)
    q33, q66 = np.quantile(vals, [0.33, 0.66])
    city2scenario = {}
    for city, v in los_proxy_by_city.items():
        if v <= q33: city2scenario[city] = "light"
        elif v <= q66: city2scenario[city] = "medium"
        else: city2scenario[city] = "heavy"

    # 三类均衡抽取 total_cities
    cls_light = [(c, v) for c, v in los_proxy_by_city.items() if city2scenario[c] == "light"]
    cls_medium= [(c, v) for c, v in los_proxy_by_city.items() if city2scenario[c] == "medium"]
    cls_heavy = [(c, v) for c, v in los_proxy_by_city.items() if city2scenario[c] == "heavy"]
    random.shuffle(cls_light); random.shuffle(cls_medium); random.shuffle(cls_heavy)
    per = max(1, total_cities // 3)
    picked = cls_light[:per] + cls_medium[:per] + cls_heavy[:per]
    if len(picked) < total_cities:
        others = cls_light[per:] + cls_medium[per:] + cls_heavy[per:]
        random.shuffle(others)
        picked += others[:(total_cities - len(picked))]
    picked_cities = [c for c, _ in picked]

    # 可用Tx统计
    available_tx_counts = get_available_tx_count_per_model_and_city(root_path, all_models, set(picked_cities))
    k32_cities = set(random.sample(picked_cities, min(extra_k32, len(picked_cities))))

    # 逐 城市-模型 选择TX
    city_model_tx_plan = {}
    for c in picked_cities:
        ant_path = pjoin(root_path, "antenna", f"{c}.json")
        if not file_exists(ant_path):
            print(f"[WARN] missing antenna json: {ant_path}")
            continue
        with open(ant_path, "r", encoding="utf-8") as f:
            coords_json = json.load(f)
        coords = [(float(x), float(y)) for (x, y) in coords_json]
        for model in all_models:
            if model not in available_tx_counts:
                continue
            avail = available_tx_counts[model].get(c, 0)
            if avail <= 0:
                continue
            if model in ABLATION_MODEL_SET:
                ks_final = min(avail, len(coords))
                idxs = list(range(ks_final))
            else:
                reqK = 32 if (c in k32_cities and model in CORE_MODEL_SET) else k_core
                ks_final = min(reqK, avail, len(coords))
                if ks_final <= 0:
                    print(f"[WARN] city {c}, model {model} no TX; skip")
                    continue
                if ks_final < reqK:
                    print(f"[Info] city {c}, model {model}: requested {reqK}, limited {ks_final}")
                idxs = farthest_point_sample(coords, ks_final)
            city_model_tx_plan[(c, model)] = sorted([int(i) for i in idxs])

    # subset_plan.json
    plan = {
        "seed": random_seed,
        "noniid_quantiles": {"q33": float(q33), "q66": float(q66)},
        "city2scenario": {str(c): city2scenario.get(c, "medium") for c in picked_cities},
        "models": {"core": core_models, "ablations": ablation_models},
        "tx_selection_per_model": {f"{k}": v for k, v in city_model_tx_plan.items()},
        "k32_cities": list(map(int, [c for c in k32_cities if any((c, m) in city_model_tx_plan for m in all_models)])),
        "selection_mode": "auto_from_gain_mean"
    }
    ensure_dir(output_dir)
    with open(pjoin(output_dir, "subset_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    # 写 Non-IID 场景清单
    mani_dir = pjoin(output_dir, "manifests"); ensure_dir(mani_dir)
    noniid_csv = pjoin(mani_dir, "noniid_manifest.csv")
    with open(noniid_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["city_id", "scenario"])  # light/medium/heavy
        for c in sorted(picked_cities):
            w.writerow([c, city2scenario.get(c, "medium")])
    print(f"[OK] noniid_manifest.csv -> {noniid_csv}")

    # multibs_manifest.csv（pl_path* 一律用磁盘真实目录名）
    multibs_csv = pjoin(mani_dir, "multibs_manifest.csv")
    headers = ["city_id", "model_dir"] + [f"tx{i}" for i in range(1, 81)] + [f"pl_path{i}" for i in range(1, 81)]
    with open(multibs_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv); w.writerow(headers)
        for (c, m), txs in sorted(city_model_tx_plan.items()):
            disk_model = resolve_model_dir_name(root_path, m)
            if disk_model is None:
                print(f"[WARN] {m} no disk gain folder, skip row.")
                continue
            row = [c, m] + [""] * 80 + [""] * 80
            for i, tx in enumerate(txs[:80]):
                row[2 + i] = tx
                row[2 + 80 + i] = pjoin("gain", disk_model, f"{c}_{tx}.png")
            w.writerow(row)

    # ========== sanity_report 写盘 ==========
    sr_dir = pjoin(output_dir, "sanity_report"); ensure_dir(sr_dir)
    summary_csv = pjoin(sr_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fsum:
        w = csv.writer(fsum)
        w.writerow(["city_id", "model_dir", "K_selected", "K_available"])
        for (c, m), txs in sorted(city_model_tx_plan.items()):
            kavail = get_available_tx_count_per_model_and_city(root_path, [m], {c}).get(m, {}).get(c, 0)
            w.writerow([c, m, len(txs), kavail])
    plan_txt = pjoin(sr_dir, "plan_readable.txt")
    with open(plan_txt, "w", encoding="utf-8") as ftxt:
        ftxt.write(f"seed={random_seed}\n")
        ftxt.write(f"k_core={k_core}, extra_k32={list(map(int, plan['k32_cities']))}\n")
        ftxt.write(f"core_models={core_models}, ablation_models={ablation_models}\n\n")
        for (c, m), txs in sorted(city_model_tx_plan.items()):
            ftxt.write(f"[city {c} | model {m}] K={len(txs)}  tx_ids={txs[:10]}{'...' if len(txs)>10 else ''}\n")

    print(f"[OK] subset_plan.json + manifests/multibs_manifest.csv 写入完成。")
    print(f"[OK] sanity_report 写入：{summary_csv} / {plan_txt}")
    used_cities = set(k[0] for k in city_model_tx_plan)
    return multibs_csv, used_cities, plan, noniid_csv

# ===================== 其它 manifests =====================
def find_first_png(dir_rel: str, key_prefix: str, root_path: str):
    d = pjoin(root_path, dir_rel)
    if not os.path.isdir(d): return ""
    for fn in sorted(os.listdir(d)):
        if fn.startswith(key_prefix) and fn.lower().endswith(".png"):
            return pjoin(dir_rel, fn)
    return ""

def _binary_roads_mask_to_weight(arr_u8: np.ndarray, mode="binary", dilate_r=1):
    m = (arr_u8 > 0).astype(np.float32)
    if mode == "binary": return m
    w = m.copy()
    for _ in range(max(1, int(dilate_r))):
        w = np.maximum.reduce([
            w,
            np.pad(w, ((1,0),(0,0)))[:-1, :],
            np.pad(w, ((0,1),(0,0)))[1:, :],
            np.pad(w, ((0,0),(1,0)))[:, :-1],
            np.pad(w, ((0,0),(0,1)))[:, 1:],
        ])
    return w

def build_geom_manifest(cities_used: Set[int], root_path: str, output_dir: str):
    """写 geom_manifest.csv，并估算 num_users（基于 roads 权重累加）。"""
    buildings_json_dir = "polygon/buildings_complete"
    roads_json_dir     = "polygon/roads"
    bnc_json_dir       = "polygon/buildings_and_cars"
    antennas_png_dir   = "png/antennas"
    buildings_png_dir  = "png/buildings_complete"
    cars_png_dir       = "png/cars"
    roads_png_dir      = "png/roads"

    mani_dir = pjoin(output_dir, "manifests"); ensure_dir(mani_dir)
    out_csv = pjoin(mani_dir, "geom_manifest.csv")
    rows = []
    for city in sorted(map(int, cities_used)):
        cstr = str(city)
        bjson = pjoin(buildings_json_dir, f"{cstr}.json")
        rjson = pjoin(roads_json_dir,     f"{cstr}.json")
        bcjson= pjoin(bnc_json_dir,       f"{cstr}.json")
        antennas_any = ""
        ant_dir_full = pjoin(root_path, antennas_png_dir)
        if os.path.isdir(ant_dir_full):
            preferred = pjoin(antennas_png_dir, f"{cstr}_0.png")
            antennas_any = preferred if file_exists(pjoin(root_path, preferred)) \
                                     else find_first_png(antennas_png_dir, f"{cstr}_", root_path)
        buildings_png = pjoin(buildings_png_dir, f"{cstr}.png") if file_exists(pjoin(root_path, buildings_png_dir, f"{cstr}.png")) else ""
        cars_png      = pjoin(cars_png_dir,      f"{cstr}.png") if file_exists(pjoin(root_path, cars_png_dir,      f"{cstr}.png")) else ""
        roads_png     = pjoin(roads_png_dir,     f"{cstr}.png") if file_exists(pjoin(root_path, roads_png_dir,     f"{cstr}.png")) else ""

        # 估算 num_users：对 roads 做权重（膨胀）后求和
        num_users = np.nan
        if roads_png and file_exists(pjoin(root_path, roads_png)):
            im = Image.open(pjoin(root_path, roads_png)).convert("L")
            W, H = im.size
            arr = np.asarray(im.resize((W, H), Image.NEAREST), dtype=np.uint8)
            weight = _binary_roads_mask_to_weight(arr, mode=UE_WEIGHT_MODE, dilate_r=UE_DILATE_RADIUS).astype(np.float32)
            # 归一化到均值1，累计和可视为“用户像素数”
            m = weight.mean()
            weight = weight / m if m > 1e-6 else np.ones_like(weight)
            num_users = float(weight.sum())

        rows.append([cstr,
                     bjson if file_exists(pjoin(root_path, bjson)) else "",
                     rjson if file_exists(pjoin(root_path, rjson)) else "",
                     bcjson if file_exists(pjoin(root_path, bcjson)) else "",
                     antennas_any, buildings_png, cars_png, roads_png,
                     num_users])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["city_id","buildings_json","roads_json","buildings_and_cars_json",
                    "antennas_png_any","buildings_png","cars_png","roads_png","num_users"])
        w.writerows(rows)
    print(f"[OK] geom_manifest.csv  -> {out_csv} (rows={len(rows)})")
    return out_csv

def build_tx_coords(multibs_csv: str, root_path: str, output_dir: str):
    need = set()
    with open(multibs_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        tx_cols = [h for h in r.fieldnames if h.startswith("tx")]
        for row in r:
            for h in tx_cols:
                if row[h] != "":
                    need.add((int(row["city_id"]), int(row[h])))
    mani_dir = pjoin(output_dir, "manifests"); ensure_dir(mani_dir)
    out_csv = pjoin(mani_dir, "tx_coords.csv")
    rows, miss = [], []
    for city, tx in sorted(need):
        path = pjoin(root_path, "antenna", f"{city}.json")
        if not os.path.exists(path):
            miss.append((city, tx, "antenna_json_missing")); continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not (0 <= tx < len(data)):
                miss.append((city, tx, f"tx_index_oob(len={len(data)})")); continue
            x, y = float(data[tx][0]), float(data[tx][1])
            rows.append([city, tx, x, y])
        except Exception as e:
            miss.append((city, tx, f"parse_fail:{e}"))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["city_id","tx_id","tx_x","tx_y"]); w.writerows(rows)
    print(f"[OK] tx_coords.csv      -> {out_csv} (rows={len(rows)})")
    if miss:
        print(f"[WARN] {len(miss)} tx entries failed (show first 10):")
        for it in miss[:10]: print("  ", it)
    return out_csv

# ===================== 特征 Schema / LOS / KPI =====================
def export_schema_json(path_json: str, norm_stats: Optional[dict] = None):
    schema = {
        "feature_version": FEATURE_VERSION,
        "in_dim": SCHEMA_IN_DIM,
        "base_channels": BASE_CHANNELS,
        "K_max": K_MAX,
        "note": "前5维通用 + 最多97个Tx派生特征；不足Tx用0填充，并在训练侧使用 tx_mask 屏蔽。",
    }
    if norm_stats is not None:
        schema["norm_stats"] = norm_stats
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"[OK] schema_v1.json 写出：{path_json}")

def _binary_roads_mask_to_weight_for_kpi(arr_u8: np.ndarray, mode="binary", dilate_r=1):
    m = (arr_u8 > 0).astype(np.float32)
    if mode == "binary": return m
    w = m.copy()
    for _ in range(max(1, int(dilate_r))):
        w = np.maximum.reduce([
            w,
            np.pad(w, ((1,0),(0,0)))[:-1, :],
            np.pad(w, ((0,1),(0,0)))[1:, :],
            np.pad(w, ((0,0),(1,0)))[:, :-1],
            np.pad(w, ((0,0),(0,1)))[:, 1:],
        ])
    return w

def load_roads_weight(geom_row: dict, shape_hw, root=ROOT_PATH):
    h, w = shape_hw
    rpath = geom_row.get("roads_png", "") if geom_row else ""
    if rpath and os.path.exists(pjoin(root, rpath)):
        im = Image.open(pjoin(root, rpath)).convert("L").resize((w, h), Image.NEAREST)
        arr = np.asarray(im, dtype=np.uint8)
        weight = _binary_roads_mask_to_weight_for_kpi(arr, mode=UE_WEIGHT_MODE, dilate_r=UE_DILATE_RADIUS).astype(np.float32)
        cov = float((weight > 0).mean())
        if cov < UE_MIN_COVERAGE or cov > UE_MAX_COVERAGE:
            return np.ones((h, w), dtype=np.float32)
        m = weight.mean()
        return weight / m if m > 1e-6 else np.ones((h, w), dtype=np.float32)
    return np.ones((h, w), dtype=np.float32)

def load_cars_mask(geom_row: dict, shape_hw, root=ROOT_PATH, dilate_r=1):
    h, w = shape_hw
    cpath = geom_row.get("cars_png", "") if geom_row else ""
    if cpath and os.path.exists(pjoin(root, cpath)):
        im = Image.open(pjoin(root, cpath)).convert("L").resize((w, h), Image.NEAREST)
        arr = (np.asarray(im, dtype=np.uint8) > 0).astype(np.uint8)
        m = arr
        for _ in range(max(1, int(dilate_r))):
            m = np.maximum.reduce([
                m,
                np.pad(m, ((1,0),(0,0)))[:-1, :],
                np.pad(m, ((0,1),(0,0)))[1:, :],
                np.pad(m, ((0,0),(1,0)))[:, :-1],
                np.pad(m, ((0,0),(0,1)))[:, 1:],
            ])
        return m.astype(np.float32)
    return np.zeros((h, w), dtype=np.float32)

def build_features(city_id: int, geom_row: dict, tx_coords_df: pd.DataFrame,
                   target_shape: Tuple[int,int], root_path: str,
                   pad_to: int = SCHEMA_IN_DIM, k_max: int = K_MAX) -> Tuple[np.ndarray, np.ndarray]:
    """统一 102维特征；返回 X[N,102] 与 tx_mask[97]。"""
    H, W = target_shape
    N = H * W

    buildings_path = pjoin(root_path, geom_row.get("buildings_png", ""))
    roads_path     = pjoin(root_path, geom_row.get("roads_png", ""))
    cars_path      = pjoin(root_path, geom_row.get("cars_png", ""))

    buildings_img = Image.open(buildings_path).convert("L").resize((W, H), Image.NEAREST)
    roads_img     = Image.open(roads_path).convert("L").resize((W, H), Image.NEAREST) if roads_path else Image.new("L", (W, H), 0)
    cars_img      = Image.open(cars_path).convert("L").resize((W, H), Image.NEAREST) if cars_path else Image.new("L", (W, H), 0)

    buildings_arr = np.asarray(buildings_img, dtype=np.float32) / 255.0
    roads_arr     = np.asarray(roads_img,     dtype=np.float32) / 255.0
    cars_arr      = np.asarray(cars_img,      dtype=np.float32) / 255.0

    x_coords, y_coords = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x_norm = x_coords / max(1.0, float(W))
    y_norm = y_coords / max(1.0, float(H))

    tx_city = tx_coords_df[tx_coords_df["city_id"] == city_id]
    tx_x = tx_city["tx_x"].values
    tx_y = tx_city["tx_y"].values
    K = len(tx_x)

    X = np.zeros((N, pad_to), dtype=np.float32)
    X[:, 0] = x_norm.flatten()
    X[:, 1] = y_norm.flatten()
    X[:, 2] = buildings_arr.flatten()
    X[:, 3] = roads_arr.flatten()
    X[:, 4] = cars_arr.flatten()

    k_use = min(K, k_max)
    for k_idx in range(k_use):
        tx_x_n = tx_x[k_idx] / max(1.0, float(W))
        tx_y_n = tx_y[k_idx] / max(1.0, float(H))
        dist = np.sqrt((x_norm - tx_x_n) ** 2 + (y_norm - tx_y_n) ** 2)
        X[:, 5 + k_idx] = dist.flatten()

    tx_mask = np.zeros((k_max,), dtype=np.float32)
    tx_mask[:k_use] = 1.0
    return X, tx_mask

# -------- LOS 重建（几何投射） --------
def _bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0); dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x += sx
        if e2 <= dx: err += dx; y += sy
    return points

def _line_of_sight(binary_obstacle: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    h, w = binary_obstacle.shape
    pts = _bresenham_line(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    hit = 0
    for (x, y) in pts:
        if 0 <= x < w and 0 <= y < h and binary_obstacle[y, x] > 0:
            hit += 1
    if hit == 0: return 1.0
    return max(0.0, 1.0 - 0.9 * hit / len(pts))

def build_los_map(city_id: int, model_dir_on_disk: str, geom_row: dict,
                  tx_coords_sel_df: pd.DataFrame, target_shape: Tuple[int,int],
                  root_path: str, smooth_sigma: float = 1.0, stride: int = 2) -> np.ndarray:
    H, W = target_shape
    buildings_path = pjoin(root_path, geom_row.get("buildings_png", ""))
    tx_x = tx_coords_sel_df["tx_x"].values if len(tx_coords_sel_df) > 0 else np.array([])
    tx_y = tx_coords_sel_df["tx_y"].values if len(tx_coords_sel_df) > 0 else np.array([])
    K = len(tx_x)

    if not file_exists(buildings_path):
        print(f"[WARN][LOS] city={city_id}: buildings_png missing -> uniform CONF_NLOS")
        return (CONF_NLOS * np.ones((H, W, K), dtype=np.float32))

    if K == 0:
        return np.zeros((H, W, 0), dtype=np.float32)

    bimg = Image.open(buildings_path).convert("L").resize((W, H), Image.NEAREST)
    bbin = (np.asarray(bimg, dtype=np.uint8) > 0).astype(np.uint8)

    if stride > 1:
        hh, ww = (H + stride - 1) // stride, (W + stride - 1) // stride
    else:
        hh, ww = H, W

    xs, ys = np.meshgrid(np.arange(ww), np.arange(hh))
    xs_full = np.clip(xs * stride, 0, W - 1)
    ys_full = np.clip(ys * stride, 0, H - 1)

    los_small = np.zeros((hh, ww, K), dtype=np.float32)
    for k in range(K):
        tx_x_pix = int(round(tx_x[k])); tx_y_pix = int(round(tx_y[k]))
        for yy in range(hh):
            for xx in range(ww):
                px = int(xs_full[yy, xx]); py = int(ys_full[yy, xx])
                los_small[yy, xx, k] = _line_of_sight(bbin, tx_x_pix, tx_y_pix, px, py)

    if stride > 1:
        los_up = np.zeros((H, W, K), dtype=np.float32)
        for k in range(K):
            im = Image.fromarray((los_small[..., k] * 255.0).astype(np.uint8), mode="L")
            im = im.resize((W, H), Image.NEAREST)
            los_up[..., k] = np.asarray(im, dtype=np.uint8) / 255.0
    else:
        los_up = los_small

    if smooth_sigma and smooth_sigma > 0:
        for k in range(K):
            im = Image.fromarray((los_up[..., k] * 255.0).astype(np.uint8), mode="L")
            im = im.filter(ImageFilter.GaussianBlur(radius=float(smooth_sigma)))
            los_up[..., k] = np.asarray(im, dtype=np.uint8) / 255.0

    return np.clip(los_up, 0.0, 1.0).astype(np.float32)

# -------- KPI 计算 --------
def strategy_weights(sinr_vec: np.ndarray, rate_vec: np.ndarray, los_stack: Optional[np.ndarray], mode: str):
    H, W, K = rate_vec.shape
    if mode == "max-sinr":
        idx = np.argmax(sinr_vec, axis=-1)
        w = np.zeros_like(rate_vec, dtype=np.float32)
        for k in range(K): w[..., k] = (idx == k).astype(np.float32)
        return w
    if mode == "prop-rate-alpha":
        base = np.power(np.maximum(rate_vec, 0.0) + SAFETY_EPS, ALPHA)
        denom = base.sum(axis=-1, keepdims=True) + SAFETY_EPS
        return base / denom
    if mode == "ua-rate-alpha-beta":
        base = np.power(np.maximum(rate_vec, 0.0) + SAFETY_EPS, ALPHA)
        if los_stack is not None:
            if los_stack.shape[-1] != K:
                raise ValueError(f"LOS K mismatch: {los_stack.shape[-1]} vs rate K={K}")
            conf = los_stack.astype(np.float32) + (1.0 - los_stack.astype(np.float32)) * CONF_NLOS
        else:
            conf = CONF_NLOS * np.ones_like(base, dtype=np.float32)
        base *= np.power(conf, BETA)
        denom = base.sum(axis=-1, keepdims=True) + SAFETY_EPS
        return base / denom
    raise ValueError(f"Unknown strategy: {mode}")

def calculate_kpi_from_dB_and_config(dB_stack: np.ndarray, los_stack: Optional[np.ndarray],
                                     geom_row: dict, city_id: int, model_dir_on_disk: str,
                                     strategies: List[str] = ["max-sinr","prop-rate-alpha","ua-rate-alpha-beta"]):
    H, W, K = dB_stack.shape
    if (los_stack is not None) and (los_stack.shape != dB_stack.shape):
        raise ValueError(f"[{city_id}_{model_dir_on_disk}] dB {dB_stack.shape} vs LOS {los_stack.shape} mismatch.")

    canon = canonical_model_for_ranges(model_dir_on_disk)
    is_gain_model = MODEL_DB_RANGES.get(canon, (IS_GAIN_FALLBACK, GAIN_DB_MIN, GAIN_DB_MAX))[0]

    pr_dbm = (TX_POWER_DBM + dB_stack) if is_gain_model else (TX_POWER_DBM - dB_stack)
    pr_mw  = np.power(10.0, pr_dbm / 10.0, dtype=np.float32)

    interf_mw = INTERF_COEFF * (pr_mw.sum(axis=-1, keepdims=True) - pr_mw)
    noise_mw  = np.power(10.0, noise_dbm(B_HZ) / 10.0, dtype=np.float32)

    sinr_vec = pr_mw / (interf_mw + noise_mw + SAFETY_EPS)
    rate_vec = B_HZ * np.log2(1.0 + np.maximum(sinr_vec, 0.0))

    ue_w = load_roads_weight(geom_row, (H, W))
    um = ue_w[..., None].astype(np.float32)
    cars_mask = load_cars_mask(geom_row, (H, W))
    S_bits = AVG_FILE_SIZE_MB * 8e6
    lambda_u = REQ_RATE_PER_USER

    kpi_labels = {}
    for mode in strategies:
        w_base = strategy_weights(sinr_vec, rate_vec, los_stack, mode)
        if mode == "ua-rate-alpha-beta" and (cars_mask is not None) and (los_stack is not None):
            nlos = (1.0 - los_stack.astype(np.float32))
            atten = 1.0 - 0.4 * cars_mask[..., None]
            w = w_base * (1.0 - nlos + nlos * np.clip(atten, 0.0, 1.0))
            w = w / (w.sum(axis=-1, keepdims=True) + SAFETY_EPS)
        else:
            w = w_base

        user_thr = ((w * rate_vec) * um).sum(axis=-1)
        sys_thr  = float(user_thr.sum())
        avg_thr  = float(user_thr.sum() / (ue_w.sum() + SAFETY_EPS))

        L_k = (w * um).sum(axis=(0, 1))
        jfi = (L_k.sum() ** 2) / (K * np.square(L_k).sum() + SAFETY_EPS); jfi = float(jfi)

        R_alloc_k = (w * rate_vec * um).sum(axis=(0, 1))
        mu_k = R_alloc_k / (S_bits + SAFETY_EPS)
        lambda_k = lambda_u * L_k
        rho_k = np.where(mu_k > SAFETY_EPS, lambda_k / (mu_k + SAFETY_EPS), np.inf)
        rho_k = np.clip(rho_k, 0.0, UTIL_CAP)
        mu_eff = lambda_k / np.maximum(rho_k, SAFETY_EPS)
        denom = np.maximum(mu_eff - lambda_k, SAFETY_EPS)
        D_k = 1.0 / denom
        D_k = np.where(np.isfinite(D_k), D_k, DELAY_CAP_S)
        D_k = np.clip(D_k, 0.0, DELAY_CAP_S)

        wsum = (w * um).sum(axis=-1) + SAFETY_EPS
        D_u = (w * um * D_k.reshape(1, 1, K)).sum(axis=-1) / wsum
        avg_delay = float((D_u * ue_w).sum() / (ue_w.sum() + SAFETY_EPS))

        a, b, c = 1.0, 0.1, 0.0
        q_u = a * np.log(np.maximum(user_thr, SAFETY_EPS)) - b * D_u + c
        avg_qoe = float((q_u * ue_w).sum() / (ue_w.sum() + SAFETY_EPS))

        kpi_labels[mode] = {
            "sys_throughput_Mbps": float(sys_thr / 1e6),
            "avg_user_rate_Mbps":  float(avg_thr / 1e6),
            "jfi": jfi,
            "avg_delay_s": avg_delay,
            "avg_qoe": avg_qoe
        }
    return kpi_labels

# ===================== 数据落盘（PL/KPI/LOS/归一化） =====================
def prepare_federated_datasets(multibs_csv: str, geom_manifest_csv: str, tx_coords_csv: str,
                               root_path: str, output_dir: str, test_ratio: float = 0.2,
                               los_stride: int = 2, los_sigma: float = 1.0):
    ensure_dir(output_dir)
    users_dir = pjoin(output_dir, "users"); ensure_dir(users_dir)
    users_kpi_dir = pjoin(output_dir, "users_kpi"); ensure_dir(users_kpi_dir)
    norm_dir = pjoin(output_dir, "normalization_params"); ensure_dir(norm_dir)
    los_root_out = pjoin(output_dir, "los"); ensure_dir(los_root_out)

    multibs_df = pd.read_csv(multibs_csv)
    geom_df = pd.read_csv(geom_manifest_csv)
    tx_coords_df = pd.read_csv(tx_coords_csv)
    geom_map = geom_df.set_index('city_id').to_dict('index')

    export_schema_json(pjoin(output_dir, "schema_v1.json"))

    for city_id in multibs_df['city_id'].unique():
        print(f"[City {city_id}] 准备数据...")
        city_models_df = multibs_df[multibs_df['city_id'] == city_id]
        geom_row = geom_map.get(str(city_id)) or geom_map.get(int(city_id))
        if not geom_row:
            print(f"[SKIP][CITY {city_id}] geom_manifest 缺该城市，整城跳过。")
            continue

        ref_png = pjoin(root_path, geom_row.get("buildings_png", ""))
        if not (ref_png and os.path.exists(ref_png)):
            print(f"[SKIP][CITY {city_id}] buildings_png 缺失（{ref_png}），整城跳过。")
            continue
        W, H = Image.open(ref_png).size

        X, tx_mask = build_features(city_id, geom_row, tx_coords_df, (H, W), root_path,
                                    pad_to=SCHEMA_IN_DIM, k_max=K_MAX)
        N = X.shape[0]
        n_test = int(N * test_ratio)
        idx = np.random.permutation(N)
        te_idx = idx[:n_test]; tr_idx = idx[n_test:]

        for _, row in city_models_df.iterrows():
            logical_model = row['model_dir']
            disk_model = resolve_model_dir_name(root_path, logical_model)
            if disk_model is None:
                print(f"  [WARN] model {logical_model} has no gain folder; skip.")
                continue
            print(f"  - 处理模型 {logical_model} -> '{disk_model}'")

            # 取 TX 列表
            tx_ids = []
            for k in range(1, 81):
                key = f"tx{k}"
                if key in row and not pd.isna(row[key]) and row[key] != "":
                    tx_ids.append(int(row[key]))
                else:
                    break
            K = len(tx_ids)
            # 仍写一个空 LOS 以标记触发过
            if K == 0:
                print(f"    [WARN] no TX in manifest; skip this pair.")
                model_los_dir = pjoin(los_root_out, disk_model); ensure_dir(model_los_dir)
                los_path_out = pjoin(model_los_dir, f"{city_id}_LOS.npy")
                np.save(los_path_out, np.zeros((H, W, 0), dtype=np.float32))
                # 镜像到别名目录
                for alias in LOS_ALIAS_MIRRORS.get(logical_model, []):
                    alias_dir = pjoin(los_root_out, alias); ensure_dir(alias_dir)
                    np.save(pjoin(alias_dir, f"{city_id}_LOS.npy"), np.zeros((H, W, 0), dtype=np.float32))
                print(f"    [OK][LOS] 保存（空通道K=0）：{los_path_out}")
                continue

            # 读取 dB 标签（确保使用磁盘真实目录）
            dB_maps = []
            for k_idx in range(K):
                pl_key = f"pl_path{k_idx+1}"
                rel = row[pl_key] if (pl_key in row and isinstance(row[pl_key], str)) else ""
                if rel.startswith("gain/"):
                    parts = rel.replace("\\", "/").split("/")
                    if len(parts) >= 3:
                        parts[1] = disk_model  # 强制替换为真实目录名
                        rel = "/".join(parts)
                abs_path = pjoin(root_path, rel) if rel else ""
                if abs_path and os.path.exists(abs_path):
                    db_map, _ = read_gray_to_db_with_model(abs_path, disk_model)
                    dB_maps.append(db_map.astype(np.float32))
                else:
                    print(f"    [WARN] dB not found: {abs_path or rel}; use zeros.")
                    dB_maps.append(np.zeros((H, W), dtype=np.float32))
            Y_full = np.stack(dB_maps, axis=-1).astype(np.float32)   # [H,W,K]
            Y_flat = Y_full.reshape(-1, K)

            # 切分 & 保存 PL
            Xtr, Ytr = X[tr_idx], Y_flat[tr_idx]
            Xte, Yte = X[te_idx], Y_flat[te_idx]

            y_mean = Ytr.mean(axis=0); y_std = Ytr.std(axis=0) + 1e-8
            norm_file = pjoin(norm_dir, f"city_{city_id}_model_{logical_model}_mean_std.npz")
            np.savez_compressed(norm_file, y_mean=y_mean, y_std=y_std)
            print(f"    -> norm params：{norm_file} (K={K})")

            pl_path = pjoin(users_dir, f"user_{city_id}_{logical_model}.npz")
            np.savez_compressed(pl_path, Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, tx_mask=tx_mask)
            print(f"    -> PL数据：{pl_path}  (tr:{Xtr.shape}, te:{Xte.shape})")

            # ===== LOS：必生成 + 别名镜像 =====
            # 严格按 manifest 的 tx_ids 顺序抓坐标
            tx_all = pd.read_csv(tx_coords_csv)
            sel_rows = []
            for tx in tx_ids:
                tmp = tx_all[(tx_all["city_id"] == city_id) & (tx_all["tx_id"] == tx)]
                if len(tmp) == 1:
                    sel_rows.append(tmp.iloc[0])
                else:
                    print(f"    [WARN][LOS] tx {tx} 在 tx_coords.csv 中缺失或不唯一，将被忽略")
            tx_sel_df = (pd.DataFrame(sel_rows)
                         if len(sel_rows) > 0
                         else pd.DataFrame(columns=["city_id", "tx_id", "tx_x", "tx_y"]))

            los_stack = build_los_map(city_id, disk_model, geom_row, tx_sel_df, (H, W), root_path,
                                      smooth_sigma=los_sigma, stride=los_stride)

            # 保存到 canonical 目录
            model_los_dir = pjoin(los_root_out, disk_model); ensure_dir(model_los_dir)
            los_path_out = pjoin(model_los_dir, f"{city_id}_LOS.npy")
            np.save(los_path_out, los_stack.astype(np.float32))
            print(f"    [OK][LOS] 保存：{los_path_out} (shape={'x'.join(map(str, los_stack.shape))})")

            # 镜像到别名目录（如 carslRT2/carslRT4）
            for alias in LOS_ALIAS_MIRRORS.get(logical_model, []):
                alias_dir = pjoin(los_root_out, alias); ensure_dir(alias_dir)
                alias_path = pjoin(alias_dir, f"{city_id}_LOS.npy")
                np.save(alias_path, los_stack.astype(np.float32))
                print(f"    [OK][LOS] 镜像：{alias_path}")

            # ===== KPI：始终生成（3×5=15） =====
            print(f"    -> 计算 KPI(15维)...")
            kpi = calculate_kpi_from_dB_and_config(Y_full, los_stack, geom_row, city_id, disk_model)
            order = ["max-sinr", "prop-rate-alpha", "ua-rate-alpha-beta"]
            vec = []
            for s in order:
                m = kpi[s]
                vec.extend([m["sys_throughput_Mbps"], m["avg_user_rate_Mbps"], m["jfi"], m["avg_delay_s"], m["avg_qoe"]])
            Yk = np.array(vec, dtype=np.float32)[None, :]    # [1,15]
            Yk_full = np.repeat(Yk, N, axis=0)
            Yk_tr = Yk_full[tr_idx]; Yk_te = Yk_full[te_idx]
            kpi_path = pjoin(users_kpi_dir, f"user_{city_id}_{logical_model}_kpi.npz")
            np.savez_compressed(kpi_path, Xtr=Xtr, Ytr=Yk_tr, Xte=Xte, Yte=Yk_te, tx_mask=tx_mask)
            print(f"    -> KPI数据：{kpi_path}  (tr:{Yk_tr.shape}, te:{Yk_te.shape})")

# ===================== 主程序 =====================
def main():
    global B_HZ, TX_POWER_DBM, NOISE_PSD_DBM_PER_HZ, NOISE_FIGURE_DB, INTERF_COEFF
    global ALPHA, BETA, CONF_NLOS, REQ_RATE_PER_USER, AVG_FILE_SIZE_MB
    global UTIL_CAP, DELAY_CAP_S, UE_WEIGHT_MODE, UE_DILATE_RADIUS
    parser = argparse.ArgumentParser(description="Dataset Preparation (Schema + Geometric LOS + KPI + Non-IID)")
    parser.add_argument('--config', type=str, default='configs/dataset.yaml')
    parser.add_argument('--root_path', type=str, default=ROOT_PATH)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--total_cities', type=int, default=TOTAL_CITIES)
    parser.add_argument('--k_core', type=int, default=K_CORE)
    parser.add_argument('--extra_k32', type=int, default=EXTRA_K32_CITIES)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--core_models', nargs='+', default=CORE_MODELS)
    parser.add_argument('--ablation_models', nargs='+', default=ABLATION_MODELS)
    parser.add_argument('--los_stride', type=int, default=2, help="LOS 投射下采样步长(>=1)")
    parser.add_argument('--los_sigma', type=float, default=1.0, help="LOS 平滑半径")
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--bandwidth_hz', type=float, default=B_HZ)
    parser.add_argument('--tx_power_dbm', type=float, default=TX_POWER_DBM)
    parser.add_argument('--noise_psd_dbm_per_hz', type=float, default=NOISE_PSD_DBM_PER_HZ)
    parser.add_argument('--noise_figure_db', type=float, default=NOISE_FIGURE_DB)
    parser.add_argument('--interference_coeff', type=float, default=INTERF_COEFF)
    parser.add_argument('--alpha', type=float, default=ALPHA)
    parser.add_argument('--beta', type=float, default=BETA)
    parser.add_argument('--conf_nlos', type=float, default=CONF_NLOS)
    parser.add_argument('--req_rate_per_user_mbps', type=float, default=REQ_RATE_PER_USER)
    parser.add_argument('--avg_file_size_mb', type=float, default=AVG_FILE_SIZE_MB)
    parser.add_argument('--util_cap', type=float, default=UTIL_CAP)
    parser.add_argument('--delay_cap_s', type=float, default=DELAY_CAP_S)
    parser.add_argument('--ue_weight_mode', choices=['binary', 'dilate'], default=UE_WEIGHT_MODE)
    parser.add_argument('--ue_dilate_radius', type=int, default=UE_DILATE_RADIUS)
    args = parse_args_with_config(parser)

    B_HZ = args.bandwidth_hz
    TX_POWER_DBM = args.tx_power_dbm
    NOISE_PSD_DBM_PER_HZ = args.noise_psd_dbm_per_hz
    NOISE_FIGURE_DB = args.noise_figure_db
    INTERF_COEFF = args.interference_coeff
    ALPHA = args.alpha
    BETA = args.beta
    CONF_NLOS = args.conf_nlos
    REQ_RATE_PER_USER = args.req_rate_per_user_mbps
    AVG_FILE_SIZE_MB = args.avg_file_size_mb
    UTIL_CAP = args.util_cap
    DELAY_CAP_S = args.delay_cap_s
    UE_WEIGHT_MODE = args.ue_weight_mode
    UE_DILATE_RADIUS = args.ue_dilate_radius

    ensure_dir(args.output_dir); ensure_dir(pjoin(args.output_dir, "sanity_report"))
    print("=== Dataset Preparation start ===")
    print(f"[ARGS] root_path={args.root_path}")
    print(f"[ARGS] output_dir={args.output_dir}")
    print(f"[FLAGS] (LOS=always-on, KPI=always-on, Non-IID labeling=on)")

    multibs_csv, cities_used, plan, noniid_csv = select_subset_and_write_plan(
        total_cities=args.total_cities, k_core=args.k_core, extra_k32=args.extra_k32,
        core_models=args.core_models, ablation_models=args.ablation_models,
        root_path=args.root_path, output_dir=args.output_dir, random_seed=args.random_seed
    )

    geom_csv = build_geom_manifest(cities_used, args.root_path, args.output_dir)
    tx_coords_csv = build_tx_coords(multibs_csv, args.root_path, args.output_dir)

    prepare_federated_datasets(multibs_csv, geom_csv, tx_coords_csv,
                               args.root_path, args.output_dir,
                               test_ratio=float(args.test_ratio),
                               los_stride=max(1, int(args.los_stride)),
                               los_sigma=float(args.los_sigma))

    # --- 校验：LOS 是否至少有一个 .npy ---
    los_root = pjoin(args.output_dir, "los")
    generated = []
    for root, _, files in os.walk(los_root):
        for fn in files:
            if fn.lower().endswith(".npy"):
                generated.append(pjoin(root, fn))
    if not generated:
        raise RuntimeError(
            "LOS 生成校验未通过：los/ 目录下没有任何 .npy 文件。\n"
            "常见原因：1) multibs_manifest 无有效 city-model；2) tx_coords 与 manifest 中 tx* 不匹配；"
            "3) buildings_png 缺失导致整城跳过。\n"
            f"请查看 {pjoin(args.output_dir,'sanity_report','summary.csv')}、{pjoin(args.output_dir,'manifests','noniid_manifest.csv')} 与控制台 [SKIP][CITY]/[WARN][LOS] 日志。"
        )
    else:
        print(f"[CHECK][LOS] OK，共生成 {len(generated)} 个 LOS 文件。示例：{generated[0]}")

    print("=== Dataset Preparation complete ===")
    print(f"Manifests:          {pjoin(args.output_dir, 'manifests')}")
    print(f"  - multibs_manifest.csv")
    print(f"  - geom_manifest.csv (含 num_users)")
    print(f"  - noniid_manifest.csv (light/medium/heavy)")
    print(f"Users (PL):         {pjoin(args.output_dir, 'users')}")
    print(f"Users_KPI:          {pjoin(args.output_dir, 'users_kpi')}")
    print(f"Norm params:        {pjoin(args.output_dir, 'normalization_params')}")
    print(f"LOS (per model):    {pjoin(args.output_dir, 'los')}  # 已镜像别名目录")
    print(f"Schema:             {pjoin(args.output_dir, 'schema_v1.json')}")
    print(f"Subset plan:        {pjoin(args.output_dir, 'subset_plan.json')}")
    print(f"Sanity report:      {pjoin(args.output_dir, 'sanity_report')}")

if __name__ == "__main__":
    main()
