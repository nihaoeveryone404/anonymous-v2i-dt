# -*- coding: utf-8 -*-
"""
buzhou2_pfl_fixed.py —— Personalized FL（支持动态K值 & KPI预测）

仅联邦训练与评估，不做预测文件导出（第三步另行处理）。

更新要点：
- 保留：动态输入/输出维度、个性化聚合（backbone 聚合、head 本地）、通信压缩（Top-K+EF+量化）、EMA、详细评估
- 加速：CUDA AMP（--amp，默认开启）、cudnn.benchmark、pin_memory
- 修复：quant_bits 默认值与 choices 不一致的问题（默认改为 8）
- 消除 FutureWarning：统一使用 torch.amp.autocast('cuda', ...) 与 amp.GradScaler('cuda', ...)
- 场景与训练超参数由 YAML 配置；命令行参数可覆盖 YAML。
"""

import os, json, random, csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import amp

from src.models.cpfl_models import PFL_KPIPredictor, PFL_REMNet
from src.utils.io_utils import parse_args_with_config

SCENARIO_CHOICES = ["All", "Light", "Medium", "Heavy"]

# ========================= 默认路径 / 超参 =========================
USERS_DIR_DEFAULT = "outputs/dataset/users"
USERS_KPI_DIR_DEFAULT = "outputs/dataset/users_kpi"
# 自动推断 dataset_preparation_output 根目录（= users_dir 的上一级）
DATASET_PREP_OUTPUT_DIR_DEFAULT = os.path.dirname(USERS_DIR_DEFAULT)
SAVE_DIR_DEFAULT = "outputs/checkpoints/pl"
INIT_CKPT_DEFAULT = ""  # 置空表示从零开始

TASK_TYPE_DEFAULT = "pl"  # "pl" 或 "kpi"

ROUNDS_DEFAULT = 400      # 保持（JSON 未指定）
LOCAL_EPOCHS_DEFAULT = 3  # 来自 JSON "local_epochs"
LR_DEFAULT = 0.00026352989957644847  # 来自 JSON "lr"
BATCH_SIZE_DEFAULT = 128  # 来自 JSON "batch_size"

SEED_DEFAULT = 2025
PERSONALIZED_DEFAULT = True

LOSS_DEFAULT = "huber"          # 来自 JSON "loss"
HUBER_DELTA_DEFAULT = 0.5678308362384664  # 来自 JSON "huber_delta"（huber 损失时生效）

HEAD_LR_MULT_DEFAULT = 1.4257782865726316   # 来自 JSON "head_lr_mult"
TWO_LAYER_HEAD_DEF = True                   # 来自 JSON "two_layer_head"
HEAD_DROPOUT_DEFAULT = 0.25063511246027054  # 来自 JSON "head_dropout"

COMPRESS_DEFAULT = "sparse8"                # 来自 JSON "compress"
UPLOAD_RATIO_DEFAULT = 0.3709044684802143   # 来自 JSON "upload_ratio"
QUANT_BITS_DEFAULT = 8                      # 来自 JSON "quant_bits"
LOCAL_SYNC_DEFAULT = 1                      # 来自 JSON "local_sync"

LOG_COMM_DEFAULT = True
EVAL_USE_GLOBAL_HEAD_DEFAULT = False
CLIP_GRAD_DEFAULT = 1.8785657134034506      # 来自 JSON "clip_grad"

USE_EMA_DEFAULT = False                     # 来自 JSON "use_ema"
EMA_DECAY_DEFAULT = 0.9844872387387853      # 来自 JSON "ema_decay"

# KPI 任务
KPI_LR_MULT_DEFAULT = 1.7651878818775586        # 来自 JSON "kpi_lr_mult"
KPI_HEAD_DROPOUT_DEFAULT = 0.12075679951163695  # 来自 JSON "kpi_head_dropout"
KPI_HIDDEN_DIM_DEFAULT = 384                    # 来自 JSON "kpi_hidden_dim"
KPI_NUM_OUTPUTS = 15

# 训练/评估 AMP
AMP_DEFAULT = True


# ========================= 工具 =========================
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def bytes_to_mb(x: int) -> float: return float(x) / (1024.0 * 1024.0)

# ========== 文件名/清单辅助 ==========
def _parse_user_filename(fname: str, task: str) -> Optional[Tuple[int, str]]:
    """解析 user_239_carsIRT2.npz 或 user_239_carsIRT2_kpi.npz -> (city, model)"""
    if not (fname.startswith("user_") and fname.endswith(".npz")):
        return None
    name = fname[:-4]
    parts = name.split("_")
    if task == "kpi":
        if len(parts) < 4 or parts[-1].lower() != "kpi":
            return None
        try:
            city = int(parts[1]); model = "_".join(parts[2:-1]); return city, model
        except:
            return None
    else:
        if len(parts) < 3:
            return None
        try:
            city = int(parts[1]); model = "_".join(parts[2:]); return city, model
        except:
            return None

def _infer_dataset_root(users_like_dir: str) -> str:
    return os.path.dirname(users_like_dir.rstrip("\\/")) if users_like_dir else DATASET_PREP_OUTPUT_DIR_DEFAULT

def _load_allowed_cities_by_scenario(dataset_prep_output_dir: str, scenario: str) -> Optional[set]:
    """用 csv 标准库读取 manifests/noniid_manifest.csv，按 regime==scenario 过滤 city_id 集合"""
    if scenario.lower() == "all":
        return None
    csvp = os.path.join(dataset_prep_output_dir, "manifests", "noniid_manifest.csv")
    if not os.path.isfile(csvp):
        print(f"[WARN] noniid_manifest.csv not found: {csvp}; fallback to All.")
        return None
    allowed = set()
    try:
        with open(csvp, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            lower = scenario.lower()
            for row in r:
                rid = row.get("regime", "") or row.get("Regime", "") or row.get("scene", "")
                cid = row.get("city_id", "") or row.get("city", "") or row.get("City", "")
                if str(rid).strip().lower() == lower:
                    try:
                        allowed.add(int(cid))
                    except:
                        continue
    except Exception as e:
        print(f"[WARN] read manifest failed ({e}); fallback to All.")
        return None
    if not allowed:
        print(f"[WARN] no cities for scenario={scenario}; fallback to All.")
        return None
    return allowed

# ========================= 指标 =========================
def _rank_average_numpy(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64); i = 0; cur = 1
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]: j += 1
        avg_rank = (cur + (cur + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank; cur += (j - i); i = j
    return ranks
def spearman_rho_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().float().cpu().numpy()
    targ_np = target.detach().float().cpu().numpy()
    C = pred_np.shape[1]
    if C == 0: return 0.0
    rhos = []
    for j in range(C):
        x = pred_np[:, j]; y = targ_np[:, j]
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            rhos.append(0.0)
        else:
            rx = _rank_average_numpy(x); ry = _rank_average_numpy(y)
            rxm = rx - rx.mean(); rym = ry - ry.mean()
            denom = np.sqrt((rxm ** 2).sum() * (rym ** 2).sum()) + 1e-12
            rhos.append(float((rxm * rym).sum() / denom))
    return float(np.mean(rhos))
def rmse_mae_np(pred_np: np.ndarray, targ_np: np.ndarray) -> Tuple[float, float]:
    diff = pred_np - targ_np
    rmse = float(np.sqrt((diff ** 2).mean()))
    mae = float(np.abs(diff).mean())
    return rmse, mae

# ========================= 数据（含归一化） =========================
class RadioMapDataset(Dataset):
    """
    X: (N, D)，前两维 (x,y) 做 per-client min-max -> [0,1]
    Y: (N, K) 做 per-client z-score；K 由数据决定
    KPI 数据集的 Y 每行相同（同一图的全局 KPI 向量），KPI 模型中做 GAP 后用第1行做监督
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 xy_min: Optional[np.ndarray] = None, xy_max: Optional[np.ndarray] = None,
                 y_mean: Optional[np.ndarray] = None, y_std: Optional[np.ndarray] = None):
        X = X.astype(np.float32); Y = Y.astype(np.float32)
        if xy_min is None or xy_max is None:
            xy_min = X[:, :2].min(axis=0); xy_max = X[:, :2].max(axis=0)
        scale = np.maximum(xy_max - xy_min, 1e-6)
        X[:, :2] = (X[:, :2] - xy_min) / scale
        if y_mean is None or y_std is None:
            y_mean = Y.mean(axis=0); y_std = Y.std(axis=0) + 1e-8
        Yn = (Y - y_mean) / y_std
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Yn).float()
        self.xy_min = torch.from_numpy(xy_min).float()
        self.xy_max = torch.from_numpy(xy_max).float()
        self.y_mean = torch.from_numpy(y_mean).float()
        self.y_std = torch.from_numpy(y_std).float()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

def load_users(users_dir: str, seed: int, task: str = "pl",
               allowed_cities: Optional[set] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    rng = np.random.RandomState(seed)
    files_all = sorted([f for f in os.listdir(users_dir) if f.startswith("user_") and f.endswith(".npz")])
    files = []
    for f in files_all:
        parsed = _parse_user_filename(f, task=task)
        if parsed is None:
            continue
        city, _ = parsed
        if (task == "kpi" and not f.endswith("_kpi.npz")):
            continue
        if (task == "pl" and f.endswith("_kpi.npz")):
            continue
        if (allowed_cities is not None) and (city not in allowed_cities):
            continue
        files.append(f)
    if not files:
        raise FileNotFoundError(f"No user files for task '{task}' after scenario filter under {users_dir}")
    splits = []
    for f in files:
        fp = os.path.join(users_dir, f)
        d = np.load(fp)
        if all(k in d for k in ["Xtr", "Ytr", "Xte", "Yte"]):
            Xtr, Ytr, Xte, Yte = d["Xtr"], d["Ytr"], d["Xte"], d["Yte"]
        else:
            raise ValueError(f"Bad file keys in {fp}")
        K_or_num = Ytr.shape[1] if Ytr.ndim > 1 else 1
        if task == "kpi" and K_or_num != KPI_NUM_OUTPUTS:
            print(f"[WARN] {f} kpi_dim={K_or_num}, expected={KPI_NUM_OUTPUTS}. 继续使用 {K_or_num}.")
        splits.append((Xtr.astype(np.float32), Ytr.astype(np.float32),
                       Xte.astype(np.float32), Yte.astype(np.float32), K_or_num))
    return splits

# ========================= 通信压缩（Top-K + EF + 量化） =========================
def per_key_topk_quantize(delta_flat: torch.Tensor,
                          ef_buf: Optional[torch.Tensor],
                          k_ratio: float,
                          quant_bits: int = 8):
    device = delta_flat.device
    if (ef_buf is None) or (ef_buf.numel() != delta_flat.numel()):
        ef_buf = torch.zeros_like(delta_flat, device=device)
    else:
        ef_buf = ef_buf.to(device)
    full = delta_flat + ef_buf
    n = full.numel(); k = max(1, int(n * k_ratio))
    topk = torch.topk(full.abs(), k, dim=0, largest=True, sorted=False)
    idx = topk.indices; nz_vals = full[idx]
    new_ef = full.clone(); new_ef[idx] = 0.0
    if quant_bits == 8:
        maxabs = torch.max(nz_vals.abs()); scale = (maxabs / 127.0 + 1e-12).float()
        q = torch.round(nz_vals / scale).to(torch.int8)
        est_bytes = int(idx.numel() * 4 + q.numel() * 1 + 4)
        pkt = {"idx": idx.detach().cpu().to(torch.int32), "val": q.detach().cpu(),
               "scale": float(scale.detach().cpu().item()), "bits": 8}
    else:
        vals32 = nz_vals.to(torch.float32)
        est_bytes = int(idx.numel() * 4 + vals32.numel() * 4)
        pkt = {"idx": idx.detach().cpu().to(torch.int32), "val": vals32.detach().cpu(),
               "scale": 1.0, "bits": 32}
    return pkt, new_ef.detach().cpu(), est_bytes

# ========================= 客户端 / 服务器 =========================
class Client:
    def __init__(self, cid: int,
                 Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, Yte: np.ndarray, k: int,
                 device: torch.device,
                 head_lr_mult: float = 3.0, batch_size: int = 256,
                 huber_delta: float = 1.0,
                 clip_grad: float = 1.0,
                 head_dropout: float = 0.10,
                 task: str = "pl",
                 kpi_lr_mult: float = KPI_LR_MULT_DEFAULT,
                 kpi_head_dropout: float = KPI_HEAD_DROPOUT_DEFAULT,
                 use_amp: bool = AMP_DEFAULT):
        self.cid = cid; self.k = k; self.device = device
        self.head_lr_mult = head_lr_mult
        self.kpi_lr_mult = kpi_lr_mult
        self.batch_size = batch_size
        self.huber_delta = huber_delta
        self.clip_grad = clip_grad
        self.head_dropout = head_dropout
        self.kpi_head_dropout = kpi_head_dropout
        self.task = task
        self.use_amp = use_amp
        # per-client norm
        xy_min = Xtr[:, :2].min(axis=0); xy_max = Xtr[:, :2].max(axis=0)
        y_mean = Ytr.mean(axis=0); y_std = Ytr.std(axis=0) + 1e-8
        self.tr_ds = RadioMapDataset(Xtr, Ytr, xy_min, xy_max, y_mean, y_std)
        self.te_ds = RadioMapDataset(Xte, Yte, xy_min, xy_max, y_mean, y_std)
        self.y_mean = self.tr_ds.y_mean.to(self.device)
        self.y_std = self.tr_ds.y_std.to(self.device)
        self.local_head_state = None
        self.ef_buffer: Dict[str, torch.Tensor] = {}

    def _loader(self, ds, shuffle=True):
        bs = self.batch_size if self.task == "pl" else len(ds)  # KPI：整图
        if self.task == "kpi" and bs > 200000:  # 防 OOM
            bs = 200000
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0,
                          pin_memory=(self.device.type == "cuda"))

    def _agg_keys(self, personalized: bool, state_keys: List[str], task: str) -> List[str]:
        if personalized:
            if task == "pl":
                return [k for k in state_keys if k.startswith("extractors.") or k.startswith("backbone.")]
            elif task == "kpi":
                return [k for k in state_keys if k.startswith("backbone.")]
        return list(state_keys)

    def _build_loss(self, loss_name: str):
        if loss_name == "huber": return nn.HuberLoss(delta=self.huber_delta)
        if loss_name == "mae":   return nn.L1Loss()
        return nn.MSELoss()

    def local_train_and_compress(self,
                                 global_model: nn.Module,
                                 loss_name: str,
                                 epochs: int,
                                 lr: float,
                                 personalized: bool,
                                 upload_ratio: float,
                                 local_sync: int,
                                 quant_bits: int,
                                 round_idx: int) -> Tuple[Dict[str, dict], int]:
        scaler = amp.GradScaler('cuda', enabled=self.use_amp and (self.device.type == "cuda"))

        # 构造本地模型并加载全局权重
        if self.task == "pl":
            assert isinstance(global_model, PFL_REMNet)
            global_model.update_head_for_k(self.k)
            model = PFL_REMNet(input_dim=self.tr_ds.X.shape[1], initial_k=self.k,
                               two_layer_head=bool(getattr(global_model, "two_layer_head", False)),
                               head_dropout=self.head_dropout).to(self.device)
            g = global_model.state_dict(); m = model.state_dict()
            for k_key in g.keys():
                if k_key.startswith("extractors.") or k_key.startswith("backbone."):
                    if k_key in m: m[k_key].copy_(g[k_key].to(self.device))
            if personalized and (self.local_head_state is not None):
                for k_key, v in self.local_head_state.items():
                    if k_key in m and k_key.startswith("head."):
                        m[k_key].copy_(v.to(self.device))
            model.load_state_dict(m, strict=True)
            params = [
                {"params": list(model.extractors.parameters()) + list(model.backbone.parameters()), "lr": lr},
                {"params": model.head.parameters(), "lr": lr * self.head_lr_mult},
            ]
        else:
            assert isinstance(global_model, PFL_KPIPredictor)
            model = PFL_KPIPredictor(input_dim=self.tr_ds.X.shape[1],
                                     initial_kpi_outputs=self.k,
                                     hidden_dim=KPI_HIDDEN_DIM_DEFAULT,
                                     dropout=self.kpi_head_dropout).to(self.device)
            g = global_model.state_dict(); m = model.state_dict()
            for k_key in g.keys():
                if k_key.startswith("backbone.") and k_key in m:
                    m[k_key].copy_(g[k_key].to(self.device))
            if personalized and (self.local_head_state is not None):
                for k_key, v in self.local_head_state.items():
                    if k_key in m and k_key.startswith("head."):
                        m[k_key].copy_(v.to(self.device))
            model.load_state_dict(m, strict=True)
            params = [
                {"params": list(model.backbone.parameters()), "lr": lr},
                {"params": list(model.head.parameters()), "lr": lr * self.kpi_lr_mult},
            ]

        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        loss_fn = self._build_loss(loss_name)

        # 训练
        model.train()
        for _ in range(epochs):
            for xb, yb in self._loader(self.tr_ds, shuffle=True):
                xb = xb.to(self.device, non_blocking=True); yb = yb.to(self.device, non_blocking=True)
                with amp.autocast('cuda', enabled=self.use_amp and (self.device.type == "cuda")):
                    if self.task == "kpi":
                        target = yb[0:1, :]  # (1, 15)
                        pred = model(xb)     # (1, 15)
                        loss = loss_fn(pred, target)
                    else:
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                opt.zero_grad()
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if self.clip_grad and self.clip_grad > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad and self.clip_grad > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                    opt.step()

        # 缓存本地 head
        self.local_head_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith("head.")}

        # 非上传轮，跳过通信
        if (round_idx % local_sync) != 0: return {}, 0

        # 打包上传（稀疏/量化或全量）
        agg_keys = self._agg_keys(personalized, list(model.state_dict().keys()), self.task)
        packets: Dict[str, dict] = {}; bytes_up_total = 0
        g = global_model.state_dict()
        compress_mode = getattr(global_model, "_compress_mode", "off")
        fulldense = (upload_ratio >= 1.0 and quant_bits == 32 and compress_mode == "off")
        for k in agg_keys:
            local_param = model.state_dict()[k]
            global_param = g[k]
            if fulldense:
                delta = (local_param - global_param).detach().cpu().float().contiguous()
                packets[k] = {"type": "dense", "val": delta}
                bytes_up_total += int(delta.numel() * 4)
            else:
                delta = (local_param - global_param).detach().to(self.device).view(-1)
                ef = self.ef_buffer.get(k, None)
                pkt, new_ef, est = per_key_topk_quantize(delta, ef, k_ratio=upload_ratio,
                                                         quant_bits=(8 if compress_mode == "sparse8" else 32))
                pkt["type"] = "sparse"
                self.ef_buffer[k] = new_ef
                packets[k] = pkt; bytes_up_total += est
        return packets, bytes_up_total

    @torch.no_grad()
    def evaluate(self,
                 global_backbone_model: nn.Module,
                 loss_name: str,
                 eval_use_global_head: bool,
                 personalized: bool,
                 use_ema: bool,
                 ema_state: Optional[Dict[str, torch.Tensor]] = None):
        if self.task == "pl":
            global_backbone_model.update_head_for_k(self.k)
            eval_model = PFL_REMNet(input_dim=self.tr_ds.X.shape[1], initial_k=self.k,
                                    two_layer_head=bool(getattr(global_backbone_model, "two_layer_head", False)),
                                    head_dropout=0.0).to(self.device)
            state = (ema_state if (use_ema and ema_state is not None) else global_backbone_model.state_dict())
            ms = eval_model.state_dict()
            for k_key in state.keys():
                if k_key.startswith("extractors.") or k_key.startswith("backbone."):
                    if k_key in ms: ms[k_key].copy_(state[k_key].to(self.device))
            use_local_head = (personalized and (not eval_use_global_head))
            if use_local_head and (self.local_head_state is not None):
                for k, v in self.local_head_state.items():
                    if k in ms and k.startswith("head."): ms[k].copy_(v.to(self.device))
            eval_model.load_state_dict(ms, strict=True); eval_model.eval()
            preds, tgts = [], []
            for xb, yb in self._loader(self.te_ds, shuffle=False):
                xb = xb.to(self.device, non_blocking=True); yb = yb.to(self.device, non_blocking=True)
                with amp.autocast('cuda', enabled=self.use_amp and (self.device.type == "cuda")):
                    pr = eval_model(xb)
                preds.append(pr); tgts.append(yb)
            Pn = torch.cat(preds, dim=0); Tn = torch.cat(tgts, dim=0)
            y_mean = self.y_mean; y_std = self.y_std
            P = (Pn * y_std + y_mean).detach().cpu().numpy()
            T = (Tn * y_std + y_mean).detach().cpu().numpy()
            rmse, mae = rmse_mae_np(P, T)
            rho = spearman_rho_torch(torch.from_numpy(P), torch.from_numpy(T))
            return rmse, mae, rho
        else:
            eval_model = PFL_KPIPredictor(input_dim=self.tr_ds.X.shape[1], initial_kpi_outputs=self.k,
                                          hidden_dim=KPI_HIDDEN_DIM_DEFAULT, dropout=0.0).to(self.device)
            state = (ema_state if (use_ema and ema_state is not None) else global_backbone_model.state_dict())
            ms = eval_model.state_dict()
            for k_key in state.keys():
                if k_key.startswith("backbone.") and k_key in ms: ms[k_key].copy_(state[k_key].to(self.device))
            use_local_head = (personalized and (not eval_use_global_head))
            if use_local_head and (self.local_head_state is not None):
                for k, v in self.local_head_state.items():
                    if k in ms and k.startswith("head."): ms[k].copy_(v.to(self.device))
            eval_model.load_state_dict(ms, strict=True); eval_model.eval()
            preds, tgts = [], []
            for xb, yb in self._loader(self.te_ds, shuffle=False):
                xb = xb.to(self.device, non_blocking=True); yb = yb.to(self.device, non_blocking=True)
                with amp.autocast('cuda', enabled=self.use_amp and (self.device.type == "cuda")):
                    pr = eval_model(xb)
                target = yb[0:1, :]
                preds.append(pr); tgts.append(target)
            Pn = torch.cat(preds, dim=0); Tn = torch.cat(tgts, dim=0)
            P = Pn.detach().cpu().numpy(); T = Tn.detach().cpu().numpy()
            rmse, mae = rmse_mae_np(P, T)
            rho = spearman_rho_torch(torch.from_numpy(P), torch.from_numpy(T))
            return rmse, mae, rho

    @torch.no_grad()
    def evaluate_detailed(self,
                          global_backbone_model: nn.Module,
                          eval_use_global_head: bool,
                          personalized: bool,
                          use_ema: bool,
                          ema_state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        if self.task == "pl":
            global_backbone_model.update_head_for_k(self.k)
            eval_model = PFL_REMNet(input_dim=self.tr_ds.X.shape[1], initial_k=self.k,
                                    two_layer_head=bool(getattr(global_backbone_model, "two_layer_head", False)),
                                    head_dropout=0.0).to(self.device)
            state = (ema_state if (use_ema and ema_state is not None) else global_backbone_model.state_dict())
            ms = eval_model.state_dict()
            for k_key in state.keys():
                if k_key.startswith("extractors.") or k_key.startswith("backbone."):
                    if k_key in ms: ms[k_key].copy_(state[k_key].to(self.device))
            use_local_head = (personalized and (not eval_use_global_head))
            if use_local_head and (self.local_head_state is not None):
                for k, v in self.local_head_state.items():
                    if k in ms and k.startswith("head."): ms[k].copy_(v.to(self.device))
            eval_model.load_state_dict(ms, strict=True); eval_model.eval()
            loader = self._loader(self.te_ds, shuffle=False)
            K = self.k
            sse = np.zeros(K, dtype=np.float64); sae = np.zeros(K, dtype=np.float64); cnt = np.zeros(K, dtype=np.int64)
            sse_all = 0.0; sae_all = 0.0; cnt_all = 0
            P_list, T_list = [], []
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True); yb = yb.to(self.device, non_blocking=True)
                with amp.autocast('cuda', enabled=self.use_amp and (self.device.type == "cuda")):
                    prn = eval_model(xb)
                y_mean = self.y_mean; y_std = self.y_std
                pr = (prn * y_std + y_mean).detach().cpu().numpy()
                yt = (yb   * y_std + y_mean).detach().cpu().numpy()
                diff = pr - yt
                sse += np.sum(diff * diff, axis=0); sae += np.sum(np.abs(diff), axis=0); cnt += diff.shape[0]
                sse_all += float(np.sum(diff * diff)); sae_all += float(np.sum(np.abs(diff))); cnt_all += int(diff.size)
                P_list.append(pr); T_list.append(yt)
            rmse_k = np.sqrt(sse / np.maximum(cnt, 1)); mae_k = sae / np.maximum(cnt, 1)
            rmse_micro = float(np.sqrt(sse_all / max(cnt_all, 1))); mae_micro = float(sae_all / max(cnt_all, 1))
            rmse_macro = float(np.mean(rmse_k)); mae_macro = float(np.mean(mae_k))
            P_all = torch.from_numpy(np.vstack(P_list)); T_all = torch.from_numpy(np.vstack(T_list))
            rho = spearman_rho_torch(P_all, T_all)
            metrics = {
                "rmse_micro": rmse_micro, "rmse_macro": rmse_macro,
                "mae_micro": mae_micro, "mae_macro": mae_macro,
                "rho": float(rho), "cnt_all": int(cnt_all)
            }
            for i in range(K):
                bs_name = f"BS{i + 1}" if K > 4 else ["A", "B", "C", "D"][i]
                metrics[f"rmse_{bs_name}"] = float(rmse_k[i])
                metrics[f"mae_{bs_name}"] = float(mae_k[i])
                metrics[f"cnt_{bs_name}"] = int(cnt[i])
            return metrics
        else:
            eval_model = PFL_KPIPredictor(input_dim=self.tr_ds.X.shape[1], initial_kpi_outputs=self.k,
                                          hidden_dim=KPI_HIDDEN_DIM_DEFAULT, dropout=0.0).to(self.device)
            state = (ema_state if (use_ema and ema_state is not None) else global_backbone_model.state_dict())
            ms = eval_model.state_dict()
            for k_key in state.keys():
                if k_key.startswith("backbone.") and k_key in ms: ms[k_key].copy_(state[k_key].to(self.device))
            use_local_head = (personalized and (not eval_use_global_head))
            if use_local_head and (self.local_head_state is not None):
                for k, v in self.local_head_state.items():
                    if k in ms and k.startswith("head."): ms[k].copy_(v.to(self.device))
            eval_model.load_state_dict(ms, strict=True); eval_model.eval()
            loader = self._loader(self.te_ds, shuffle=False)
            sse_all = 0.0; sae_all = 0.0; cnt_all = 0
            P_list, T_list = [], []
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True); yb = yb.to(self.device, non_blocking=True)
                with amp.autocast('cuda', enabled=self.use_amp and (self.device.type == "cuda")):
                    prn = eval_model(xb)
                target = yb[0:1, :]
                pr = prn.detach().cpu().numpy(); yt = target.detach().cpu().numpy()
                diff = pr - yt
                sse_all += float(np.sum(diff * diff))
                sae_all += float(np.sum(np.abs(diff)))
                cnt_all += int(diff.size)
                P_list.append(pr); T_list.append(yt)
            rmse_micro = float(np.sqrt(sse_all / max(cnt_all, 1)))
            mae_micro = float(sae_all / max(cnt_all, 1))
            P_all = np.vstack(P_list); T_all = np.vstack(T_list)
            diff_all = P_all - T_all
            rmse_per_kpi = np.sqrt(np.mean(diff_all**2, axis=0))
            mae_per_kpi = np.mean(np.abs(diff_all), axis=0)
            rho = spearman_rho_torch(torch.from_numpy(P_all), torch.from_numpy(T_all))
            metrics = {
                "rmse_micro": rmse_micro, "rmse_macro": float(np.mean(rmse_per_kpi)),
                "mae_micro": mae_micro, "mae_macro": float(np.mean(mae_per_kpi)),
                "rho": float(rho), "cnt_all": int(cnt_all)
            }
            return metrics

class Server:
    def __init__(self, global_model: nn.Module, clients: List[Client], device: torch.device,
                 personalized: bool, upload_ratio: float, quant_bits: int,
                 local_sync: int, log_comm: bool, save_dir: str,
                 eval_use_global_head: bool,
                 use_ema: bool, ema_decay: float, task: str):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.personalized = personalized
        self.upload_ratio = upload_ratio
        self.quant_bits = quant_bits
        self.local_sync = local_sync
        self.log_comm = log_comm
        self.save_dir = save_dir
        self.eval_use_global_head = eval_use_global_head
        self.task = task
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_state: Optional[Dict[str, torch.Tensor]] = None
        if self.use_ema:
            self.ema_state = {k: v.detach().clone().to(self.device) for k, v in self.global_model.state_dict().items()}
        self.cum_bytes = 0
        self.comm_fh = None

        ensure_dir(self.save_dir)
        self.eval_csv_path = os.path.join(self.save_dir, f"eval_summary_{task}.csv")
        if not os.path.exists(self.eval_csv_path):
            with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["round", "rmse_micro", "rmse_macro", "mae_micro", "mae_macro", "rho", "cum_bytes_mb"]
                w.writerow(header)

    def agg_keys(self) -> List[str]:
        keys = list(self.global_model.state_dict().keys())
        if self.personalized:
            if self.task == "pl":
                return [k for k in keys if k.startswith("extractors.") or k.startswith("backbone.")]
            else:
                return [k for k in keys if k.startswith("backbone.")]
        return keys

    def open_comm_log(self):
        if self.log_comm:
            ensure_dir(self.save_dir)
            self.comm_fh = open(os.path.join(self.save_dir, f"comm_log_{self.task}.csv"),
                                "w", encoding="utf-8", newline="")
            self.comm_fh.write("round,client_id,bytes_up_round,cum_bytes\n")

    def close_comm_log(self):
        if self.comm_fh:
            self.comm_fh.close(); self.comm_fh = None

    def _ema_update(self):
        if not self.use_ema or self.ema_state is None: return
        with torch.no_grad():
            cur = self.global_model.state_dict()
            for k in cur.keys():
                self.ema_state[k].mul_((self.ema_decay)).add_(cur[k].detach() * (1.0 - self.ema_decay))

    def run(self, rounds: int, loss_name: str, local_epochs: int, lr: float, clip_grad: float):
        self.open_comm_log()
        for r in range(1, rounds + 1):
            packets_per_client: List[Dict[str, dict]] = []
            bytes_round = 0
            for c in self.clients:
                pkts, b_up = c.local_train_and_compress(
                    self.global_model, loss_name,
                    epochs=local_epochs, lr=lr,
                    personalized=self.personalized,
                    upload_ratio=self.upload_ratio,
                    local_sync=self.local_sync,
                    quant_bits=self.quant_bits,
                    round_idx=r
                )
                packets_per_client.append(pkts); bytes_round += b_up
                if self.comm_fh:
                    self.comm_fh.write(f"{r},{c.cid},{b_up},{self.cum_bytes + b_up}\n")

            # 每 local_sync 轮做一次聚合
            if (r % self.local_sync) == 0:
                agg_keys = self.agg_keys()
                new_state = {k: v.clone().to(self.device) for k, v in self.global_model.state_dict().items()}
                for k in agg_keys:
                    param = self.global_model.state_dict()[k]
                    shape = param.shape; numel = param.numel()
                    deltas = []
                    for pkts in packets_per_client:
                        pkt = pkts.get(k, None)
                        if pkt is None: continue
                        if pkt.get("type", "sparse") == "dense":
                            v = pkt["val"].to(self.device).view(shape).float()
                            deltas.append(v)
                        else:
                            idx = pkt["idx"]; bits = int(pkt["bits"])
                            if isinstance(idx, torch.Tensor):
                                idx = idx.to(device=self.device, dtype=torch.long)
                            else:
                                idx = torch.tensor(idx, device=self.device, dtype=torch.long)
                            if bits == 8:
                                vals = pkt["val"].to(self.device).float() * float(pkt["scale"])
                            else:
                                vals = pkt["val"].to(self.device).float()
                            flat = torch.zeros(numel, device=self.device, dtype=torch.float32)
                            if idx.numel() > 0: flat.index_copy_(0, idx, vals)
                            deltas.append(flat.view(shape))
                    if len(deltas) > 0:
                        avg_delta = torch.stack(deltas, dim=0).mean(dim=0)
                        new_state[k] = (param.to(self.device) + avg_delta)
                self.global_model.load_state_dict(new_state, strict=True)
                self._ema_update()

            self.cum_bytes += bytes_round

            # 详细评测（按客户端），并汇总到 summary
            per_client_metrics: List[Tuple[int, Dict[str, float]]] = []
            for c in self.clients:
                m = c.evaluate_detailed(self.global_model,
                                        eval_use_global_head=self.eval_use_global_head,
                                        personalized=self.personalized,
                                        use_ema=self.use_ema, ema_state=self.ema_state)
                per_client_metrics.append((c.cid, m))

            # 写 per-client CSV
            per_client_csv = os.path.join(self.save_dir, f"eval_per_client_{self.task}_round{r:03d}.csv")
            all_keys = sorted({k for _, m in per_client_metrics for k in m.keys()})
            with open(per_client_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["client_id"] + all_keys)
                for cid, m in per_client_metrics:
                    w.writerow([cid] + [m.get(k, 0.0) for k in all_keys])

            # 汇总均值 -> summary
            avg = {}
            for key in all_keys:
                if key != "cnt_all":
                    avg[key] = float(np.mean([m.get(key, 0.0) for _, m in per_client_metrics]))
            with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                row = [r,
                       avg.get('rmse_micro', 0.0), avg.get('rmse_macro', 0.0),
                       avg.get('mae_micro', 0.0),  avg.get('mae_macro', 0.0),
                       avg.get('rho', 0.0),
                       bytes_to_mb(self.cum_bytes)]
                w.writerow(row)

            print(f"[Round {r:03d} - {self.task.upper()}] "
                  f"RMSE micro/macro = {avg.get('rmse_micro',0.0):.4f} / {avg.get('rmse_macro',0.0):.4f} | "
                  f"MAE micro/macro = {avg.get('mae_micro',0.0):.4f} / {avg.get('mae_macro',0.0):.4f} | "
                  f"bytes_up_round={bytes_to_mb(bytes_round):.3f} MB | "
                  f"per-client csv: {os.path.basename(per_client_csv)}")

            # 保存权重
            ensure_dir(self.save_dir)
            torch.save(self.global_model.state_dict(), os.path.join(self.save_dir, f"global_{self.task}_r{r}.pth"))
            if self.use_ema and self.ema_state is not None:
                torch.save(self.ema_state, os.path.join(self.save_dir, f"global_ema_{self.task}_r{r}.pth"))

        self.close_comm_log()
        print(f"[Done] eval summary: {self.eval_csv_path}]")

# ========================= 参数 & 主程序 =========================
def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_pl.yaml")
    ap.add_argument("--users_dir", type=str, default=USERS_DIR_DEFAULT)
    ap.add_argument("--users_kpi_dir", type=str, default=USERS_KPI_DIR_DEFAULT)
    ap.add_argument("--dataset_prep_output_dir", type=str, default=DATASET_PREP_OUTPUT_DIR_DEFAULT,
                    help="指向 data 制作输出根目录（包含 manifests/），缺省从 users_dir / users_kpi_dir 的上级推断")
    ap.add_argument("--save_dir", type=str, default=SAVE_DIR_DEFAULT)
    ap.add_argument("--init_ckpt", type=str, default=INIT_CKPT_DEFAULT)
    ap.add_argument("--task", type=str, default=TASK_TYPE_DEFAULT, choices=["pl", "kpi"])
    ap.add_argument("--scenario", type=str, default="All", choices=SCENARIO_CHOICES)
    ap.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT)
    ap.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS_DEFAULT)
    ap.add_argument("--lr", type=float, default=LR_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--personalized", dest="personalized", action="store_true", default=PERSONALIZED_DEFAULT)
    ap.add_argument("--no-personalized", dest="personalized", action="store_false")
    ap.add_argument("--loss", type=str, default=LOSS_DEFAULT, choices=["huber", "mse", "mae"])
    ap.add_argument("--huber_delta", type=float, default=HUBER_DELTA_DEFAULT)
    ap.add_argument("--head_lr_mult", type=float, default=HEAD_LR_MULT_DEFAULT)
    ap.add_argument("--two_layer_head", action="store_true", default=TWO_LAYER_HEAD_DEF)
    ap.add_argument("--head_dropout", type=float, default=HEAD_DROPOUT_DEFAULT)
    ap.add_argument("--compress", type=str, default=COMPRESS_DEFAULT, choices=["off", "sparse8", "sparse32"])
    ap.add_argument("--upload_ratio", type=float, default=UPLOAD_RATIO_DEFAULT)
    ap.add_argument("--quant_bits", type=int, default=QUANT_BITS_DEFAULT, choices=[8, 32])
    ap.add_argument("--local_sync", type=int, default=LOCAL_SYNC_DEFAULT)
    ap.add_argument("--log_comm", action="store_true", default=LOG_COMM_DEFAULT)
    ap.add_argument("--eval_use_global_head", dest="eval_use_global_head", action="store_true")
    ap.add_argument("--eval_use_local_head", dest="eval_use_global_head", action="store_false")
    ap.set_defaults(eval_use_global_head=EVAL_USE_GLOBAL_HEAD_DEFAULT)
    ap.add_argument("--clip_grad", type=float, default=CLIP_GRAD_DEFAULT)
    ap.add_argument("--use_ema", action="store_true", default=USE_EMA_DEFAULT)
    ap.add_argument("--no_ema", dest="use_ema", action="store_false")
    ap.add_argument("--ema_decay", type=float, default=EMA_DECAY_DEFAULT)
    ap.add_argument("--kpi_lr_mult", type=float, default=KPI_LR_MULT_DEFAULT)
    ap.add_argument("--kpi_head_dropout", type=float, default=KPI_HEAD_DROPOUT_DEFAULT)
    ap.add_argument("--amp", action="store_true", default=AMP_DEFAULT, help="开启自动混合精度训练与评估（CUDA）")
    return parse_args_with_config(ap)

def main():
    args = parse_args()

    scenario = args.scenario

    set_seed(args.seed)
    ensure_dir(args.save_dir)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"[Config] Task={args.task} | device={device} | AMP={args.amp}")
    print(f"[Config] Personalized(backbone-only agg)={args.personalized} | EMA={args.use_ema} (decay={args.ema_decay})")
    print(f"[Config] Compress={args.compress} | upload_ratio={args.upload_ratio} | bits={args.quant_bits} | local_sync={args.local_sync}")
    print(f"[Config] Scenario={scenario}")

    # 自动推断 dataset_prep_output_dir（若未显式传入或与 users 路径不一致）
    if args.task == "pl":
        inferred = _infer_dataset_root(args.users_dir)
    else:
        inferred = _infer_dataset_root(args.users_kpi_dir)
    if not args.dataset_prep_output_dir or not os.path.isdir(args.dataset_prep_output_dir):
        args.dataset_prep_output_dir = inferred

    # 场景 -> city 列表
    allowed_cities = _load_allowed_cities_by_scenario(args.dataset_prep_output_dir, scenario)

    # 选择数据目录并加载（带城市过滤）
    data_dir = args.users_dir if args.task == "pl" else args.users_kpi_dir
    print(f">>> Loading data from: {data_dir}")
    splits = load_users(data_dir, seed=args.seed, task=args.task, allowed_cities=allowed_cities)
    print(f">>> Num clients: {len(splits)}")
    print(f">>> Output dims per client: {[s[4] for s in splits]}")
    input_dim = splits[0][0].shape[1]
    print(f">>> Input feature dim: {input_dim}")

    # 构造客户端
    clients: List[Client] = []
    for cid, (Xtr, Ytr, Xte, Yte, K_or_num) in enumerate(splits):
        clients.append(Client(
            cid, Xtr, Ytr, Xte, Yte, K_or_num, device,
            head_lr_mult=args.head_lr_mult, batch_size=args.batch_size,
            huber_delta=args.huber_delta, clip_grad=args.clip_grad,
            head_dropout=args.head_dropout, task=args.task,
            kpi_lr_mult=args.kpi_lr_mult, kpi_head_dropout=args.kpi_head_dropout,
            use_amp=args.amp
        ))

    # 全局模型
    if args.task == "pl":
        max_k = max([c.k for c in clients], default=4)
        global_model = PFL_REMNet(input_dim=input_dim, initial_k=max_k,
                                  two_layer_head=args.two_layer_head,
                                  head_dropout=args.head_dropout).to(device)
    else:
        max_kpi = max([c.k for c in clients], default=KPI_NUM_OUTPUTS)
        global_model = PFL_KPIPredictor(input_dim=input_dim, initial_kpi_outputs=max_kpi,
                                        hidden_dim=KPI_HIDDEN_DIM_DEFAULT,
                                        dropout=args.kpi_head_dropout).to(device)

    # 压缩模式标签（仅用于本地逻辑分支）
    global_model._compress_mode = args.compress

    # 可选热启动
    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        try:
            state = torch.load(args.init_ckpt, map_location=device)
            global_model.load_state_dict(state, strict=False)
            print(f"[Init] loaded: {args.init_ckpt}")
        except Exception as e:
            print(f"[Init] load failed: {e}")

    # 服务器
    server = Server(global_model=global_model,
                    clients=clients,
                    device=device,
                    personalized=args.personalized,
                    upload_ratio=args.upload_ratio,
                    quant_bits=args.quant_bits,
                    local_sync=args.local_sync,
                    log_comm=bool(args.log_comm),
                    save_dir=args.save_dir,
                    eval_use_global_head=args.eval_use_global_head,
                    use_ema=args.use_ema, ema_decay=args.ema_decay,
                    task=args.task)

    # 保存配置（把场景也写进去便于复现实验）
    args_dict = vars(args).copy()
    args_dict["scenario"] = scenario
    args_dict["dataset_prep_output_dir"] = args.dataset_prep_output_dir
    save_json(args_dict, os.path.join(args.save_dir, f"args_{args.task}.json"))

    # 训练
    server.run(rounds=args.rounds, loss_name=args.loss,
               local_epochs=args.local_epochs, lr=args.lr, clip_grad=args.clip_grad)

if __name__ == "__main__":
    main()
