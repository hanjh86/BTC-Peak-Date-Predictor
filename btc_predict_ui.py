# -*- coding: utf-8 -*-
"""
BTC 예측 UI (디버그 로그 + 인덱스 하드닝 + PDF/확률 유틸 통합본, .iloc/중복컬럼 fix)
- 프리셋: JSON 텍스트를 정렬해 텍스트박스에 넣고, 입력칸에 직접 채움
- 프리셋 미적용/학습 없음/형식 오류 → 예측 실행 차단
- 예측 알고리즘: featureExtract + ElasticNet
- Upbit API 미사용: since_date = 테스트 시작 '월의 1일'
- dtype 안전: numpy 배열은 Open,High,Low,Close,Adj Close (float64)만 포함
- 상세 로그: btc_predict_debug.log
- PDF 히스토그램 + 총 면적 체크, 구간 확률 계산(pdf 면적/표본비율)
- FIX: .at → .iloc로 변경, load 시 중복 컬럼 제거
- 추가: 슬라이스 실제 날짜/품질 로그 (train/test/figure 요약)
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, date, timedelta
from typing import Optional, List

from collections import Counter
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from scipy.ndimage import gaussian_filter
import warnings, logging, traceback
warnings.filterwarnings("ignore")

# ---------- 로깅 ----------
logging.basicConfig(
    filename="btc_predict_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
DEBUG = True

def logdbg(msg: str):
    if DEBUG:
        logging.debug(msg)

def as_int(x, label="index") -> int:
    """정수 스칼라 강제 변환 + 실패 시 상세 로그"""
    try:
        if hasattr(x, "item"):
            v = x.item()
            return int(v)
        if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
            if len(x) == 1:
                v = list(x)[0]
                return int(v)
            raise TypeError(f"{label} is non-scalar with len={len(x)}; value={x}")
        return int(x)
    except Exception:
        logging.error(f"as_int failed for {label}: type={type(x)}, value={x!r}")
        logging.error("traceback:\n" + traceback.format_exc())
        raise

def ensure_bounds(idx, n, label="index") -> int:
    """0..n-1 범위로 강제 + 로그"""
    idx = as_int(idx, label)
    if idx < 0:
        logging.warning(f"{label} < 0 -> clamp to 0 (was {idx})")
        idx = 0
    if idx >= n:
        logging.warning(f"{label} >= {n} -> clamp to {n-1} (was {idx})")
        idx = n - 1
    return idx

# ---------- 유틸 ----------
def parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def last_month_end(today: Optional[date] = None) -> date:
    if today is None:
        today = date.today()
    y, m = today.year, today.month
    if m == 1:
        y2, m2 = y - 1, 12
    else:
        y2, m2 = y, m - 1
    if m2 == 12:
        return date(y2, 12, 31)
    first_this = date(y2, m2 + 1, 1)
    return first_this - timedelta(days=1)

def month_end_index_for_iso(month_end_df: pd.DataFrame, iso_str: str) -> int:
    """
    YYYY-MM-DD 가 포함된 달의 '월말' 행 인덱스(정수).
    없으면 이후 가장 가까운 월말.
    """
    dt = pd.Timestamp(iso_str)
    mask = month_end_df["Date"].dt.to_period("M") == pd.Period(dt, freq="M")
    idxs = month_end_df.index[mask]
    if len(idxs) > 0:
        out = as_int(idxs.max(), f"month_end_index_for_iso({iso_str})")
        logdbg(f"month_end_index_for_iso hit month_end: {iso_str} -> {out}")
        return out
    later = month_end_df[month_end_df["Date"] >= dt]
    if not later.empty:
        out = as_int(later.index[0], f"month_end_index_for_iso({iso_str})")
        logdbg(f"month_end_index_for_iso fallback later: {iso_str} -> {out}")
        return out
    out = as_int(month_end_df.index.max(), f"month_end_index_for_iso({iso_str})")
    logdbg(f"month_end_index_for_iso fallback last: {iso_str} -> {out}")
    return out

def get_since_date_from_month_start(mdf: pd.DataFrame, test_start_idx: int) -> str:
    """Upbit API 대신: 테스트 시작 '월의 1일'을 since_date 로 사용. (포지션 기반 접근, .iloc)"""
    i = ensure_bounds(test_start_idx, len(mdf), "since.test_start_idx")
    d = pd.to_datetime(mdf["Date"].iloc[i]).date()
    return f"{d.year:04d}-{d.month:02d}-01"

# ---------- 확률 계산 유틸 (PDF/표본비율) ----------
def _to_num_seconds(dt_like):
    """datetime / np.datetime64 / pandas.Timestamp/Series/Index/array를 '초' 단위 float로 변환"""
    dt = pd.to_datetime(dt_like)
    if isinstance(dt, pd.Timestamp):
        return dt.value / 1e9  # ns -> sec
    return dt.astype("int64") / 1e9

def pdf_area_between(samples_dt, start_dt, end_dt, bins=20):
    """
    samples_dt: datetime 리스트/배열
    [start_dt, end_dt) 구간의 확률(PDF 면적, 히스토그램 근사)을 반환
    """
    x = _to_num_seconds(samples_dt)
    a = _to_num_seconds(start_dt)
    b = _to_num_seconds(end_dt)
    hist, edges = np.histogram(x, bins=bins, density=True)
    bin_left  = edges[:-1]
    bin_right = edges[1:]
    overlap_left  = np.maximum(bin_left, a)
    overlap_right = np.minimum(bin_right, b)
    overlap_width = np.maximum(0.0, overlap_right - overlap_left)
    area = float(np.sum(hist * overlap_width))
    logdbg(f"pdf_area_between [{start_dt},{end_dt}) -> {area}")
    return area

def empirical_prob_between(samples_dt, start_dt, end_dt):
    """표본 비율로 계산한 경험적 확률 [start_dt, end_dt)"""
    s = pd.to_datetime(samples_dt)
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    prob = float(((s >= start_dt) & (s < end_dt)).mean())
    logdbg(f"empirical_prob_between [{start_dt},{end_dt}) -> {prob}")
    return prob

# ---------- (신규) 슬라이스 실제 날짜/품질 로그 유틸 ----------
def log_slice_dates(tag: str, mdf: pd.DataFrame, start_idx: int, end_idx: int):
    """특징추출 구간에 대응하는 실제 날짜/인덱스/샘플 수/NaN 여부를 자세히 로그"""
    n = len(mdf)
    si = ensure_bounds(start_idx, n, f"{tag}.start_idx")
    ei = ensure_bounds(end_idx,   n, f"{tag}.end_idx")
    d0 = pd.to_datetime(mdf["Date"].iloc[si]).date()
    d1 = pd.to_datetime(mdf["Date"].iloc[ei]).date()
    logdbg(f"[{tag}] idx={si}..{ei}  dates={d0}..{d1}  (rows={ei-si+1})")

def log_segment_quality(tag: str, arr: np.ndarray):
    """
    arr: data_np[start:end+1, :]
    각 열(Open,High,Low,Close,Adj Close) NaN/Inf 유무, 기본 통계 로그
    """
    colnames = ["Open","High","Low","Close"]
    if arr.size == 0:
        logdbg(f"[{tag}] segment is EMPTY")
        return
    for j, name in enumerate(colnames):
        col = arr[:, j]
        isn = np.isnan(col)
        isf = ~np.isfinite(col)
        logdbg(f"[{tag}] {name}: len={len(col)} min={np.nanmin(col):.6g} max={np.nanmax(col):.6g} "
               f"mean={np.nanmean(col):.6g} NaN={int(isn.sum())} !finite={int(isf.sum())}")

# ---------- 알고리즘 구성요소 ----------
def gaussian(x):
    return gaussian_filter(x, sigma=5)

def cv(x: np.ndarray, eps=1e-12) -> float:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return sd / (abs(mu) + eps)

def winsorize(x, lo=0.01, hi=0.99):
    lo_v, hi_v = np.quantile(x, [lo, hi])
    return np.clip(x, lo_v, hi_v)

def featureExtract(data_np: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    global Open, High, Low, Close
    n = data_np.shape[0]
    start_idx = ensure_bounds(start_idx, n, "featureExtract.start_idx")
    end_idx   = ensure_bounds(end_idx,   n, "featureExtract.end_idx")
    if end_idx < start_idx:
        logging.error(f"featureExtract: end<start (start={start_idx}, end={end_idx})")
        raise ValueError("featureExtract: end_idx < start_idx")

    logdbg(f"featureExtract slice [{start_idx}:{end_idx+1}] on array shape={data_np.shape}, dtype={data_np.dtype}")

    seg = data_np[start_idx:end_idx+1, :]
    Open  = seg[:, 0]
    High  = seg[:, 1]
    Low   = seg[:, 2]
    Close = seg[:, 3]

    eps = 1e-12

    # ---- 1~4: CV(Open/High/Low/Close)
    f1 = cv(Open)
    f2 = cv(High)
    f3 = cv(Low)
    f4 = cv(Close)
    f5 = f4 - f1
    f6 = f2 - f3
    f7 = f5 / f6

    # ---- 7~9: 전체 구간 '단순' 수익률 평균/표준편차/Sharpe-like  (로그 제거)
    if len(Close) >= 2:
        simple_r = (Close[1:] / (Close[:-1] + eps)) - 1.0
        f8 = float(np.mean(simple_r))
        f9 = float(np.std(simple_r))
        f10 = float(f6 / (f7 + eps))
    else:
        f8 = f9 = f10 = 0.0

    # ---- 11: 평균 스토캐스틱 위치
    hl = High - Low
    stoch = np.where(hl > 0.0, (Close - Low) / (hl + eps), 0.0)
    stoch = np.clip(stoch, 0.0, 1.0)
    f11 = float(np.mean(stoch))

    out = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11], dtype=float)
    logdbg(f"featureExtract out shape={out.shape}, values={out}")
    return out

def predict(trainData: np.ndarray, label: np.ndarray, testData: np.ndarray) -> float:
    alpha = 0.01
    l1_ratio = 0.5
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=True),      
        # StandardScaler(),                  
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)
    )
    model.fit(trainData, label)
    y_pred = model.predict(testData.reshape(1, -1))
    out = float(np.array(y_pred).item())
    logdbg(f"predict -> {out}")
    return out

# ---------- 데이터 준비 ----------
def load_month_end(start="2015-01-01", end=None):
    """월말 DF(mdf)와 float64 numpy(month_end_np)를 반환."""
    btc = yf.download("BTC-USD", start=start, end=end, auto_adjust=False)
    if btc is None or btc.empty:
        raise RuntimeError("BTC-USD 데이터를 가져오지 못했습니다.")
    df = btc.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df["MonthEnd"] = df["Date"].dt.is_month_end
    mdf = df[df["MonthEnd"]].copy().reset_index(drop=True)

    # ★ 중복 컬럼 제거 (라벨 기반 인덱싱 오류 방지)
    mdf = mdf.loc[:, ~mdf.columns.duplicated()].copy()

    if "Adj Close" not in mdf.columns:
        mdf["Adj Close"] = mdf["Close"]

    # (선택) 진단: 중복 컬럼 있었는지 기록
    dup = mdf.columns[mdf.columns.duplicated()].tolist()
    if dup:
        logdbg(f"[WARN] duplicated columns removed: {dup}")

    month_end_np = mdf[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
    logdbg(f"load_month_end: rows={len(mdf)}, np.shape={month_end_np.shape}, dtype={month_end_np.dtype}")
    return mdf, month_end_np

def to_safe_day_offset(pred, *, min_days=0, max_days=365*15):
    import numpy as np
    val = float(np.asarray(pred).item())
    if not np.isfinite(val):
        raise ValueError(f"Predicted days is not finite: {val}")
    ival = int(round(val))
    ival = max(min_days, min(max_days, ival))
    return ival

# ---------- 예측 실행 ----------
def run_prediction_with_user_config(config: dict):
    global result_dates, mdf, month_end_np, roll_days, most_common_days, tsi, tei, segT, feat 
    """
    config = {"training":[{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"},...],
              "test":{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}}
    """
    try:
        TRAIN = config.get("training", []) or []
        TEST  = config.get("test", {}) or {}
        if not TRAIN or not TEST.get("start") or not TEST.get("end"):
            raise ValueError("학습/테스트 구성이 비었습니다.")
        logdbg(f"run_prediction config: {config}")

        mdf, month_end_np = load_month_end(start="2015-01-01", end=None)

        # 학습 구성
        feats, labels = [], []
        for k, r in enumerate(TRAIN):
            s, e = r.get("start"), r.get("end")
            si = ensure_bounds(month_end_index_for_iso(mdf, s), len(month_end_np), f"train[{k}].start_idx")
            ei = ensure_bounds(month_end_index_for_iso(mdf, e), len(month_end_np), f"train[{k}].end_idx")

            # ★ 실제 사용된 월말 인덱스/날짜 로그 + 품질 로그
            log_slice_dates(f"train[{k}]", mdf, si, ei)
            seg = month_end_np[si:ei+1, :]
            log_segment_quality(f"train[{k}]", seg)

            feats.append(featureExtract(month_end_np, si, ei))
            labels.append(float((pd.Timestamp(e) - pd.Timestamp(s)).days))

        trainData = np.vstack(feats).astype(float)
        label = np.array(labels, dtype=float)
        logdbg(f"trainData shape={trainData.shape}, label shape={label.shape}, labels={labels}")

        # 테스트
        ts, te = TEST["start"], TEST["end"]
        tsi = ensure_bounds(month_end_index_for_iso(mdf, ts), len(month_end_np), "test.start_idx")
        tei = ensure_bounds(month_end_index_for_iso(mdf, te), len(month_end_np), "test.end_idx")

        # ★ 실제 사용된 월말 인덱스/날짜 로그 + 품질 로그
        log_slice_dates("test", mdf, tsi, tei)
        segT = month_end_np[tsi:tei+1, :]
        log_segment_quality("test", segT)

        testData = featureExtract(month_end_np, tsi, tei)

        # 예측
        y_days = predict(trainData, label, testData)
        since_date = get_since_date_from_month_start(mdf, tsi)
        since_dt = datetime.strptime(since_date, "%Y-%m-%d")

        # 롤링 예측 분포
        roll_days = []
        result_dates = []
        for i in range(tsi + 1, tei + 1):
            i2 = ensure_bounds(i, len(month_end_np), "roll.i")
            rollTest = featureExtract(month_end_np, tsi, i2)
            roll_pred = predict(trainData, label, rollTest)
            # 안전하게 정수 일수로 변환
            try:
                days = int(round(float(roll_pred)))
            except Exception:
                continue
            roll_days.append(days)
            result_dates.append((since_dt + timedelta(days=float(roll_pred))))
        price_peak_date = since_dt + timedelta(days=float(y_days))
        logdbg(f"[pred] since_date={since_date}  pred_days={y_days:.3f}  pred_date={price_peak_date.date()}")
        
        # ---------------------------
        # 최빈값 선택
        # ---------------------------
        if roll_days:
            most_common_days, freq = Counter(roll_days).most_common(1)[0]
            price_peak_date = since_dt + timedelta(days=most_common_days)
            print(f"최빈 예측일수={most_common_days} (빈도={freq}), 피크일={price_peak_date}")
        else:
            price_peak_date = None
            print("롤링 예측 결과 없음")
            
        # ★ 롤링 샘플 요약 로그
        logdbg(
            f"[roll] samples={len(result_dates)} "
            f"first5={list(map(lambda d: pd.to_datetime(d).date(), result_dates[:5]))} "
            f"last5={list(map(lambda d: pd.to_datetime(d).date(), result_dates[-5:]))}"
        )

        # ---------------------------
        # 확률 분포로 시각화 (PDF)
        # ---------------------------
        if result_dates:
            result_array = np.array(result_dates, dtype="datetime64[ns]")

            # ★ figure 요약 로그
            all_dt = pd.to_datetime(result_array)
            all_dt_s = pd.Series(all_dt)
            med = all_dt_s.sort_values().iloc[len(all_dt_s)//2] if len(all_dt_s) > 0 else None
            logdbg(f"[figure] result_array summary: min={all_dt_s.min()} max={all_dt_s.max()} "
                   f"median={med} count={len(all_dt_s)}")

            plt.figure(figsize=(10, 6))
            densities, bin_edges, _ = plt.hist(
                pd.to_datetime(result_array).to_pydatetime(),
                bins=20,
                density=True,
                alpha=0.5,
                edgecolor="black"
            )
            plt.title("Probability Density Function (PDF)")
            plt.xlabel("Price Peak (date)")
            plt.ylabel("Probability Density")
            plt.grid(True)
                       
            # X축 날짜 포맷을 년-월-일로 표시           
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            
            plt.tight_layout()
            plt.show()

            # PDF 총 면적(히스토그램 기반) 확인
            hist, edges = np.histogram(bin_edges, bins=20, density=True)
            area = float(np.sum(hist * np.diff(edges)))

            print("PDF 총 면적(히스토그램 기반) ≈", area)
        else:
            result_array = np.array([], dtype="datetime64[ns]")

        return {
            "since": since_date,
            "pred_days": float(y_days),
            "pred_peak_date": price_peak_date.date(),
            "train_samples": int(trainData.shape[0]),
            "test_range": (
                pd.to_datetime(mdf["Date"].iloc[tsi]).date(),
                pd.to_datetime(mdf["Date"].iloc[tei]).date()
            ),
            "result_array": result_array
        }
    except Exception as e:
        logging.error("run_prediction_with_user_config FAILED: " + str(e))
        logging.error("traceback:\n" + traceback.format_exc())
        raise

# ---------- Tkinter UI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BTC 예측: JSON → 입력칸 / Upbit API 미사용 / 인덱스 하드닝 / PDF 확률")
        self.geometry("980x900")
        self.train_rows: List = []  # (frame, s_entry, e_entry, status_lbl, sel_var)
        self.preset_applied = False
        self.last_result_array = None  # 확률 계산용 샘플 저장
        self._build_ui()

    def _build_ui(self):
        # 프리셋
        top = ttk.LabelFrame(self, text="프리셋")
        top.pack(fill="x", padx=12, pady=8)
        ttk.Label(
            top,
            text="프리셋 버튼을 누르면 JSON 텍스트를 정렬하고, 학습/테스트 입력칸에 그대로 채웁니다."
        ).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Button(
            top, text="샘플 구성 불러오기 (JSON → 입력칸 + 텍스트 갱신)",
            command=self.load_from_json_and_fill
        ).grid(row=0, column=1, padx=8, pady=6)

        # 학습 구간
        train_frame = ttk.LabelFrame(self, text="학습 구간(여러 개 가능)")
        train_frame.pack(fill="both", expand=True, padx=12, pady=8)
        head = ttk.Frame(train_frame); head.pack(fill="x", pady=(6, 0))
        ttk.Label(head, text="시작일 (YYYY-MM-DD)", width=25).grid(row=0, column=0, padx=8)
        ttk.Label(head, text="종료일 (YYYY-MM-DD)", width=25).grid(row=0, column=1, padx=8)
        ttk.Label(head, text="상태", width=30).grid(row=0, column=2, padx=8)
        self.train_list_frame = ttk.Frame(train_frame); self.train_list_frame.pack(fill="both", expand=True, pady=6)
        tbtn = ttk.Frame(train_frame); tbtn.pack(fill="x", pady=(0, 6))
        ttk.Button(tbtn, text="+ 학습 구간 추가", command=self.add_train_row).pack(side="left", padx=6)
        ttk.Button(tbtn, text="선택 행 삭제", command=self.delete_selected_train_row).pack(side="left", padx=6)

        # 테스트 구간
        test = ttk.LabelFrame(self, text="테스트 구간(1개)")
        test.pack(fill="x", padx=12, pady=8)
        ttk.Label(test, text="시작일").grid(row=0, column=0, padx=8, pady=6, sticky="e")
        self.test_start_entry = ttk.Entry(test, width=20)
        self.test_start_entry.grid(row=0, column=1, padx=4)
        ttk.Label(test, text="종료일").grid(row=0, column=2, padx=8, pady=6, sticky="e")
        self.test_end_entry = ttk.Entry(test, width=20)
        self.test_end_entry.grid(row=0, column=3, padx=4)
        self.test_status = ttk.Label(test, text="상태: 입력 대기", foreground="#666")
        self.test_status.grid(row=0, column=4, padx=12, sticky="w")
        ttk.Button(test, text="유효성 검사", command=self.validate_all).grid(row=0, column=5, padx=8)

        # JSON 박스
        io = ttk.LabelFrame(self, text="JSON")
        io.pack(fill="both", expand=False, padx=12, pady=8)
        self.preview = tk.Text(io, height=12, wrap="none")
        self.preview.pack(fill="both", expand=True, padx=8, pady=6)

        # 시작 시 JSON 텍스트 자동 입력
        example = {
            "training": [
                {"start": "2016-07-31", "end": "2017-12-31"},
                {"start": "2020-05-31", "end": "2021-11-30"}
            ],
            "test": {"start": "2024-04-30", "end": last_month_end().isoformat()}
        }
        self.preview.insert("1.0", json.dumps(example, indent=2, ensure_ascii=False))

        actions = ttk.Frame(io); actions.pack(fill="x", pady=(0, 6))
        ttk.Button(actions, text="JSON 저장", command=self.save_json).pack(side="left", padx=6)
        ttk.Button(actions, text="클립보드 복사", command=self.copy_to_clipboard).pack(side="left", padx=6)
        ttk.Button(actions, text="예측 실행", command=self.run_now).pack(side="right", padx=6)

        # 확률 계산 UI
        prob = ttk.LabelFrame(self, text="구간 확률 계산 (PDF 면적/표본 비율)")
        prob.pack(fill="x", padx=12, pady=8)
        ttk.Label(prob, text="구간 시작(YYYY-MM-DD)").grid(row=0, column=0, padx=8, pady=6, sticky="e")
        self.prob_start_entry = ttk.Entry(prob, width=20)
        self.prob_start_entry.grid(row=0, column=1, padx=4)
        ttk.Label(prob, text="구간 종료(YYYY-MM-DD)").grid(row=0, column=2, padx=8, pady=6, sticky="e")
        self.prob_end_entry = ttk.Entry(prob, width=20)
        self.prob_end_entry.grid(row=0, column=3, padx=4)
        ttk.Label(prob, text="Bins").grid(row=0, column=4, padx=8, pady=6, sticky="e")
        self.prob_bins_entry = ttk.Entry(prob, width=8)
        self.prob_bins_entry.insert(0, "20")
        self.prob_bins_entry.grid(row=0, column=5, padx=4)
        ttk.Button(prob, text="확률 계산", command=self.calc_probs).grid(row=0, column=6, padx=8)

    # ---- 학습 행 관리 (Entry 직접 사용)
    def add_train_row(self, start="", end=""):
        row = ttk.Frame(self.train_list_frame)
        s_entry = ttk.Entry(row, width=25)
        e_entry = ttk.Entry(row, width=25)
        status_lbl = ttk.Label(row, text="상태: 입력 대기", foreground="#666")
        sel_var = tk.BooleanVar(value=False)
        sel_chk = ttk.Checkbutton(row, variable=sel_var)

        s_entry.grid(row=0, column=0, padx=8, pady=6)
        e_entry.grid(row=0, column=1, padx=8, pady=6)
        status_lbl.grid(row=0, column=2, padx=8, pady=6, sticky="w")
        sel_chk.grid(row=0, column=3, padx=8, pady=6)
        row.pack(fill="x")

        if start:
            s_entry.insert(0, start)
        if end:
            e_entry.insert(0, end)

        self.train_rows.append((row, s_entry, e_entry, status_lbl, sel_var))

    def delete_selected_train_row(self):
        kept = []
        for (f, s_entry, e_entry, lbl, chk) in self.train_rows:
            if chk.get():
                f.destroy()
            else:
                kept.append((f, s_entry, e_entry, lbl, chk))
        self.train_rows = kept

    # ---- 프리셋: JSON → 입력칸 + 텍스트 갱신
    def load_from_json_and_fill(self):
        raw = self.preview.get("1.0", "end").strip()
        if not raw:
            messagebox.showwarning("안내", "JSON 텍스트가 비어 있습니다.")
            self.preset_applied = False
            return
        try:
            cfg = json.loads(raw)
        except Exception as e:
            messagebox.showerror("JSON 오류", f"파싱 실패: {e}")
            self.preset_applied = False
            return

        # 1) 텍스트박스 정렬 갱신
        pretty = json.dumps(cfg, indent=2, ensure_ascii=False)
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", pretty)

        # 2) 학습 입력칸 초기화 후 직접 채우기
        for (f, *_rest) in self.train_rows:
            f.destroy()
        self.train_rows = []
        tr = cfg.get("training", []) or []
        for r in tr:
            s = (r.get("start", "") or "").strip()
            e = (r.get("end", "") or "").strip()
            self.add_train_row()  # 빈 행 생성
            _f, s_entry, e_entry, _lbl, _sel = self.train_rows[-1]
            s_entry.delete(0, tk.END); s_entry.insert(0, s)
            e_entry.delete(0, tk.END); e_entry.insert(0, e)

        # 3) 테스트 입력칸 직접 채우기
        t = cfg.get("test", {}) or {}
        ts = (t.get("start", "") or "").strip()
        te = (t.get("end", "") or "").strip()
        self.test_start_entry.delete(0, tk.END); self.test_start_entry.insert(0, ts)
        self.test_end_entry.delete(0, tk.END);   self.test_end_entry.insert(0, te)

        self.preset_applied = True
        self.validate_all()
        self.update_idletasks()
        messagebox.showinfo("완료", f"학습 {len(tr)}개, 테스트 1개가 입력칸에 적용되었습니다.")

    # ---- 검증/JSON 동기화
    def validate_all(self) -> bool:
        ok = True
        has_training = False
        for (_f, s_entry, e_entry, lbl, _chk) in self.train_rows:
            s, e = s_entry.get().strip(), e_entry.get().strip()
            sdt, edt = parse_date(s), parse_date(e)
            if not sdt or not edt:
                lbl.config(text="상태: 형식 오류(YYYY-MM-DD)", foreground="#b45309")
                ok = False
            elif sdt > edt:
                lbl.config(text="상태: 시작>종료", foreground="#b45309")
                ok = False
            else:
                lbl.config(text="상태: 유효", foreground="#047857")
                has_training = True

        t_s = self.test_start_entry.get().strip()
        t_e = self.test_end_entry.get().strip()
        ts, te = parse_date(t_s), parse_date(t_e)
        if not ts or not te:
            self.test_status.config(text="상태: 형식 오류(YYYY-MM-DD).", foreground="#b45309")
            ok = False
        elif ts > te:
            self.test_status.config(text="상태: 시작>종료.", foreground="#b45309")
            ok = False
        else:
            self.test_status.config(text="상태: 유효", foreground="#047857")

        if not has_training:
            ok = False

        # 현재 입력칸 상태를 JSON 텍스트에 동기화
        cfg = self.build_config()
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", json.dumps(cfg, indent=2, ensure_ascii=False))

        return ok

    def build_config(self) -> dict:
        training = []
        for (_f, s_entry, e_entry, _lbl, _chk) in self.train_rows:
            s, e = s_entry.get().strip(), e_entry.get().strip()
            sdt, edt = parse_date(s), parse_date(e)
            if sdt and edt:
                training.append({"start": sdt.isoformat(), "end": edt.isoformat()})
        t_s = self.test_start_entry.get().strip()
        t_e = self.test_end_entry.get().strip()
        return {
            "training": training,
            "test": {"start": t_s or None, "end": t_e or None}
        }

    def save_json(self):
        cfg = self.build_config()
        path = filedialog.asksaveasfilename(
            title="JSON 저장",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="period_config.json"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("완료", f"저장됨: {path}")

    def copy_to_clipboard(self):
        cfg = self.build_config()
        txt = json.dumps(cfg, indent=2, ensure_ascii=False)
        self.clipboard_clear()
        self.clipboard_append(txt)
        messagebox.showinfo("완료", "JSON이 클립보드에 복사되었습니다.")

    # ---- 실행
    def run_now(self):
        if not self.preset_applied:
            messagebox.showwarning("안내", "먼저 '샘플 구성 불러오기' 버튼을 눌러 JSON을 적용하세요.")
            return
        if not self.validate_all():
            messagebox.showerror("에러", "학습/테스트 구간이 올바르지 않습니다.")
            return
        cfg = self.build_config()
        logdbg(f"run_now cfg: {cfg}")
        if not cfg["training"]:
            messagebox.showerror("에러", "학습 구간이 비어 있습니다.")
            return
        if not cfg["test"]["start"] or not cfg["test"]["end"]:
            messagebox.showerror("에러", "테스트 구간 시작/종료가 비어 있습니다.")
            return
        try:
            result = run_prediction_with_user_config(cfg)
            # 결과 저장(구간 확률 계산용)
            self.last_result_array = result.get("result_array", None)

            msg = (
                f"Since(테스트 시작 월의 1일): {result['since']}\n"
                f"테스트 구간(월말): {result['test_range'][0]} → {result['test_range'][1]}\n"
                f"예측 피크까지 일수: {result['pred_days']:.2f}\n"
                f"예측 피크 날짜    : {result['pred_peak_date']}\n"
                f"학습 샘플 수      : {result['train_samples']}"
            )
            messagebox.showinfo("예측 결과", msg)
        except Exception as e:
            messagebox.showerror("실행 에러", str(e))

    # ---- 구간 확률 계산
    def calc_probs(self):
        if self.last_result_array is None or len(self.last_result_array) == 0:
            messagebox.showwarning("안내", "먼저 '예측 실행'으로 PDF 샘플을 생성하세요.")
            return
        p_start = self.prob_start_entry.get().strip()
        p_end = self.prob_end_entry.get().strip()
        try:
            bins = int(self.prob_bins_entry.get().strip() or "20")
        except ValueError:
            bins = 20
            self.prob_bins_entry.delete(0, tk.END); self.prob_bins_entry.insert(0, "20")
        if not p_start or not p_end:
            messagebox.showerror("에러", "구간 시작/종료 날짜를 입력하세요 (YYYY-MM-DD).")
            return
        try:
            area_prob = pdf_area_between(self.last_result_array, p_start, p_end, bins=bins)
            emp_prob  = empirical_prob_between(self.last_result_array, p_start, p_end)
            messagebox.showinfo(
                "구간 확률",
                f"[{p_start} ~ {p_end}\n"
                f"- 히스토그램 면적 기반 확률: {area_prob*100:.2f}%\n"  
                f"- 경험적 확률: {emp_prob*100:.2f}%\n"
            )
        except Exception as e:
            messagebox.showerror("계산 에러", str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()
