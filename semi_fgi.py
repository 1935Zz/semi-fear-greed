"""美股半导体恐慌贪婪指数 (Semiconductor Fear & Greed Index)

参照 CNN Fear & Greed Index 的 7 个分项，把每个分项换成半导体板块的对应指标，
另加一个半导体特有的杠杆 ETF 资金情绪分项。每个分项按过去 504 个交易日做 z-score，
再经正态分布映射到 0-100，最后等权平均得到综合指数。

用法:
    .venv/bin/python semi_fgi.py               # 计算并生成 output/ 下的 HTML、JSON、CSV
    .venv/bin/python semi_fgi.py --no-options  # 跳过期权 put/call 快照
    .venv/bin/python semi_fgi.py --no-cnn      # 不抓取 CNN 指数做对比
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
DATA = ROOT / "data"                      # 需要长期保存、提交到仓库的数据
CACHE = ROOT / "cache" / "prices.pkl"     # 当天的行情缓存，不提交
PUTCALL_LOG = DATA / "putcall_log.csv"
TEMPLATE = ROOT / "template.html"
ET = "America/New_York"

# 半导体成分股：SOXX / SMH / 费城半导体指数的主要成分，加上中小盘设备、材料、模拟芯片公司，
# 让新高新低和成交量广度的统计更稳。上市不足 200 个交易日的会自动跳过
CONSTITUENTS = [
    "NVDA", "AVGO", "AMD", "TSM", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC",
    "INTC", "ADI", "MRVL", "NXPI", "MCHP", "ON", "MPWR", "ASML", "TER", "ENTG",
    "SWKS", "QRVO", "ARM", "GFS", "COHR", "MKSI", "LSCC", "ALAB", "CRDO", "SNPS",
    "CDNS", "AMKR", "RMBS", "SLAB", "SYNA", "CRUS", "DIOD", "FORM", "ONTO", "NVMI",
    "UCTT", "ICHR", "CAMT", "MTSI", "SMTC", "POWI", "SITM", "ALGM", "ACLS", "AOSL",
    "VSH", "STM", "UMC", "ASX",
]
MARKET = ["^SOX", "SPY", "TLT", "IEF", "HYG", "^VXN", "^VIX", "^VIX3M", "SOXL", "SOXS"]
OPTION_TICKERS = ["SMH", "SOXX", "SOXL"]

Z_WINDOW = 504      # 标准化回看窗口：约两年。一年窗口下动能等慢变量有 1/4 的天数顶格
Z_MIN = 252         # 至少一年数据才开始打分
HISTORY_YEARS = 8
NEAR_HL = 0.03      # 距 52 周高点/低点 3% 以内算"接近新高/新低"

CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CNN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0 Safari/537.36",
    "Referer": "https://www.cnn.com/",
    "Origin": "https://www.cnn.com",
    "Accept": "application/json",
}

# CNN 的分档：0-24 极度恐慌，25-44 恐慌，45-55 中性，56-75 贪婪，76-100 极度贪婪
RATINGS = [(25, "extreme_fear"), (45, "fear"), (56, "neutral"), (76, "greed"), (101, "extreme_greed")]


def rating(score: float | None) -> str | None:
    if score is None or not np.isfinite(score):
        return None
    for upper, name in RATINGS:
        if score < upper:
            return name
    return "extreme_greed"


# ---------------------------------------------------------------- 数据获取

def download(tickers: list[str], period: str) -> pd.DataFrame:
    yf.set_tz_cache_location(str(OUT / ".yf_cache"))
    kw = dict(period=period, auto_adjust=True, progress=False, threads=True)

    def missing_in(df):
        close = df["Close"]
        return [t for t in tickers if t not in close or close[t].notna().sum() == 0]

    df = yf.download(tickers, **kw)
    missing = missing_in(df)
    if missing:  # Yahoo 偶尔单个代码失败，单独重试一次
        time.sleep(2)
        extra = yf.download(missing, **kw)
        df = df.drop(columns=missing, level=1, errors="ignore").join(extra, how="outer")
        missing = missing_in(df)
    if "^SOX" in missing:
        raise RuntimeError("费城半导体指数 ^SOX 下载失败，请稍后重试")
    if missing:
        print(f"  ! 无数据，已跳过: {', '.join(missing)}")
    return df


def load_prices(tickers: list[str]) -> pd.DataFrame:
    """每个交易日第一次运行时下载全部历史并缓存，之后只下载最近 10 天覆盖上去。

    每天重下一次全量，是因为复权价格会随分红、拆股整体调整，不能一直在旧数据后面追加。
    """
    today = pd.Timestamp.now(tz=ET).date()
    cached = None
    if CACHE.exists():
        try:
            cached = pickle.loads(CACHE.read_bytes())
        except Exception:
            cached = None
    if (cached and cached.get("built") == today and cached.get("years") == HISTORY_YEARS
            and set(tickers) <= set(cached.get("tickers", []))):
        old = cached["df"]
        # 已退市的（比如 2026-10 并入 Skyworks 的 QRVO）保留历史，但不再请求
        cutoff = pd.Timestamp(today) - pd.Timedelta(days=15)
        active = [t for t in tickers if (old["Close"][t].last_valid_index() or cutoff) > cutoff]
        print(f"  使用今天的历史缓存，只下载最近 10 天（{len(active)} 个代码）")
        recent = download(active, "10d")
        return recent.combine_first(old).sort_index()

    df = download(tickers, f"{HISTORY_YEARS}y")
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps({"built": today, "years": HISTORY_YEARS, "tickers": tickers, "df": df}))
    return df


def market_status(now: pd.Timestamp | None = None) -> dict:
    """按纽交所日历判断当前交易状态：open 盘中，pre 盘前，closed 收盘后或休市。"""
    import pandas_market_calendars as mcal

    now = now or pd.Timestamp.now(tz="UTC")
    today = now.tz_convert(ET).date()
    sched = mcal.get_calendar("NYSE").schedule(start_date=today, end_date=today)
    if sched.empty:
        return {"state": "closed", "trading_day": False, "date": str(today)}
    open_, close = sched.iloc[0]["market_open"], sched.iloc[0]["market_close"]
    state = "pre" if now < open_ else "open" if now <= close else "closed"
    return {"state": state, "trading_day": True, "date": str(today),
            "open": open_.isoformat(), "close": close.isoformat()}


def fetch_cnn() -> dict | None:
    try:
        r = requests.get(CNN_URL, headers=CNN_HEADERS, timeout=20)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        print(f"  ! CNN 指数获取失败（不影响计算）: {e}")
        return None
    fg = d.get("fear_and_greed", {})
    hist = d.get("fear_and_greed_historical", {}).get("data", [])
    keys = {
        "market_momentum_sp500": "动能", "stock_price_strength": "股价强度",
        "stock_price_breadth": "股价广度", "put_call_options": "看跌/看涨期权",
        "market_volatility_vix": "波动率", "junk_bond_demand": "垃圾债需求",
        "safe_haven_demand": "避险需求",
    }
    comps = [{"name": n, "score": round(d[k]["score"], 1), "rating": d[k]["rating"].replace(" ", "_")}
             for k, n in keys.items() if k in d and "score" in d[k]]
    return {
        "score": round(fg.get("score", float("nan")), 1),
        "rating": (fg.get("rating") or "").replace(" ", "_"),
        "timestamp": fg.get("timestamp"),
        "prev": {k: round(fg[f"previous_{k}"], 1) for k in ("close", "1_week", "1_month", "1_year")
                 if f"previous_{k}" in fg},
        "components": comps,
        "history": {
            datetime.fromtimestamp(p["x"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"): round(p["y"], 1)
            for p in hist
        },
    }


def fetch_putcall(max_expiries: int = 6) -> dict:
    """当前期权链的 put/call 成交量与持仓量快照。Yahoo 不提供历史，所以只做参考、不计分。"""
    out = {}
    for t in OPTION_TICKERS:
        try:
            tk = yf.Ticker(t)
            pv = cv = poi = coi = 0.0
            for exp in tk.options[:max_expiries]:
                ch = tk.option_chain(exp)
                pv += ch.puts["volume"].fillna(0).sum()
                cv += ch.calls["volume"].fillna(0).sum()
                poi += ch.puts["openInterest"].fillna(0).sum()
                coi += ch.calls["openInterest"].fillna(0).sum()
            out[t] = {
                "put_volume": int(pv), "call_volume": int(cv),
                "volume_ratio": round(pv / cv, 3) if cv else None,
                "put_oi": int(poi), "call_oi": int(coi),
                "oi_ratio": round(poi / coi, 3) if coi else None,
            }
        except Exception as e:
            print(f"  ! {t} 期权数据获取失败: {e}")
    return out


# ---------------------------------------------------------------- 指标计算

def zscore_to_score(raw: pd.Series, direction: int) -> pd.Series:
    mean = raw.rolling(Z_WINDOW, min_periods=Z_MIN).mean()
    std = raw.rolling(Z_WINDOW, min_periods=Z_MIN).std()
    z = (raw - mean) / std.replace(0, np.nan) * direction
    erf = np.vectorize(lambda v: math.erf(v) if np.isfinite(v) else np.nan)
    return pd.Series(50 * (1 + erf(z.to_numpy() / math.sqrt(2))), index=raw.index)


def compute(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    close = df["Close"].ffill(limit=3)
    volume = df["Volume"]
    sox = close["^SOX"]
    ret20 = close / close.shift(20) - 1

    stocks = [t for t in CONSTITUENTS if t in close and close[t].notna().sum() > 0]
    px = close[stocks]

    # 股价强度：接近 52 周新高的家数减去接近新低的家数，占有效家数的比例（CNN 的做法）
    hi = px.rolling(252, min_periods=200).max()
    lo = px.rolling(252, min_periods=200).min()
    has_range = hi.notna() & px.notna()   # 已退市的不再计入分母
    near_hi = ((px >= hi * (1 - NEAR_HL)) & has_range).sum(axis=1)
    near_lo = ((px <= lo * (1 + NEAR_HL)) & has_range).sum(axis=1)
    net_hl = (near_hi - near_lo) / has_range.sum(axis=1).replace(0, np.nan)

    # 股价广度：上涨股成交量减下跌股成交量，占总成交量的比例，取 10 日均值。
    # 比"站上均线比例"更接近 CNN 的 McClellan 成交量指标，也和动能重复得少
    chg = px.diff()
    vol = volume[stocks]
    net_vol = ((vol * (chg > 0)).sum(axis=1) - (vol * (chg < 0)).sum(axis=1)) / vol.sum(axis=1).replace(0, np.nan)
    breadth = net_vol.rolling(10).mean()

    # Yahoo 的 VXN 盘中不更新、偶有缺失：缺失日按 VIX 当天相对上一个 VXN 有值日的涨跌推算，
    # 否则波动率分项会停在上一交易日
    vix = df["Close"]["^VIX"].ffill()
    if "^VXN" in df["Close"]:
        vxn = df["Close"]["^VXN"]
        est = vxn.ffill() * vix / vix.where(vxn.notna()).ffill()
        vol_idx = vxn.fillna(est)
    else:
        vol_idx = vix

    # 期权情绪：VIX 与 3 个月 VIX 之比。近月隐含波动率高于远月（倒挂）是期权市场的恐慌信号
    term = close["^VIX"] / close["^VIX3M"]

    hyg_ief = np.log(close["HYG"] / close["IEF"])

    # 杠杆资金：先算每天 SOXL 与 SOXS 成交额之比，再取 10 日均值。
    # 盘中当天成交量不完整，但当天的比值仍然有意义，不会被稀释
    dollar = (close[["SOXL", "SOXS"]] * volume[["SOXL", "SOXS"]]).replace(0, np.nan)
    lev = np.log(dollar["SOXL"] / dollar["SOXS"]).rolling(10).mean()

    # 每个分项：原始值、方向（+1 越高越贪婪，-1 越高越恐慌）、展示格式
    specs = [
        dict(key="momentum", name="板块动能", cnn="Market Momentum",
             raw=sox / sox.rolling(125).mean() - 1, direction=1, fmt="pct",
             label="费城半导体指数 SOX 相对 125 日均线",
             desc="SOX 高于 125 日均线越多，追涨情绪越强。对应 CNN 的 S&P 500 相对 125 日均线。"),
        dict(key="strength", name="股价强度", cnn="Stock Price Strength",
             raw=net_hl, direction=1, fmt="pp_share",
             label="接近 52 周新高的家数减接近新低的家数",
             desc="距 52 周高点 3% 以内算接近新高，距低点 3% 以内算接近新低，按占成分股的比例计。对应 CNN 的 52 周新高与新低家数之差。"),
        dict(key="breadth", name="股价广度", cnn="Stock Price Breadth",
             raw=breadth, direction=1, fmt="pp_share",
             label="上涨股成交量减下跌股成交量，占总成交量（10 日均值）",
             desc="资金是在普遍买入还是普遍卖出。对应 CNN 的 McClellan 成交量累积指数。"),
        dict(key="options", name="期权情绪", cnn="Put/Call Options",
             raw=term, direction=-1, fmt="ratio",
             label="VIX 与 3 个月 VIX 之比",
             desc="比值高于 1 叫倒挂，说明市场愿意为近期保护付更高的价，是恐慌信号。CNN 用 put/call 比率；Yahoo 没有期权历史，改用同样来自期权价格的期限结构。"),
        dict(key="volatility", name="波动率", cnn="Market Volatility",
             raw=vol_idx / vol_idx.rolling(50).mean() - 1, direction=-1, fmt="pct",
             label="纳指波动率 VXN 相对 50 日均线",
             desc="VXN 中半导体权重高，比 VIX 更贴近板块。高于均线代表恐慌。对应 CNN 的 VIX 相对 50 日均线。"),
        dict(key="safe_haven", name="避险需求", cnn="Safe Haven Demand",
             raw=ret20["^SOX"] - ret20["TLT"], direction=1, fmt="pp",
             label="SOX 与长期美债 TLT 的 20 日涨幅差",
             desc="资金从半导体流向国债时为恐慌。对应 CNN 的股票与国债 20 日收益差。"),
        dict(key="junk_bond", name="垃圾债需求", cnn="Junk Bond Demand",
             raw=hyg_ief - hyg_ief.rolling(50).mean(), direction=1, fmt="pct",
             label="高收益债 HYG 与中期美债 IEF 的比价，相对 50 日均值",
             desc="信用风险偏好，宏观层面的风险胃口。高收益债相对国债走强代表利差收窄、偏贪婪。对应 CNN 的垃圾债与投资级债利差。"),
        dict(key="leverage", name="杠杆资金", cnn="（半导体特有）",
             raw=lev, direction=1, fmt="x",
             label="3 倍做多 SOXL 与 3 倍做空 SOXS 的成交额之比（10 日均值）",
             desc="散户在半导体上的杠杆押注方向。做多成交额占比越高越贪婪。"),
    ]

    raws = pd.DataFrame({s["key"]: s["raw"] for s in specs})
    scores = pd.DataFrame({s["key"]: zscore_to_score(s["raw"], s["direction"]) for s in specs})
    enough = scores.notna().sum(axis=1) >= 6
    scores["index"] = scores[[s["key"] for s in specs]].mean(axis=1, skipna=True).where(enough)
    scores["sox"] = sox
    scores = scores[scores["index"].notna()]
    print(f"  成分股 {len(stocks)} 只，指数有效区间 {scores.index[0].date()} ~ {scores.index[-1].date()}")
    return raws.loc[scores.index], scores, specs


def forward_returns(scores: pd.DataFrame, horizon: int = 21) -> list[dict]:
    """按指数分档统计 SOX 之后约一个月的表现，用来检验指数的反向指标属性。"""
    fwd = scores["sox"].shift(-horizon) / scores["sox"] - 1
    zone = scores["index"].map(rating)
    rows = []
    for _, name in RATINGS:
        f = fwd[(zone == name) & fwd.notna()]
        rows.append({
            "rating": name, "days": int(len(f)),
            "mean": round(float(f.mean()), 4) if len(f) else None,
            "median": round(float(f.median()), 4) if len(f) else None,
            "win_rate": round(float((f > 0).mean()), 3) if len(f) else None,
        })
    return rows


# ---------------------------------------------------------------- 输出

def value_ago(s: pd.Series, days: int) -> float | None:
    target = s.index[-1] - pd.Timedelta(days=days)
    past = s[s.index <= target]
    return round(float(past.iloc[-1]), 1) if len(past) else None


def last_putcall() -> tuple[dict, str | None]:
    """不抓期权时，沿用日志里最近一次的快照。"""
    if not PUTCALL_LOG.exists():
        return {}, None
    log = pd.read_csv(PUTCALL_LOG)
    if log.empty:
        return {}, None
    date = log["date"].max()
    rows = log[log["date"] == date]
    snap = {r.pop("ticker"): {k: (None if pd.isna(v) else v) for k, v in r.items() if k != "date"}
            for r in rows.to_dict("records")}
    return snap, date


def build_payload(raws, scores, specs, cnn, putcall, putcall_date, market) -> dict:
    idx = scores["index"]
    last = scores.index[-1]
    comps = []
    year = scores.index >= last - pd.Timedelta(days=365)
    for s in specs:
        k = s["key"]
        comps.append({
            "key": k, "name": s["name"], "cnn": s["cnn"], "label": s["label"], "desc": s["desc"],
            "fmt": s["fmt"], "direction": s["direction"],
            "raw": round(float(raws[k].iloc[-1]), 5) if pd.notna(raws[k].iloc[-1]) else None,
            "score": round(float(scores[k].iloc[-1]), 1) if pd.notna(scores[k].iloc[-1]) else None,
            "rating": rating(scores[k].iloc[-1]),
            "week_ago": value_ago(scores[k].dropna(), 7),
            "spark": [None if pd.isna(v) else round(float(v), 1) for v in scores[k][year]],
        })
    score = round(float(idx.iloc[-1]), 1)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        "as_of": last.strftime("%Y-%m-%d"),
        "score": score,
        "rating": rating(score),
        "prev": {
            "close": round(float(idx.iloc[-2]), 1),
            "1_week": value_ago(idx, 7),
            "1_month": value_ago(idx, 30),
            "1_year": value_ago(idx, 365),
        },
        "sox_last": round(float(scores["sox"].iloc[-1]), 2),
        "sox_chg": round(float(scores["sox"].iloc[-1] / scores["sox"].iloc[-2] - 1), 4),
        "components": comps,
        "history": {
            "dates": [d.strftime("%Y-%m-%d") for d in scores.index],
            "index": [round(float(v), 1) for v in idx],
            "sox": [round(float(v), 2) for v in scores["sox"]],
        },
        "forward": forward_returns(scores),
        "cnn": cnn,
        "putcall": putcall,
        "putcall_date": putcall_date,
        "market": market,
        "params": {"z_window": Z_WINDOW, "constituents": CONSTITUENTS},
    }


def write_outputs(payload: dict, scores: pd.DataFrame, raws: pd.DataFrame) -> Path:
    OUT.mkdir(exist_ok=True)
    (OUT / "semi_fgi.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    table = scores.round(2).join(raws.add_prefix("raw_").round(5))
    table.index.name = "date"
    table.to_csv(OUT / "semi_fgi_history.csv")

    if payload["putcall"] and payload["putcall_date"] == payload["as_of"]:
        DATA.mkdir(exist_ok=True)
        log = PUTCALL_LOG
        rows = [{"date": payload["as_of"], "ticker": t, **v} for t, v in payload["putcall"].items()]
        new = pd.DataFrame(rows)
        if log.exists():
            old = pd.read_csv(log)
            old = old[~((old["date"] == payload["as_of"]) & old["ticker"].isin(new["ticker"]))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(log, index=False)

    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    path = OUT / "semi_fgi.html"
    path.write_text(html, encoding="utf-8")
    return path


RATING_CN = {"extreme_fear": "极度恐慌", "fear": "恐慌", "neutral": "中性",
             "greed": "贪婪", "extreme_greed": "极度贪婪"}


def run(options: bool = True, cnn: bool = True) -> dict:
    OUT.mkdir(exist_ok=True)
    market = market_status()
    print("下载行情 …")
    df = load_prices(MARKET + CONSTITUENTS)
    print("计算分项 …")
    raws, scores, specs = compute(df)
    cnn_data = fetch_cnn() if cnn else None
    as_of = scores.index[-1].strftime("%Y-%m-%d")
    if options:
        putcall, putcall_date = fetch_putcall(), as_of
    else:
        putcall, putcall_date = last_putcall()
    payload = build_payload(raws, scores, specs, cnn_data, putcall, putcall_date, market)
    path = write_outputs(payload, scores, raws)

    print(f"\n半导体恐慌贪婪指数  {payload['as_of']}:  {payload['score']:.1f}  "
          f"{RATING_CN[payload['rating']]}")
    p = payload["prev"]
    print(f"  前收 {p['close']}  一周前 {p['1_week']}  一月前 {p['1_month']}  一年前 {p['1_year']}")
    for c in payload["components"]:
        print(f"  {c['name']:<6} {c['score'] if c['score'] is not None else '-':>5}  "
              f"{RATING_CN.get(c['rating'], '-')}")
    if cnn_data:
        print(f"  CNN 大盘指数 {cnn_data['score']:.1f} {RATING_CN.get(cnn_data['rating'], cnn_data['rating'])}")
    print(f"\n已生成 {path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="美股半导体恐慌贪婪指数")
    ap.add_argument("--no-options", action="store_true", help="跳过期权 put/call 快照，沿用最近一次")
    ap.add_argument("--no-cnn", action="store_true", help="不抓取 CNN 指数做对比")
    args = ap.parse_args()
    run(options=not args.no_options, cnn=not args.no_cnn)


if __name__ == "__main__":
    main()
