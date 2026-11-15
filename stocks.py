"""半导体个股看盘数据：估值、营收与 EPS 趋势、业绩超预期情况、分析师预期变化、评级、新闻，以及据此整理的要点。

数据来自 Yahoo Finance（yfinance）。新闻优先用 Yahoo；Yahoo 新闻接口限流或失败时改用 Google News RSS。
要点只陈述数据（增长、预期变化、估值相对同行的位置、分析师观点），不给买卖建议。

用法:
    .venv/bin/python stocks.py              # 刷新基本面和新闻，生成 output/stocks.html
    .venv/bin/python stocks.py --news-only  # 只刷新新闻（基本面沿用上次）
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import semi_fgi as m

STOCKS_CACHE = m.ROOT / "cache_stocks" / "stocks.json"
TEMPLATE = m.ROOT / "template_stocks.html"

# 美股上市、市值最大的 16 家半导体公司（含 ADR）
STOCKS = [
    ("NVDA", "英伟达", "AI 加速器 / GPU"),
    ("AVGO", "博通", "网络芯片 / 定制 AI 芯片"),
    ("TSM", "台积电", "晶圆代工"),
    ("AMD", "AMD", "CPU / GPU"),
    ("ASML", "阿斯麦", "光刻设备"),
    ("MU", "美光", "存储（DRAM / HBM / NAND）"),
    ("QCOM", "高通", "手机 / 汽车芯片"),
    ("AMAT", "应用材料", "半导体设备"),
    ("LRCX", "泛林", "刻蚀 / 沉积设备"),
    ("KLAC", "科磊", "检测量测设备"),
    ("TXN", "德州仪器", "模拟芯片"),
    ("ADI", "亚德诺", "模拟 / 混合信号"),
    ("ARM", "Arm", "芯片架构授权"),
    ("MRVL", "迈威尔", "数据中心网络 / 定制芯片"),
    ("INTC", "英特尔", "CPU / 代工"),
    ("NXPI", "恩智浦", "汽车 / 工业芯片"),
]
TICKERS = [s[0] for s in STOCKS]

NEWS_MAX_AGE = 60 * 60          # 新闻超过 1 小时就刷新
NEWS_KEEP = 6
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


# ---------------------------------------------------------------- 工具

def num(x):
    """转成可以写进 JSON 的数字，缺失或非有限值返回 None。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rnd(x, n=4):
    v = num(x)
    return None if v is None else round(v, n)


def retry(f, tries=3, wait=2.0):
    for i in range(tries):
        try:
            return f()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def safe(f, default=None):
    try:
        return retry(f)
    except Exception as e:
        print(f"    ! {getattr(f, '__name__', '')} 失败: {str(e)[:80]}")
        return default


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- 基本面

def quarterly(t: yf.Ticker) -> list[dict]:
    q = t.quarterly_income_stmt
    if q is None or q.empty:
        return []
    q = q.T.sort_index()
    rows = []
    for d, r in q.tail(8).iterrows():
        rev, gp = num(r.get("Total Revenue")), num(r.get("Gross Profit"))
        if rev is None:      # 早期季度 Yahoo 有时只给部分科目，没有营收的整行跳过
            continue
        rows.append({
            "period": pd.Timestamp(d).strftime("%Y-%m"),
            "revenue": rev,
            "eps": rnd(r.get("Diluted EPS"), 3),
            "gross_margin": rnd(gp / rev) if rev and gp is not None else None,
            "op_income": num(r.get("Operating Income")),
            "net_income": num(r.get("Net Income")),
        })
    # 同比、环比按相邻季度和 4 个季度前计算（Yahoo 通常给 5 个季度，所以最近一季有同比）
    for i, r in enumerate(rows):
        prev = rows[i - 1]["revenue"] if i >= 1 else None
        yago = rows[i - 4]["revenue"] if i >= 4 else None
        r["rev_qoq"] = rnd(r["revenue"] / prev - 1) if r["revenue"] and prev else None
        r["rev_yoy"] = rnd(r["revenue"] / yago - 1) if r["revenue"] and yago else None
    return rows


def frame_rows(df: pd.DataFrame | None, cols: list[str]) -> dict:
    if df is None or df.empty:
        return {}
    out = {}
    for idx, r in df.iterrows():
        out[str(idx)] = {c: rnd(r.get(c), 4) for c in cols if c in r}
    return out


def fx_usd(cur: str | None) -> float:
    """1 单位外币等于多少美元。"""
    if not cur or cur == "USD":
        return 1.0
    h = yf.Ticker(f"{cur}USD=X").history(period="5d")["Close"].dropna()
    return float(h.iloc[-1])


def to_usd(d: dict, sym: str, info: dict) -> None:
    """台积电、阿斯麦等公司的财报以本币披露，Yahoo 的季报、营收预期、TTM 营收也是本币，
    而价格、市值、EPS 预期是美元。这里把营收类数据统一换成美元，并重算市销率和 EV/EBITDA。"""
    cur = d.get("currency")
    if not cur or cur == "USD":
        return
    r = fx_usd(cur)
    d["fx"] = {"currency": cur, "usd": round(r, 6)}
    for q in d["quarters"]:
        for k in ["revenue", "op_income", "net_income"]:
            if q.get(k) is not None:
                q[k] *= r
        if q.get("eps") is not None:
            q["eps"] = round(q["eps"] * r, 3)   # Yahoo 给的已是每份 ADR 的本币 EPS
    for e in d["rev_est"].values():
        for k in ["avg", "low", "high", "yearAgoRevenue"]:
            if e.get(k) is not None:
                e[k] *= r
    if d.get("next_earnings", {}).get("rev_avg"):
        d["next_earnings"]["rev_avg"] *= r
    if d.get("revenue_ttm"):
        d["revenue_ttm"] *= r
        d["ps_ttm"] = rnd(d["market_cap"] / d["revenue_ttm"], 2) if d.get("market_cap") else None
    debt, cash, ebitda = num(info.get("totalDebt")), num(info.get("totalCash")), num(info.get("ebitda"))
    if d.get("market_cap") and ebitda and ebitda > 0:
        ev = d["market_cap"] + ((debt or 0) - (cash or 0)) * r
        d["ev_ebitda"] = rnd(ev / (ebitda * r), 2)
    else:
        d["ev_ebitda"] = None


def fundamentals(sym: str) -> dict:
    t = yf.Ticker(sym)
    info = safe(t.get_info, {}) or {}
    d = {
        "updated": now_iso(),
        "price": num(info.get("currentPrice") or info.get("regularMarketPrice")),
        "prev_close": num(info.get("regularMarketPreviousClose") or info.get("previousClose")),
        "market_cap": num(info.get("marketCap")),
        "shares": num(info.get("sharesOutstanding")),
        "currency": info.get("financialCurrency") or info.get("currency"),
        "fundamentals_price": num(info.get("currentPrice") or info.get("regularMarketPrice")),
        "pe_ttm": rnd(info.get("trailingPE"), 2),
        "pe_fwd": rnd(info.get("forwardPE"), 2),
        "eps_ttm": rnd(info.get("trailingEps"), 3),
        "eps_fwd": rnd(info.get("forwardEps"), 3),
        "ps_ttm": rnd(info.get("priceToSalesTrailing12Months"), 2),
        "ev_ebitda": rnd(info.get("enterpriseToEbitda"), 2),
        "peg": rnd(info.get("trailingPegRatio") or info.get("pegRatio"), 2),
        "revenue_ttm": num(info.get("totalRevenue")),
        "gross_margin": rnd(info.get("grossMargins")),
        "op_margin": rnd(info.get("operatingMargins")),
        "net_margin": rnd(info.get("profitMargins")),
        "rev_growth": rnd(info.get("revenueGrowth")),
        "earn_growth": rnd(info.get("earningsGrowth")),
        "high_52w": num(info.get("fiftyTwoWeekHigh")),
        "low_52w": num(info.get("fiftyTwoWeekLow")),
        "beta": rnd(info.get("beta"), 2),
        "target": {
            "mean": num(info.get("targetMeanPrice")), "high": num(info.get("targetHighPrice")),
            "low": num(info.get("targetLowPrice")), "n": info.get("numberOfAnalystOpinions"),
            "rec_mean": rnd(info.get("recommendationMean"), 2), "rec_key": info.get("recommendationKey"),
        },
    }
    d["quarters"] = safe(lambda: quarterly(t), []) or []

    eh = safe(lambda: t.earnings_history)
    d["surprises"] = [] if eh is None or eh.empty else [
        {"quarter": pd.Timestamp(i).strftime("%Y-%m"), "actual": rnd(r["epsActual"], 3),
         "estimate": rnd(r["epsEstimate"], 3), "surprise": rnd(r["surprisePercent"])}
        for i, r in eh.sort_index().tail(4).iterrows()]

    d["eps_est"] = frame_rows(safe(lambda: t.earnings_estimate),
                              ["avg", "low", "high", "yearAgoEps", "numberOfAnalysts", "growth"])
    d["rev_est"] = frame_rows(safe(lambda: t.revenue_estimate),
                              ["avg", "low", "high", "yearAgoRevenue", "numberOfAnalysts", "growth"])
    d["eps_trend"] = frame_rows(safe(lambda: t.eps_trend),
                                ["current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo"])
    d["eps_rev"] = frame_rows(safe(lambda: t.eps_revisions),
                              ["upLast7days", "upLast30days", "downLast30days", "downLast7Days"])

    rec = safe(lambda: t.recommendations)
    if rec is not None and not rec.empty:
        r0 = rec.iloc[0]
        d["ratings"] = {k: int(r0.get(k, 0) or 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]}
        if len(rec) > 1:
            r1 = rec.iloc[1]
            d["ratings_prev"] = {k: int(r1.get(k, 0) or 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]}

    ud = safe(lambda: t.upgrades_downgrades)
    if ud is not None and not ud.empty:
        ud = ud.sort_index(ascending=False)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        recent = ud[ud.index >= cutoff]
        d["rating_changes_30d"] = {
            "n": int(len(recent)),
            "up": int((recent["Action"] == "up").sum()),
            "down": int((recent["Action"] == "down").sum()),
            "pt_raise": int((recent.get("priceTargetAction", pd.Series(dtype=str)) == "Raises").sum()),
            "pt_lower": int((recent.get("priceTargetAction", pd.Series(dtype=str)) == "Lowers").sum()),
        }
        d["rating_changes"] = [
            {"date": pd.Timestamp(i).strftime("%Y-%m-%d"), "firm": r.get("Firm"), "to": r.get("ToGrade"),
             "from": r.get("FromGrade"), "action": r.get("Action"),
             "pt": num(r.get("currentPriceTarget")), "pt_prev": num(r.get("priorPriceTarget"))}
            for i, r in ud.head(6).iterrows()]

    cal = safe(lambda: t.calendar, {}) or {}
    ed = cal.get("Earnings Date") or []
    if ed:
        d["next_earnings"] = {
            "date": str(min(ed)), "eps_avg": rnd(cal.get("Earnings Average"), 3),
            "rev_avg": num(cal.get("Revenue Average")),
        }
    try:     # 不能用 safe()：它会重试，而换算是原地修改
        to_usd(d, sym, info)
    except Exception as e:
        print(f"    ! {sym} 换算美元失败: {str(e)[:80]}")
    return d


# ---------------------------------------------------------------- 新闻

def news_yahoo(sym: str) -> list[dict]:
    items = yf.Ticker(sym).get_news(count=12, tab="news")
    out = []
    for a in items:
        c = a.get("content", a)
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
        if not c.get("title") or not url:
            continue
        out.append({"title": c["title"], "source": (c.get("provider") or {}).get("displayName", "Yahoo Finance"),
                    "time": c.get("pubDate") or c.get("displayTime"), "url": url})
    return out


def news_google(sym: str, name_en: str) -> list[dict]:
    q = requests.utils.quote(f'"{name_en}" OR {sym} stock when:7d')
    r = requests.get(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
                     headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    out = []
    for it in re.findall(r"<item>(.*?)</item>", r.text, re.S):
        g = lambda k: (re.search(fr"<{k}[^>]*>(.*?)</{k}>", it, re.S) or [None, ""])[1]
        title, source = html.unescape(g("title")), html.unescape(g("source"))
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3]
        try:
            ts = parsedate_to_datetime(g("pubDate")).astimezone(timezone.utc).isoformat(timespec="seconds")
        except Exception:
            ts = None
        out.append({"title": title, "source": source or "Google News", "time": ts, "url": html.unescape(g("link"))})
    return out


def fetch_news(sym: str, name_en: str) -> tuple[list[dict], str]:
    try:
        items, src = retry(lambda: news_yahoo(sym), tries=2), "Yahoo Finance"
        if not items:
            raise ValueError("Yahoo 新闻为空")
    except Exception:
        items, src = retry(lambda: news_google(sym, name_en), tries=2), "Google News"
    seen, uniq = set(), []
    for a in sorted(items, key=lambda a: a.get("time") or "", reverse=True):
        key = re.sub(r"\W+", "", a["title"].lower())[:60]
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    return uniq[:NEWS_KEEP], src


# ---------------------------------------------------------------- 要点

def pct(x, digits=0, sign=True):
    if x is None:
        return "—"
    s = f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"
    return s.replace("-", "−")


def money(x):
    """美元金额写成亿 / 万亿。"""
    if x is None:
        return "—"
    if abs(x) >= 1e12:
        return f"{x / 1e12:.2f} 万亿美元"
    return f"{x / 1e8:,.0f} 亿美元"


def notes(sym: str, d: dict, peers: dict) -> list[dict]:
    """把数据整理成几条要点，每条带一个倾向（pos / neg / neu）用来标颜色。只陈述事实，不下买卖结论。"""
    out = []
    q = [r for r in d.get("quarters", []) if r.get("revenue")]
    if q:
        last = q[-1]
        parts = [f"最近一季（{last['period']}）营收 {money(last['revenue'])}"]
        if last.get("rev_yoy") is not None:
            parts.append(f"同比 {pct(last['rev_yoy'])}")
        if last.get("rev_qoq") is not None:
            parts.append(f"环比 {pct(last['rev_qoq'])}")
        if last.get("eps") is not None:
            parts.append(f"EPS {last['eps']:.2f}")
        tone = "neu" if last.get("rev_yoy") is None else ("pos" if last["rev_yoy"] > 0.1 else "neg" if last["rev_yoy"] < 0 else "neu")
        out.append({"tag": "业绩", "tone": tone, "text": "，".join(parts) + "。"})

    s = d.get("surprises") or []
    s = [x for x in s if x.get("surprise") is not None]
    if s:
        beats = sum(x["surprise"] > 0 for x in s)
        avg = sum(x["surprise"] for x in s) / len(s)
        tone = "pos" if beats == len(s) else "neg" if beats <= len(s) / 2 else "neu"
        out.append({"tag": "超预期", "tone": tone,
                    "text": f"过去 {len(s)} 季 EPS {beats} 次高于预期，平均偏离 {pct(avg, 1)}。"})

    tr = (d.get("eps_trend") or {}).get("0y") or {}
    rv = (d.get("eps_rev") or {}).get("0y") or {}
    if tr.get("current") and tr.get("90daysAgo"):
        ch = tr["current"] / tr["90daysAgo"] - 1
        up, dn = rv.get("upLast30days"), rv.get("downLast30days")
        txt = f"本财年 EPS 预期 90 天内从 {tr['90daysAgo']:.2f} 变为 {tr['current']:.2f}（{pct(ch, 1)}）"
        if up is not None and dn is not None:
            txt += f"；近 30 天 {int(up)} 家上调、{int(dn)} 家下调"
        tone = "pos" if ch > 0.01 else "neg" if ch < -0.01 else "neu"
        out.append({"tag": "预期", "tone": tone, "text": txt + "。"})

    pe, med = d.get("pe_fwd"), peers.get("pe_fwd")
    if pe and med:
        rel = pe / med - 1
        word = "高于" if rel > 0.1 else "低于" if rel < -0.1 else "接近"
        g = ((d.get("eps_est") or {}).get("+1y") or {}).get("growth")
        txt = f"远期市盈率 {pe:.1f} 倍，{word}这 {len(TICKERS)} 家公司的中位数 {med:.1f} 倍"
        if g is not None:
            txt += f"；分析师预计下一财年 EPS 增长 {pct(g)}"
        out.append({"tag": "估值", "tone": "neu", "text": txt + "。"})

    tg, price = d.get("target") or {}, d.get("price")
    if tg.get("mean") and price:
        up = tg["mean"] / price - 1
        txt = f"{tg.get('n') or '—'} 位分析师平均目标价 {tg['mean']:.2f}，较现价 {pct(up)}"
        rc = d.get("rating_changes_30d")
        if rc and rc["n"]:
            txt += f"；近 30 天评级上调 {rc['up']} 次、下调 {rc['down']} 次，目标价上调 {rc['pt_raise']} 次、下调 {rc['pt_lower']} 次"
        tone = "neg" if rc and rc["down"] + rc["pt_lower"] > rc["up"] + rc["pt_raise"] + 2 else \
               "pos" if rc and rc["up"] + rc["pt_raise"] > rc["down"] + rc["pt_lower"] + 2 else "neu"
        out.append({"tag": "分析师", "tone": tone, "text": txt + "。"})

    if price and d.get("high_52w"):
        off = price / d["high_52w"] - 1
        out.append({"tag": "位置", "tone": "neg" if off < -0.25 else "neu",
                    "text": f"股价较 52 周高点 {d['high_52w']:.2f} {pct(off)}，52 周区间 {d['low_52w']:.2f}–{d['high_52w']:.2f}。"})

    ne = d.get("next_earnings")
    if ne and ne.get("date"):
        days = (pd.Timestamp(ne["date"]) - pd.Timestamp.now(tz=m.ET).normalize().tz_localize(None)).days
        if 0 <= days <= 45:
            txt = f"下次财报 {ne['date']}（{days} 天后）"
            if ne.get("eps_avg") is not None:
                txt += f"，市场预期 EPS {ne['eps_avg']:.2f}"
            if ne.get("rev_avg"):
                txt += f"、营收 {money(ne['rev_avg'])}"
            out.append({"tag": "财报", "tone": "neu", "text": txt + "。"})
    return out


# ---------------------------------------------------------------- 汇总

def load_cache() -> dict:
    try:
        return json.loads(STOCKS_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {"stocks": {}}


def peer_medians(stocks: dict) -> dict:
    out = {}
    for k in ["pe_fwd", "pe_ttm", "ps_ttm", "ev_ebitda", "gross_margin", "op_margin"]:
        vals = [s[k] for s in stocks.values() if s.get(k) and s[k] > 0]
        out[k] = round(float(np.median(vals)), 2) if vals else None
    return out


REV_POINTS = [("90daysAgo", "90 天前"), ("60daysAgo", "60 天前"), ("30daysAgo", "30 天前"),
              ("7daysAgo", "7 天前"), ("current", "当前")]
REV_LOG_KEEP = 800


def sector_revisions(stocks: dict, period: str = "+1y") -> dict:
    """全行业盈利预期修正：分析师上调 / 下调的次数（修正广度）、每家公司预期的变化，
    以及用当前市值加权、以 90 天前为 100 的预期走势。默认看下一财年（period="+1y"，大多对应 2027 年）。"""
    rows, up30, dn30, up7, dn7 = [], 0, 0, 0, 0
    for sym in TICKERS:
        st = stocks.get(sym) or {}
        t = (st.get("eps_trend") or {}).get(period) or {}
        r = (st.get("eps_rev") or {}).get(period) or {}
        u30, d30 = int(r.get("upLast30days") or 0), int(r.get("downLast30days") or 0)
        u7, d7 = int(r.get("upLast7days") or 0), int(r.get("downLast7Days") or 0)
        up30, dn30, up7, dn7 = up30 + u30, dn30 + d30, up7 + u7, dn7 + d7
        cur = t.get("current")
        ch = lambda k: rnd(cur / t[k] - 1) if cur and t.get(k) and t[k] > 0 else None
        rows.append({"symbol": sym, "name": st.get("name"), "eps": cur, "chg7": ch("7daysAgo"), "chg30": ch("30daysAgo"),
                     "chg90": ch("90daysAgo"), "up30": u30, "down30": d30, "up7": u7, "down7": d7, "cap": st.get("market_cap"),
                     "path": [t.get(k) for k, _ in REV_POINTS]})
    # 市值加权：每家公司各时点的预期除以 90 天前的预期，再按当前市值加权
    ok = [r for r in rows if r["cap"] and all(v and v > 0 for v in r["path"])]
    wsum = sum(r["cap"] for r in ok)
    weighted = [round(100 * sum(r["cap"] * r["path"][i] / r["path"][0] for r in ok) / wsum, 2) if wsum else None
                for i in range(len(REV_POINTS))]
    median = [round(100 * float(np.median([r["path"][i] / r["path"][0] for r in ok])), 2) if ok else None
              for i in range(len(REV_POINTS))]
    breadth = lambda u, d: rnd((u - d) / (u + d)) if u + d else None
    net = [r["up30"] - r["down30"] for r in rows]
    for r in rows:
        r.pop("path"); r.pop("cap")
    return {
        "period": period,
        "labels": [lab for _, lab in REV_POINTS],
        "weighted": weighted, "median": median, "n": len(ok),
        "up30": up30, "down30": dn30, "breadth30": breadth(up30, dn30),
        "up7": up7, "down7": dn7, "breadth7": breadth(up7, dn7),
        "net_up": sum(n > 0 for n in net), "net_down": sum(n < 0 for n in net), "net_flat": sum(n == 0 for n in net),
        "stocks": rows,
    }


def log_revisions(data: dict, rev: dict) -> None:
    """每天记一条全行业修正情况（基本面刷新时），积累起来看修正方向什么时候转向。同一天重复刷新只留最后一条。"""
    day = pd.Timestamp.now(tz=m.ET).strftime("%Y-%m-%d")
    w = rev["weighted"]
    entry = {"date": day, "breadth30": rev["breadth30"], "breadth7": rev["breadth7"],
             "up30": rev["up30"], "down30": rev["down30"], "net_up": rev["net_up"], "net_down": rev["net_down"],
             "chg30_w": rnd(w[-1] / w[2] - 1) if w[2] else None, "chg90_w": rnd(w[-1] / 100 - 1) if w[-1] else None}
    log = [e for e in data.get("revision_log", []) if e["date"] != day] + [entry]
    data["revision_log"] = log[-REV_LOG_KEEP:]


def price_history(df: pd.DataFrame | None) -> dict:
    if df is None:
        return {}
    close = df["Close"]
    out = {}
    for sym in TICKERS:
        if sym in close:
            s = close[sym].dropna().tail(260)
            out[sym] = {"dates": [d.strftime("%Y-%m-%d") for d in s.index], "close": [round(float(v), 2) for v in s]}
    return out


def update(full: bool, news: bool, df: pd.DataFrame | None = None) -> dict:
    """full：刷新基本面；news：刷新新闻；df：主程序已下载的行情，用来取一年走势，没有就自己下载。"""
    data = load_cache()
    stocks = data.setdefault("stocks", {})
    if full:
        print(f"刷新 {len(TICKERS)} 只个股基本面 …")
        for sym, cn, seg in STOCKS:
            t0 = time.time()
            try:
                f = fundamentals(sym)
                stocks[sym] = {**stocks.get(sym, {}), **f, "symbol": sym, "name": cn, "segment": seg}
                print(f"  {sym:5s} {time.time() - t0:4.1f}s")
            except Exception as e:
                print(f"  ! {sym} 基本面失败，沿用上次: {str(e)[:80]}")
            time.sleep(0.4)
        data["fundamentals_updated"] = now_iso()
    if news:
        print("刷新新闻 …")
        srcs = set()
        for sym, cn, seg in STOCKS:
            try:
                items, src = fetch_news(sym, NAME_EN.get(sym, sym))
                stocks.setdefault(sym, {"symbol": sym, "name": cn, "segment": seg})["news"] = items
                srcs.add(src)
            except Exception as e:
                print(f"  ! {sym} 新闻失败，沿用上次: {str(e)[:80]}")
            time.sleep(0.3)
        data["news_updated"] = now_iso()
        data["news_source"] = "、".join(sorted(srcs)) or data.get("news_source")

    if df is None:
        df = m.download(TICKERS, "1y")
    hist = price_history(df)
    for sym, cn, seg in STOCKS:
        s = stocks.setdefault(sym, {"symbol": sym, "name": cn, "segment": seg})
        s["name"], s["segment"] = cn, seg
        if sym in hist:
            s["history"] = hist[sym]
            # 用最新行情更新价格（基本面一天才刷新一次，价格每次运行都更新）
            closes = hist[sym]["close"]
            s["price"], s["prev_close"] = closes[-1], closes[-2] if len(closes) > 1 else s.get("prev_close")
    data["order"] = TICKERS
    data["peers"] = peer_medians(stocks)
    data["revisions"] = sector_revisions(stocks)
    if full:
        log_revisions(data, data["revisions"])
    for sym in TICKERS:
        stocks[sym]["notes"] = notes(sym, stocks[sym], data["peers"])
    data["generated_at"] = now_iso()
    STOCKS_CACHE.parent.mkdir(exist_ok=True)
    STOCKS_CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def render(data: dict | None, cfg: dict) -> str:
    """data 为 None 时页面打开后按 cfg["stocks_url"] 自己去读数据（网站用）。"""
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*__CFG__*/null", dumps(cfg))
    return page.replace("/*__DATA__*/null", dumps(data)) if data else page


def write_page(data: dict) -> Path:
    m.OUT.mkdir(exist_ok=True)
    (m.OUT / "stocks.json").write_text(dumps(data), encoding="utf-8")
    path = m.OUT / "stocks.html"
    path.write_text(render(data, {"live_url": os.environ.get("LIVE_URL")}), encoding="utf-8")
    return path


def due(data: dict, now: pd.Timestamp) -> tuple[bool, bool]:
    """基本面每个美东交易日刷新一次（当天第一次运行），新闻每小时刷新一次。"""
    today = now.tz_convert(m.ET).strftime("%Y-%m-%d")
    fu, nu = data.get("fundamentals_updated"), data.get("news_updated")
    full = not fu or pd.Timestamp(fu).tz_convert(m.ET).strftime("%Y-%m-%d") < today \
        or set(TICKERS) - set(data.get("stocks", {}))
    news = full or not nu or (now - pd.Timestamp(nu)).total_seconds() > NEWS_MAX_AGE
    return bool(full), bool(news)


def ci() -> None:
    """GitHub Actions 的个股任务：工作流先把 stocks 分支上的 stocks.json 放到 cache_stocks/，
    到点了就刷新并写到 stocks_out/stocks.json，由工作流推回 stocks 分支。"""
    data = load_cache()
    full, news = due(data, pd.Timestamp.now(tz="UTC"))
    if os.environ.get("FORCE") == "true":
        full = news = True
    print(f"基本面: {'刷新' if full else '沿用'}  新闻: {'刷新' if news else '沿用'}")
    changed = full or news
    if changed:
        data = update(full=full, news=news)
        out = m.ROOT / "stocks_out"
        out.mkdir(exist_ok=True)
        (out / "stocks.json").write_text(dumps(data), encoding="utf-8")
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"changed={str(changed).lower()}\n")


NAME_EN = {"NVDA": "Nvidia", "AVGO": "Broadcom", "TSM": "TSMC", "AMD": "AMD", "ASML": "ASML", "MU": "Micron",
           "QCOM": "Qualcomm", "AMAT": "Applied Materials", "LRCX": "Lam Research", "KLAC": "KLA",
           "TXN": "Texas Instruments", "ADI": "Analog Devices", "ARM": "Arm Holdings", "MRVL": "Marvell",
           "INTC": "Intel", "NXPI": "NXP Semiconductors"}


def main() -> None:
    ap = argparse.ArgumentParser(description="半导体个股看盘数据")
    ap.add_argument("--news-only", action="store_true", help="只刷新新闻")
    ap.add_argument("--ci", action="store_true", help="GitHub Actions 用：按时间判断要刷新什么")
    args = ap.parse_args()
    if args.ci:
        return ci()
    data = update(full=not args.news_only, news=True)
    path = write_page(data)
    for sym in TICKERS:
        s = data["stocks"][sym]
        print(f"{sym:5s} {s.get('price')}  远期PE {s.get('pe_fwd')}  新闻 {len(s.get('news', []))} 条  要点 {len(s.get('notes', []))} 条")
    print(f"\n已生成 {path}")


if __name__ == "__main__":
    main()
