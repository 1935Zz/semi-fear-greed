"""GitHub Actions 入口：判断这次定时触发要不要更新，需要的话计算并把网页放进 site/。

- 盘中（纽交所开盘到收盘）：每次触发都更新，不抓期权快照。最新数据写进 live/live.json，
  由工作流推到 live 分支，网页通过 GitHub API 直接读取（几秒内生效）。
  GitHub Pages 发布成功后实测要再过 2–5 分钟才真正换成新文件，所以 Pages 只每 15 分钟发布一次作备份。
- 收盘 15 分钟后的第一次触发：做当天的收盘版，抓期权快照，并把当天数据写进 data/ 由工作流提交。
- 其余时间（盘前、休市、当天收盘版已完成）：跳过，网站保持上一次的内容。

环境变量 FORCE=true 时无论什么时间都更新一次（手动触发用）。
个股看盘的基本面和新闻抓取较慢（1–3 分钟），放在工作流里单独的 stocks 任务（stocks.py --ci），不拖慢指数更新。
结果通过 GITHUB_OUTPUT 输出 live、pages、commit、save_cache（true/false）。
"""

import json
import os
import shutil
from pathlib import Path

import pandas as pd

import semi_fgi as m
import stocks

FINAL_DELAY = pd.Timedelta(minutes=15)   # 收盘后等 Yahoo 的日线数据定稿
PAGES_EVERY = 15                         # 盘中每隔多少分钟发布一次 GitHub Pages
PAGES_WINDOW = 3                         # 每个周期开头几分钟内的触发负责发布（等于触发间隔）
SITE = m.ROOT / "site"
LIVE = m.ROOT / "live"
FINAL_HISTORY = m.DATA / "semi_fgi_history.csv"


def last_final_date() -> str | None:
    if not FINAL_HISTORY.exists():
        return None
    return pd.read_csv(FINAL_HISTORY, usecols=["date"])["date"].max()


def decide(now: pd.Timestamp, force: bool) -> str:
    st = m.market_status(now)
    if st["state"] == "open":
        return "intraday"
    if st["trading_day"] and st["state"] == "closed":
        after_close = now >= pd.Timestamp(st["close"]) + FINAL_DELAY
        if after_close and (last_final_date() or "") < st["date"]:
            return "final"
    if force:
        return "intraday" if st["state"] == "open" else "final"
    return "skip"


def slim(p: dict) -> dict:
    """网页轮询用的精简版：去掉整段历史和各分项的走势，只留最新一个点，约 6 KB。"""
    live = {k: v for k, v in p.items() if k != "history"}
    live["components"] = [{k: v for k, v in c.items() if k != "spark"} for c in p["components"]]
    if p.get("cnn"):
        cnn = dict(p["cnn"])
        hist = cnn.pop("history", {})
        last = max(hist) if hist else None
        cnn["last"] = {"date": last, "value": hist[last]} if last else None
        live["cnn"] = cnn
    h = p["history"]
    live["last"] = {"date": h["dates"][-1], "index": h["index"][-1], "sox": h["sox"][-1]}
    return live


def quotes(df: pd.DataFrame | None) -> dict:
    """个股看盘页用的最新价和前收：{代码: [最新价, 前收]}。"""
    if df is None:
        return {}
    out = {}
    for sym in stocks.TICKERS:
        if sym in df["Close"]:
            c = df["Close"][sym].dropna()
            if len(c) >= 2:
                out[sym] = [round(float(c.iloc[-1]), 2), round(float(c.iloc[-2]), 2)]
    return out


def set_output(**kv) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={str(v).lower()}" for k, v in kv.items()]
    print("\n".join(lines))
    if path:
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    force = os.environ.get("FORCE") == "true"
    mode = decide(now, force)
    print(f"{now.tz_convert(m.ET):%Y-%m-%d %H:%M} ET  模式: {mode}")
    if mode == "skip":
        set_output(live=False, pages=False, commit=False, save_cache=False)
        return

    payload = m.run(options=(mode == "final"))

    live = slim(payload)
    live["quotes"] = quotes(m.LAST_DF)
    LIVE.mkdir(exist_ok=True)
    (LIVE / "live.json").write_text(json.dumps(live, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    SITE.mkdir(exist_ok=True)
    shutil.copy(m.OUT / "semi_fgi.html", SITE / "index.html")
    shutil.copy(m.OUT / "semi_fgi.json", SITE / "semi_fgi.json")
    (SITE / ".nojekyll").touch()
    # 个股看盘页不内嵌数据：打开时读 stocks 分支上的 stocks.json（由工作流里的 stocks 任务刷新）
    (SITE / "stocks.html").write_text(stocks.render(None, {
        "stocks_url": os.environ.get("STOCKS_URL"), "live_url": os.environ.get("LIVE_URL")}), encoding="utf-8")
    if mode == "final":
        m.DATA.mkdir(exist_ok=True)
        shutil.copy(m.OUT / "semi_fgi_history.csv", FINAL_HISTORY)
    pages = force or mode == "final" or now.minute % PAGES_EVERY < PAGES_WINDOW
    fresh = m.LAST_LOAD.get("full") and m.LAST_LOAD.get("complete")
    set_output(live=True, pages=pages, commit=(mode == "final"), save_cache=bool(fresh))


if __name__ == "__main__":
    main()
