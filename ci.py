"""GitHub Actions 入口：判断这次定时触发要不要更新，需要的话计算并把网页放进 site/。

- 盘中（纽交所开盘到收盘）：每次触发都更新，不抓期权快照。
- 收盘 15 分钟后的第一次触发：做当天的收盘版，抓期权快照，并把当天数据写进 data/ 由工作流提交。
- 其余时间（盘前、休市、当天收盘版已完成）：跳过，网站保持上一次的内容。

环境变量 FORCE=true 时无论什么时间都更新一次（手动触发用）。
结果通过 GITHUB_OUTPUT 输出 deploy=true/false、commit=true/false。
"""

import os
import shutil
from pathlib import Path

import pandas as pd

import semi_fgi as m

FINAL_DELAY = pd.Timedelta(minutes=15)   # 收盘后等 Yahoo 的日线数据定稿
SITE = m.ROOT / "site"
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


def set_output(**kv) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={str(v).lower()}" for k, v in kv.items()]
    print("\n".join(lines))
    if path:
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    mode = decide(now, os.environ.get("FORCE") == "true")
    print(f"{now.tz_convert(m.ET):%Y-%m-%d %H:%M} ET  模式: {mode}")
    if mode == "skip":
        set_output(deploy=False, commit=False)
        return

    m.run(options=(mode == "final"))

    SITE.mkdir(exist_ok=True)
    shutil.copy(m.OUT / "semi_fgi.html", SITE / "index.html")
    shutil.copy(m.OUT / "semi_fgi.json", SITE / "semi_fgi.json")
    (SITE / ".nojekyll").touch()
    if mode == "final":
        m.DATA.mkdir(exist_ok=True)
        shutil.copy(m.OUT / "semi_fgi_history.csv", FINAL_HISTORY)
    set_output(deploy=True, commit=(mode == "final"))


if __name__ == "__main__":
    main()
