"""
【01 采集层 fetch_roco.py】抓取无 WAF 的《洛克王国:世界》资料站 rocokingdomworld.org

背景:
    原计划主源是 B站BWIKI(MediaWiki API), 但其前置了腾讯 EdgeOne WAF —— requests 连续请求
    会返回 567 JS 挑战页并临时封禁 IP, 脚本爬取不可靠(需数分钟冷却)。经用户确认, 改用本
    无 WAF 的资料站作为主源。它是 SSG 站点, 词条内容是服务端渲染直接落在 HTML 里, 无 JS 渲染。

站点规模(中文 /zh/, 来自 sitemap-0.xml 统计):
    pokedex 631 条   -> 精灵图鉴(种族值/技能/特性/进化/属性克制/获取方式)  -> 归为 beastiary 类
    guides   13 条   -> 培养/孵化/机制等攻略长文                          -> 归为 world 类
    items   819 条   -> 道具图鉴(本次不采, 需要时在 KIND_MAP 里放开即可)
    (其余为多语言/导航/工具页, 不采)

产出(与 02 对接):
    data/raw/pages/<slug>.json 每页一条 {title, kind, url, sha1, fetched_at, html}
    data/raw/manifest.json     汇总 {count, pages:[title...], skipped 统计}
特点: 幂等断点(html 的 sha1 未变则跳过)、轻节流、退出后可 --resume 续抓。

用法:
    python chapter_10-RAG/01_fetch_roco.py [--limit N] [--resume] [--kind beastiary|world]
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests

from config import data_dir  # data 目录统一由 config 管理

# ----------------------------------------------------------------------
# 站点常量
# ----------------------------------------------------------------------
BASE = "https://rocokingdomworld.org"
SITEMAP = f"{BASE}/sitemap-0.xml"
# 目录段(URL path 第 2 段) -> 本项目的 kind。默认不采 items(道具量太大, 非本次目标)。
KIND_MAP = {
    "pokedex": "beastiary",  # 精灵图鉴
    "guides": "world",       # 攻略/机制长文
}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
THROTTLE = 0.3    # 相邻请求间隔(秒)。无 WAF 的普通站点, 适中即可, 勿打太快
MAX_RETRY = 4
TIMEOUT = 25
LOC_RE = re.compile(r"<loc>(.*?)</loc>")

KIND_TAG = {"beastiary": "精灵图鉴", "world": "攻略文本"}


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def slug_of(url: str) -> str:
    """从 URL 取词条 slug 作安全文件名: https://.../zh/pokedex/abu/ -> abu"""
    s = url.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[^0-9a-zA-Z_-]", "_", s) or "untitled"


# ----------------------------------------------------------------------
# 目标清单
# ----------------------------------------------------------------------
def list_targets(kind_allow: set[str]) -> list[dict]:
    """读 sitemap 过滤出 /zh/<kind> 页。返回 [{title, kind, url}], title 用 slug。"""
    resp = requests.get(SITEMAP, headers={"User-Agent": UA}, timeout=30)
    resp.raise_for_status()
    targets = []
    for loc in LOC_RE.findall(resp.text):
        m = re.search(r"/zh/([a-z-]+)/([a-z0-9_-]+)/?$", loc)
        if not m:
            continue
        seg, slug = m.group(1), m.group(2)
        kind = KIND_MAP.get(seg)
        if kind and kind in kind_allow:
            targets.append({"title": slug, "kind": kind, "url": loc})
    return targets


# ----------------------------------------------------------------------
# 抓取
# ----------------------------------------------------------------------
def crawl(client: requests.Session, limit: int | None, resume: bool, kind_allow: set[str]):
    pages_dir = data_dir("raw", "pages")
    manifest_path = data_dir("raw") / "manifest.json"  # 文件路径: 先拿到目录再拼文件名

    targets = list_targets(kind_allow)
    if limit:
        targets = targets[:limit]
    print(f"目标词条: {len(targets)} 条 "
          f"(分类: { {k: KIND_TAG[k] for k in kind_allow} })")

    # 已有断点: {title: sha1} 用于跳过未变页
    seen: dict[str, str] = {}
    if resume:
        for f in pages_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                seen[d["title"]] = d.get("sha1", "")
            except Exception:
                continue
        print(f"断点已存在: {len(seen)} 页(未变化则跳过)")

    fetched = skipped = failed = 0
    for i, t in enumerate(targets, 1):
        title = t["title"]
        try:
            resp = client.get(t["url"], timeout=TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            failed += 1
            print(f"  [{i}/{len(targets)}] 失败 {title}: {type(e).__name__}")
            continue
        html = resp.text
        h = sha1(html)
        if seen.get(title) == h:
            skipped += 1
            continue
        payload = {
            "title": title,
            "kind": t["kind"],
            "url": t["url"],
            "sha1": h,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "html": html,
        }
        (pages_dir / f"{title}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        fetched += 1
        if fetched % 50 == 0:
            print(f"      ...已抓取 {fetched} 页")
        time.sleep(THROTTLE)

    manifest_path.write_text(json.dumps(
        {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "count": len(targets), "kinds": sorted(kind_allow)},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n汇总: 新增抓取 %d 页, 跳过未变化 %d 页, 失败 %d 页" % (fetched, skipped, failed))
    print(f"      原始页落盘: {pages_dir}")


def main():
    p = argparse.ArgumentParser(description="抓取 rocokingdomworld.org 中文词条")
    p.add_argument("--limit", type=int, default=None, help="最多抓取条数(试跑)")
    p.add_argument("--resume", action="store_true", help="断点续抓(跳过 sha1 未变的页)")
    p.add_argument("--kind", default="beastiary,world",
                   help="要抓的分类, 逗号分隔, 可选 beastiary/world")
    args = p.parse_args()
    kind_allow = set(x.strip() for x in args.kind.split(",")) & set(KIND_MAP.values())

    client = requests.Session()
    client.headers["User-Agent"] = UA
    crawl(client, args.limit, args.resume, kind_allow)


if __name__ == "__main__":
    main()
