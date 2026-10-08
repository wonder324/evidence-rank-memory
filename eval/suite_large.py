"""At least 500 synthetic retrieval probes.

Recall@100, Recall@5, MRR, and status accuracy. Not the leaderboard Overall.
The official evaluation set is not used.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_env_override() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_env_override()

from app.config import get_settings
from app.embedder import Embedder
from app.store import MemoryStore

ZH = ["南京", "杭州", "苏州", "无锡", "宁波", "合肥", "福州", "厦门", "青岛", "大连", "沈阳", "长春", "哈尔滨", "济南", "郑州", "武汉", "长沙", "南昌", "广州", "深圳", "珠海", "佛山", "东莞", "南宁", "海口", "成都", "重庆", "昆明", "贵阳", "拉萨", "西安", "兰州", "银川", "西宁", "乌鲁木齐", "太原", "石家庄", "呼和浩特", "天津", "上海"]
EN = ["Boston", "Lisbon", "Denver", "Austin", "Seattle", "Phoenix", "Dallas", "Miami", "Atlanta", "Detroit", "Chicago", "Houston", "Portland", "Tampa", "Oakland", "Madison", "Boulder", "Reno", "Tulsa", "Fresno"]
NAMES = ["周宁", "陈默", "李娜", "王磊", "赵敏", "孙悦", "吴迪", "郑浩", "冯雪", "蒋涛"]
JOBS = ["教师", "医生", "会计", "编辑", "律师", "厨师", "司机", "护士", "设计师", "程序员"]
FOODS = ["川菜", "茶", "咖啡", "面条", "苹果", "酸奶", "面包", "米饭", "鱼", "豆腐"]
DAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _case(user: str, messages: list[str], probes: list[dict]) -> dict:
    return {"user": user, "messages": messages, "probes": probes}


def build_cases() -> list[dict]:
    cases: list[dict] = []
    for index in range(20):
        old, new, noise = ZH[index], ZH[index + 20], ZH[(index + 7) % 20]
        distractors = [
            f"同事住在{noise}。",
            f"新闻里提到{noise}下雨。",
            f"我在{noise}出差时住过酒店。",
            f"朋友计划去{noise}旅游。",
            f"{noise}的会议改期了。",
        ]
        cases.append(
            _case(
                f"zh-move-{index}",
                distractors + [f"我以前住在{old}，现在住在{new}。"],
                [
                    {"category": "zh-current", "query": "我现在住在哪里？", "golds": [new], "status": "current", "forbid_current": [f"我以前 residence = {old}", f"我曾经 residence = {old}"]},
                    {"category": "zh-history", "query": "我以前住在哪里？", "golds": [old], "status": "previous"},
                ],
            )
        )
    for index in range(10):
        old, new = EN[index], EN[index + 10]
        cases.append(
            _case(
                f"en-move-{index}",
                [f"I visited {EN[(index + 3) % 10]} for a conference and stayed one night.", f"I used to live in {old} but now I live in {new}."],
                [
                    {"category": "en-current", "query": "Which city is home now?", "golds": [new], "status": "current"},
                    {"category": "en-history", "query": "Where did I live before?", "golds": [old], "status": "previous"},
                ],
            )
        )
    for index in range(10):
        first, second, third = ZH[index], ZH[index + 10], ZH[index + 20]
        cases.append(
            _case(
                f"zh-chain-{index}",
                [f"我以前住在{first}。", f"后来我搬到了{second}。", f"我现在住在{third}。"],
                [
                    {"category": "chain-current", "query": "我现在住在哪里？", "golds": [third], "status": "current"},
                    {"category": "chain-previous", "query": "我搬到现在这座城市之前住在哪里？", "golds": [second], "status": "previous"},
                ],
            )
        )
    for index in range(10):
        name, job_old, job_new = NAMES[index], JOBS[index], JOBS[(index + 3) % 10]
        food_old, food_new = FOODS[index], FOODS[(index + 4) % 10]
        cases.append(
            _case(
                f"profile-{index}",
                [
                    f"我叫{name}。",
                    f"我是一名{job_old}。",
                    f"我的职业是{job_new}。",
                    f"我以前喜欢{food_old}，现在喜欢{food_new}。",
                    "不要在周五安排会议。",
                ],
                [
                    {"category": "name", "query": "我叫什么？", "golds": [name], "status": "current"},
                    {"category": "job", "query": "我现在的职业是什么？", "golds": [job_new], "status": "current", "forbid_current": [f"occupation = {job_old} (status=current"]},
                    {"category": "preference", "query": "我现在喜欢吃什么？", "golds": [food_new], "status": "current", "forbid_current": [f"preference = {food_old} (status=current"]},
                    {"category": "rule", "query": "周五开会有什么限制？", "golds": ["周五"]},
                ],
            )
        )
    for index in range(10):
        name, city, day = NAMES[index], ZH[index + 5], DAYS[index % 7]
        cases.append(
            _case(
                f"hop-{index}",
                [f"我{day}和{name}见面。", f"{name}住在{city}。", f"我喜欢{FOODS[index]}。"],
                [
                    {"category": "hop", "query": f"我{day}见面的人住在哪里？", "golds": [city, name]},
                ],
            )
        )
    for index in range(10):
        home, noise = EN[index + 10], EN[index]
        messages = [f"The {noise} office called about the {noise} budget."] * 8
        messages.append(f"I now live in {home}.")
        cases.append(
            _case(
                f"flood-{index}",
                messages,
                [
                    {"category": "flood", "query": "Which city is home now?", "golds": [home], "status": "current", "forbid_current": [f"residence = {noise} (status=current"]},
                ],
            )
        )
    for index in range(10):
        secret = f"编号{7000 + index}"
        cases.append(
            _case(
                f"alias-{index}",
                [f"请记住我的工号是{secret}。", "今天下雨。"],
                [
                    {"category": "alias", "query": "我的员工编号是多少？", "golds": [secret]},
                ],
            )
        )
    for index in range(10):
        old, new = ZH[index + 1], ZH[index + 21]
        cases.append(
            _case(
                f"cengjing-{index}",
                [f"我曾经住在{old}，后来搬到了{new}。", f"同事住在{ZH[index]}。"],
                [
                    {"category": "cengjing-current", "query": "我现在住在哪里？", "golds": [new], "status": "current", "forbid_current": [f"我曾经 residence = {old}"]},
                    {"category": "cengjing-history", "query": "我曾经住在哪里？", "golds": [old], "status": "previous"},
                ],
            )
        )
    for index in range(10):
        liked, hated = FOODS[index], FOODS[(index + 5) % 10]
        when = f"{8 + index}:30"
        cases.append(
            _case(
                f"neg-{index}",
                [f"我喜欢{liked}。", f"我不喜欢{hated}。", f"和Mina的会面在周三{when}。"],
                [
                    {"category": "negation", "query": "我不喜欢什么？", "golds": [hated]},
                    {"category": "time", "query": "和Mina会面是几点？", "golds": [when]},
                ],
            )
        )
    for index in range(10):
        city = ZH[30 + (index % 10)]
        cases.append(
            _case(
                f"iso-owner-{index}",
                [f"我现在住在{city}。"],
                [],
            )
        )
        cases.append(
            _case(
                f"iso-other-{index}",
                ["我今天买了面包。"],
                [
                    {"category": "isolation", "query": "我现在住在哪里？", "golds": [city], "expect_absent": True},
                ],
            )
        )
    for index in range(8):
        cases.append(
            _case(
                f"code-{index}",
                [
                    "def rebuild_index(path):\n    if not path:\n        return None\n    items = load_rows(path)\n    return sorted(items)\n",
                ],
                [
                    {"category": "code", "query": "Where is rebuild_index defined?", "golds": ["rebuild_index"]},
                ],
            )
        )
    for index in range(25):
        old, new = ZH[index % 40], ZH[(index + 17) % 40]
        cases.append(
            _case(
                f"move-qu-{index}",
                [f"我以前住在{old}。", f"后来我搬去了{new}。", f"同事住在{ZH[(index + 3) % 40]}。"],
                [
                    {"category": "move-qu-current", "query": "我现在住在哪里？", "golds": [new], "status": "current", "forbid_current": [f"residence = {old} (status=current"]},
                    {"category": "move-qu-history", "query": "我以前住在哪里？", "golds": [old], "status": "previous"},
                ],
            )
        )
    for index in range(25):
        old_name, new_name = f"旧名{index}", f"新名{index}"
        cases.append(
            _case(
                f"rename-{index}",
                [f"我以前的名字是{old_name}。", f"我改名为{new_name}。"],
                [
                    {"category": "rename-current", "query": "我现在叫什么？", "golds": [new_name], "status": "current"},
                    {"category": "rename-history", "query": "我以前叫什么？", "golds": [old_name], "status": "previous"},
                ],
            )
        )
    for index in range(25):
        number = f"1380013{index:04d}"
        cases.append(
            _case(
                f"phone-{index}",
                [f"请记住我的手机号是{number}。", "今天下雨。"],
                [{"category": "phone", "query": "我的手机号是多少？", "golds": [number]}],
            )
        )
    for index in range(25):
        mail = f"user{index}@example.com"
        cases.append(
            _case(
                f"email-{index}",
                [f"我的邮箱是{mail}。"],
                [{"category": "email", "query": "我的邮箱是什么？", "golds": [mail]}],
            )
        )
    people = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Helen", "Ivan", "Judy"]
    for index in range(25):
        person, city, day = people[index % 10], EN[index % 20], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][index % 5]
        cases.append(
            _case(
                f"en-hop-{index}",
                [f"I met {person} on {day}.", f"{person} lives in {city}.", "I like tea."],
                [{"category": "en-hop", "query": f"Where does the person I met on {day} live?", "golds": [city, person]}],
            )
        )
    for index in range(25):
        home = ZH[(index + 11) % 40]
        noise = ZH[index % 40]
        messages = [f"{noise}的天气预报说有雨。"] * 12
        messages.append(f"我现在住在{home}。")
        cases.append(
            _case(
                f"zh-flood-{index}",
                messages,
                [{"category": "zh-flood", "query": "我现在住在哪里？", "golds": [home], "status": "current"}],
            )
        )
    for index in range(20):
        liked, also = FOODS[index % 10], FOODS[(index + 3) % 10]
        cases.append(
            _case(
                f"two-likes-{index}",
                [f"我喜欢{liked}。", f"我也喜欢{also}。"],
                [{"category": "two-likes", "query": f"我喜欢{also}吗？", "golds": [also]}],
            )
        )
    for index in range(15):
        fn = ["parse_rows", "build_graph", "drop_cache", "merge_logs", "scan_ports"][index % 5]
        cases.append(
            _case(
                f"code-fn-{index}",
                [f"def {fn}(path):\n    return path\n"],
                [{"category": "code-fn", "query": f"Where is {fn} defined?", "golds": [fn]}],
            )
        )
    for index in range(20):
        sentence = f"原话编号{index}：请不要改期。"
        cases.append(
            _case(
                f"quote-{index}",
                [sentence, "另外今天下雨。"],
                [{"category": "quote", "query": f"原话编号{index}是什么？", "golds": [f"原话编号{index}"]}],
            )
        )
    for index in range(20):
        old_job, new_job = JOBS[index % 10], JOBS[(index + 4) % 10]
        cases.append(
            _case(
                f"en-job-{index}",
                [f"I used to work as a {old_job}.", f"I work as a {new_job} now."],
                [
                    {"category": "en-job-current", "query": "What is my job now?", "golds": [new_job], "status": "current", "forbid_current": [f"occupation = {old_job} (status=current"]},
                    {"category": "en-job-history", "query": "What job did I have before?", "golds": [old_job], "status": "previous"},
                ],
            )
        )
    return cases


def _hit(content: str, gold: str, status: str | None) -> bool:
    if status:
        return f"= {gold} (status={status}" in content
    return gold in content


def _rank(rows: list[dict], golds: list[str], status: str | None) -> int | None:
    for index, row in enumerate(rows):
        content = str(row["content"])
        if status:
            if _hit(content, golds[0], status) and all(gold in content for gold in golds[1:]):
                return index
        elif all(gold in content for gold in golds):
            return index
    if status:
        for index, row in enumerate(rows):
            content = str(row["content"])
            if all(gold in content for gold in golds):
                return index
    return None


def run(label: str, embedder_name: str | None = None) -> dict:
    if embedder_name:
        os.environ["EMBEDDER"] = embedder_name
    os.environ["EVAL_STRICT_EMBED"] = "1"
    cases = build_cases()
    probe_count = sum(len(case["probes"]) for case in cases)
    if probe_count < 500:
        raise SystemExit(f"need at least 500 probes, got {probe_count}")
    settings = get_settings()
    run_dir = ROOT / "eval" / "runs" / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    settings.data_dir = run_dir
    embedder = Embedder(settings)
    if embedder.kind not in {"dashscope", "hash"}:
        raise SystemExit(f"unsupported embedder {embedder.kind}")
    embedder.embed("预热")
    store = MemoryStore(settings, embedder, settings.weights())
    details = []
    for case in cases:
        if not case["messages"]:
            continue
        messages = [
            {"role": "user", "timestamp": 1_700_000_000_000 + index * 3_600_000, "content": text}
            for index, text in enumerate(case["messages"], start=1)
        ]
        store.add(request_id=f"add-{case['user']}", user_id=case["user"], session_id="large", messages=messages)
        for item in case["probes"]:
            rows = store.search(case["user"], item["query"], 100)
            if item.get("expect_absent"):
                leaked = any(item["golds"][0] in str(row["content"]) for row in rows)
                details.append({**_blank(case, item), "rank": None, "recall_at_5": 1.0 if not leaked else 0.0, "recall_at_100": 1.0 if not leaked else 0.0, "reciprocal_rank": 1.0 if not leaked else 0.0, "status_ok": not leaked, "top": [] if not leaked else [str(row["content"])[:160] for row in rows[:3]]})
                continue
            rank = _rank(rows, item["golds"], item.get("status"))
            status_ok = True
            if item.get("status"):
                status_ok = rank is not None and _hit(str(rows[rank]["content"]), item["golds"][0], item["status"])
            for banned in item.get("forbid_current", []):
                if any(banned in str(row["content"]) and "status=current" in str(row["content"]) for row in rows[:5]):
                    status_ok = False
            details.append(
                {
                    **_blank(case, item),
                    "rank": None if rank is None else rank + 1,
                    "recall_at_5": 1.0 if rank is not None and rank < 5 else 0.0,
                    "recall_at_100": 1.0 if rank is not None and rank < 100 else 0.0,
                    "reciprocal_rank": 0.0 if rank is None else round(1.0 / (rank + 1), 4),
                    "status_ok": status_ok,
                    "top": [] if rank == 0 and status_ok else [str(row["content"])[:160] for row in rows[:3]],
                }
            )
    store.close()
    report = _aggregate(label, settings, embedder.kind, details)
    out = ROOT / "eval" / "reports" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{label} n={report['probe_count']} recall@5={report['recall_at_5']:.3f} "
        f"recall@100={report['recall_at_100']:.3f} mrr={report['mrr']:.3f} status={report['status_accuracy']:.3f}"
    )
    for name, scores in report["by_category"].items():
        if scores["recall_at_5"] < 1 or scores["mrr"] < 1 or scores["status_accuracy"] < 1:
            print(
                f"  {name}: n={scores['n']} recall@5={scores['recall_at_5']:.3f} "
                f"mrr={scores['mrr']:.3f} status={scores['status_accuracy']:.3f}"
            )
    return report


def _blank(case: dict, item: dict) -> dict:
    return {"user": case["user"], "category": item["category"], "query": item["query"], "golds": item.get("golds", [])}


def _aggregate(label: str, settings, kind: str, details: list[dict]) -> dict:
    def mean(key: str, rows: list[dict]) -> float:
        return round(sum(row[key] for row in rows) / len(rows), 4)

    by_category = {}
    for row in details:
        by_category.setdefault(row["category"], []).append(row)
    return {
        "label": label,
        "embedder": kind,
        "scheme_pref": "1" if settings.scheme_pref else "0",
        "scheme_hop": "1" if settings.scheme_hop else "0",
        "hop_residence_bonus": os.environ.get("HOP_RESIDENCE_BONUS", "0.01"),
        "embedding_model": settings.embedding_model,
        "llm_extract_active": bool(settings.llm_extract and settings.openai_api_key),
        "weights": settings.weights(),
        "probe_count": len(details),
        "recall_at_5": mean("recall_at_5", details),
        "recall_at_100": mean("recall_at_100", details),
        "mrr": mean("reciprocal_rank", details),
        "status_accuracy": round(sum(1.0 if row["status_ok"] else 0.0 for row in details) / len(details), 4),
        "by_category": {
            name: {
                "n": len(rows),
                "recall_at_5": mean("recall_at_5", rows),
                "recall_at_100": mean("recall_at_100", rows),
                "mrr": mean("reciprocal_rank", rows),
                "status_accuracy": round(sum(1.0 if row["status_ok"] else 0.0 for row in rows) / len(rows), 4),
            }
            for name, rows in sorted(by_category.items())
        },
        "failures": [row for row in details if row["recall_at_5"] < 1 or not row["status_ok"]],
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    label = args[0] if args else "large-baseline"
    embedder_name = None
    if "--embedder" in args:
        embedder_name = args[args.index("--embedder") + 1]
    run(label, embedder_name)
