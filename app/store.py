from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import DEFAULT_WEIGHTS, Settings
from app.embedder import Embedder, bm25, cosine, pack_vector, unpack_vector
from app.reranker import CrossReranker
from app.extract import Fact, extract_facts
from app.textutil import (
    classify_intent,
    flatten_content,
    jaccard,
    numbers_in,
    overlap_ratio,
    query_terms,
    tokenize,
    user_token,
)

IMAGE_LIMIT = 10 * 1024 * 1024
RESPONSE_IMAGE_BUDGET = 24 * 1024 * 1024


class ContractError(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class _Hit:
    public_id: str
    content: str
    content_json: str | None
    kind: str
    active: int
    created_ms: int
    entities: list[str]
    tokens: list[str]
    tf: dict[str, int]
    doc_len: int
    embedding: object
    hop: int
    score: float = 0.0


class MemoryStore:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, weights: dict[str, float] | None = None):
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        self.reranker = CrossReranker(settings)
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.path = Path(settings.data_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path / "memory.db", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
              id INTEGER PRIMARY KEY,
              public_id TEXT UNIQUE,
              user_id TEXT NOT NULL,
              session_id TEXT NOT NULL,
              request_id TEXT NOT NULL,
              role TEXT,
              kind TEXT NOT NULL,
              content TEXT NOT NULL,
              content_json TEXT,
              indexed_text TEXT NOT NULL,
              attr_key TEXT,
              fact_value TEXT,
              active INTEGER NOT NULL DEFAULT 1,
              superseded_by TEXT,
              created_ms INTEGER NOT NULL,
              entities TEXT,
              embedding BLOB,
              doc_len INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_mem_attr ON memories(user_id, attr_key, active);
            CREATE TABLE IF NOT EXISTS adds (
              request_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              session_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS term_df (
              user_id TEXT NOT NULL,
              term TEXT NOT NULL,
              df INTEGER NOT NULL,
              PRIMARY KEY (user_id, term)
            );
            CREATE TABLE IF NOT EXISTS user_stats (
              user_id TEXT PRIMARY KEY,
              n_docs INTEGER NOT NULL,
              sum_len INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
              indexed_text,
              public_id UNINDEXED
            );
            """
        )
        self._conn.commit()

    def add(self, request_id: str, user_id: str, session_id: str, messages: list[dict], extra_facts: list[Fact] | None = None) -> None:
        if not request_id or not user_id or not session_id:
            raise ContractError(422, "request_id, user_id and session_id are required")
        if not messages:
            raise ContractError(422, "messages is required")
        prepared = [self._prepare_message(message) for message in messages]
        texts = [item["text"] for item in prepared]
        vectors = self.embedder.embed_many(texts)
        fact_groups: list[list[Fact]] = []
        fact_texts: list[str] = []
        for item in prepared:
            facts = extract_facts(item["text"])
            fact_groups.append(facts)
            for fact in facts:
                fact_texts.append(fact.render("current", item["when"]))
        if extra_facts:
            fact_groups.append(extra_facts)
            for fact in extra_facts:
                fact_texts.append(fact.render("current", prepared[-1]["when"]))
        fact_vectors = self.embedder.embed_many(fact_texts) if fact_texts else []
        with self._lock:
            existing = self._conn.execute("SELECT user_id, session_id FROM adds WHERE request_id = ?", (request_id,)).fetchone()
            if existing:
                if existing["user_id"] != user_id or existing["session_id"] != session_id:
                    raise ContractError(409, "request_id is already bound to another payload")
                return
            self._conn.execute("BEGIN")
            try:
                vector_index = 0
                for item, vector, facts in zip(prepared, vectors, fact_groups):
                    self._insert(
                        user_id=user_id,
                        session_id=session_id,
                        request_id=request_id,
                        role=item["role"],
                        kind="turn",
                        content=item["stored"],
                        content_json=item["content_json"],
                        plain_text=item["text"],
                        created_ms=item["created_ms"],
                        embedding=vector,
                        attr_key=None,
                        fact_value=None,
                        active=1,
                    )
                    for fact in facts:
                        self._apply_fact(
                            fact,
                            user_id=user_id,
                            session_id=session_id,
                            request_id=request_id,
                            role=item["role"],
                            created_ms=item["created_ms"],
                            when=item["when"],
                            embedding=fact_vectors[vector_index],
                        )
                        vector_index += 1
                if extra_facts:
                    when = prepared[-1]["when"]
                    created_ms = prepared[-1]["created_ms"]
                    for fact in extra_facts:
                        self._apply_fact(
                            fact,
                            user_id=user_id,
                            session_id=session_id,
                            request_id=request_id,
                            role="user",
                            created_ms=created_ms,
                            when=when,
                            embedding=fact_vectors[vector_index],
                        )
                        vector_index += 1
                self._upsert_profile(user_id, session_id, prepared[-1]["created_ms"])
                self._conn.execute(
                    "INSERT INTO adds(request_id, user_id, session_id) VALUES (?, ?, ?)",
                    (request_id, user_id, session_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def search(self, user_id: str, query, top_k: int, options: list[str] | None = None) -> list[dict]:
        if not user_id:
            raise ContractError(422, "user_id is required")
        if not isinstance(top_k, int) or top_k < 1:
            raise ContractError(422, "top_k must be a positive integer")
        query_text = flatten_content(query).strip()
        if not query_text:
            raise ContractError(422, "query is required")
        options = [str(item) for item in (options or []) if str(item).strip()]
        q_terms = query_terms(query_text + " " + " ".join(options))
        q_tokens = tokenize(query_text)
        intent = classify_intent(query_text)
        q_numbers = numbers_in(query_text)
        q_entities = [tok for tok in q_tokens if len(tok) > 1]
        with self._lock:
            hits = self._candidate_hits(user_id, q_terms)
            if not hits:
                return []
            hops = self._second_hop(user_id, hits[:20], {item.public_id for item in hits})
            hits.extend(hops)
            stats = self._conn.execute("SELECT n_docs, sum_len FROM user_stats WHERE user_id = ?", (user_id,)).fetchone()
            n_docs = int(stats["n_docs"]) if stats else len(hits)
            avg_len = (int(stats["sum_len"]) / n_docs) if stats and n_docs else 1.0
            dfs = self._dfs(user_id, set(q_terms))
            created = [item.created_ms for item in hits]
            min_ms, max_ms = min(created), max(created)
            query_vector = self.embedder.embed(query_text)
            option_sets = [set(tokenize(option)) for option in options]
            for hit in hits:
                raw = bm25(q_terms, hit.tf, hit.doc_len, dfs, n_docs, avg_len)
                hit.raw_bm25 = raw
            max_bm25 = max(item.raw_bm25 for item in hits) or 1.0
            for hit in hits:
                features = self._features(
                    hit,
                    max_bm25=max_bm25,
                    query_vector=query_vector,
                    q_terms=q_terms,
                    q_entities=q_entities,
                    q_numbers=q_numbers,
                    intent=intent,
                    min_ms=min_ms,
                    max_ms=max_ms,
                    option_sets=option_sets,
                )
                hit.score = sum(self.weights.get(name, 0.0) * value for name, value in features.items())
            self._promote_met_person_home(query_text, hits)
            self._promote_own_current_home(query_text, hits)
            self._promote_identity(query_text, hits)
            self._promote_attribute_alias(query_text, hits)
            self._promote_named_relation(query_text, hits)
            self._promote_social_object(query_text, hits)
            self._govern(query_text, intent, hits)
            self._promote_graph(query_text, hits)
            self._promote_timeline(query_text, hits)
            self._promote_procedure(query_text, hits)
            self._promote_care(query_text, hits)
            self._promote_explicit(query_text, intent, hits)
            self._shield_private(query_text, intent, hits)
            self._demote_hearsay(query_text, intent, hits)
            if self.settings.chronos:
                self._chronos(query_text, hits)
            hits.sort(key=lambda item: item.score, reverse=True)
            pool = hits[: max(top_k * 4, 40)]
            ranked = self._mmr(pool, top_k)
            contexts = [
                hit.content
                for hit in sorted(hits, key=lambda item: item.raw_bm25, reverse=True)[:5]
            ]
            ranked = self.reranker.rerank(query_text, ranked, contexts if self.reranker.use_context else None)
            budget = RESPONSE_IMAGE_BUDGET
            results = []
            total = len(ranked)
            for index, hit in enumerate(ranked):
                rendered, budget = self._render(hit, budget)
                results.append(
                    {
                        "id": hit.public_id,
                        "content": rendered,
                        "score": round(1.0 - index / (total + 1), 6),
                        "created_at": _iso(hit.created_ms),
                    }
                )
            return results

    def _cards(self, hits: list[_Hit]) -> list[tuple[str, str, str, _Hit]]:
        cards = []
        for hit in hits:
            match = re.match(r"(.+?) ([A-Za-z_]+) = (.+?) \(status=", hit.content)
            if match:
                cards.append((match.group(1).strip(), match.group(2), match.group(3).strip(), hit))
        return cards

    def _govern(self, query_text: str, intent: str, hits: list[_Hit]) -> None:
        if not hits:
            return
        history = intent == "history"
        asking_now = (not history) and (
            intent == "state" or re.search(r"现在|目前|当前|\bnow\b|\bcurrent\b", query_text, re.I) is not None
        )
        cancelled: list[str] = []
        previous_vals: list[str] = []
        current_vals: list[str] = []
        for hit in hits:
            match = re.search(r"= (.+?) \(status=(current|previous|retracted)", hit.content)
            if not match:
                continue
            value, status = match.group(1), match.group(2)
            if status == "retracted":
                cancelled.append(value)
            elif status == "previous":
                previous_vals.append(value)
            else:
                current_vals.append(value)
        if history:
            for hit in hits:
                if "status=previous" in hit.content or "status=retracted" in hit.content:
                    hit.score += 0.55
            return
        asks_dislike = re.search(r"不喜欢|讨厌|don't like|do not like", query_text, re.I) is not None
        if not asking_now and not cancelled and not asks_dislike:
            return
        for hit in hits:
            if asking_now and not asks_dislike and ("status=previous" in hit.content or "status=retracted" in hit.content):
                hit.score -= 1.4
            if hit.kind != "turn":
                continue
            text = hit.content
            if any(value and value in text for value in cancelled) and re.search(r"忘掉|忘记|取消|记错", query_text) is None:
                hit.score -= 2.2
            elif asking_now and any(value and value in text for value in previous_vals):
                if not any(value and value in text for value in current_vals):
                    hit.score -= 0.9
        if asks_dislike and cancelled:
            disliked = [
                hit
                for hit in hits
                if " preference = " in hit.content and any(value in hit.content for value in cancelled)
            ]
            self._lift(hits, disliked)

    def _promote_graph(self, query_text: str, hits: list[_Hit]) -> None:
        if re.search(r"住|哪里|live|where", query_text, re.I):
            target = "residence"
        elif re.search(r"职业|工作|做什么", query_text):
            target = "occupation"
        elif re.search(r"手机|电话", query_text):
            target = "phone"
        elif "邮箱" in query_text:
            target = "email"
        else:
            return
        if not hits:
            return
        words = (
            ("见面", "met"),
            ("导师", "mentor"),
            ("同事", "colleague"),
            ("朋友", "friend"),
            ("同学", "classmate"),
            ("经理", "manager"),
            ("学生", "student"),
            ("mentor", "mentor"),
            ("colleague", "colleague"),
            ("friend", "friend"),
        )
        found: list[tuple[int, str]] = []
        for word, attr in words:
            start = 0
            while True:
                index = query_text.lower().find(word.lower(), start)
                if index < 0:
                    break
                found.append((index, attr))
                start = index + len(word)
        if re.search(r"\bmet\b", query_text, re.I):
            found.append((query_text.lower().find("met"), "met"))
        if not found:
            return
        found.sort()
        rels = [attr for _, attr in found]
        if re.search(r"\bof\b", query_text, re.I):
            rels.reverse()
        cards = self._cards(hits)
        nodes = {"user"}
        for attr in rels:
            nxt: set[str] = set()
            for subject, attribute, value, _hit in cards:
                if attribute != attr or subject.lower() not in {node.lower() for node in nodes}:
                    continue
                person = value.split(":")[-1] if attr == "met" and ":" in value else value
                nxt.add(person)
            if nxt:
                nodes = nxt
        lowered = {node.lower() for node in nodes}
        homes = [hit for subject, attribute, _value, hit in cards if attribute == target and subject.lower() in lowered]
        self._lift(hits, homes)

    def _chronos(self, query_text: str, hits: list[_Hit]) -> None:
        """Prefer memories whose text carries the time the question asks for.

        Original memories stay in the list. Only their order changes.
        """
        asks_when = re.search(r"什么时候|何时|哪天|哪年|几点|时间|上次|第一次|最后一次", query_text)
        asks_past = re.search(r"以前|之前|曾经|过去|上次", query_text)
        asks_now = re.search(r"现在|目前|最近", query_text)
        asks_after = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,12})之[后後]", query_text)
        if not any((asks_when, asks_past, asks_now, asks_after)):
            return
        time_in_text = re.compile(r"\d{4}年|\d{1,2}月|\d{1,2}日|\d{1,2}点|上午|下午|晚上|昨天|上周|上周|凌晨")
        for hit in hits:
            text = hit.content or ""
            if asks_when and time_in_text.search(text):
                hit.score += 0.55
            if asks_past and re.search(r"以前|曾经|之前|过去|status=previous", text):
                hit.score += 0.35
            if asks_now and not asks_past and re.search(r"现在|目前|最近|status=current", text):
                hit.score += 0.35
            if asks_after and asks_after.group(1) in text:
                hit.score += 0.25

    def _promote_timeline(self, query_text: str, hits: list[_Hit]) -> None:
        cards = self._cards(hits)
        events: list[tuple[int, _Hit, str]] = []
        meetings: list[tuple[int, _Hit, str]] = []
        for _subject, attribute, value, hit in cards:
            if not value[:1].isdigit() or ":" not in value:
                continue
            head = value.split(":", 1)[0]
            if not head.isdigit():
                continue
            order = int(head)
            if attribute == "event":
                events.append((order, hit, value))
            elif attribute == "met":
                meetings.append((order, hit, value))
        meeting_query = "会面" in query_text or "见面" in query_text
        if re.search(r"第一次|最早|最先|第一站", query_text):
            pool = meetings if meeting_query and meetings else events
            if pool:
                pool.sort()
                self._lift(hits, [pool[0][1]])
            return
        if re.search(r"最后|最晚|最终", query_text):
            pool = meetings if meeting_query and meetings else events
            if pool:
                pool.sort()
                self._lift(hits, [pool[-1][1]])
            return
        after = re.search(r"([\u4e00-\u9fff0-9A-Za-z]{2,16})之后", query_text)
        if after and events:
            name = after.group(1)
            current = [item for item in events if name in item[2] or name in item[1].content]
            if not current:
                return
            later = sorted(item for item in events if item[0] > current[0][0])
            if later:
                self._lift(hits, [later[0][1]])

    def _promote_procedure(self, query_text: str, hits: list[_Hit]) -> None:
        if not hits or re.search(r"帮我|按我的|安排|回复|写一封|应该怎么", query_text) is None:
            return
        rules = [hit for hit in hits if hit.content.startswith("user rule = ") and "status=current" in hit.content]
        if not rules:
            return
        best = max(hit.score for hit in hits)
        ordered = sorted(rules, key=lambda hit: hit.score, reverse=True)[:3]
        step = self.settings.hop_residence_bonus
        for index, hit in enumerate(ordered):
            hit.score = max(hit.score, best + step * (len(ordered) - index) + 0.2)

    def _promote_care(self, query_text: str, hits: list[_Hit]) -> None:
        if not hits or re.search(r"忌口|过敏|照顾我|不能吃", query_text) is None:
            return
        cards = [
            hit
            for subject, attribute, _value, hit in self._cards(hits)
            if subject == "user" and attribute == "allergy" and "status=current" in hit.content
        ]
        self._lift(hits, cards)

    def _promote_explicit(self, query_text: str, intent: str, hits: list[_Hit]) -> None:
        if not hits or intent == "history" or re.search(r"导师|同事|朋友|同学|经理|学生|见面", query_text):
            return
        wanted = (
            ("工号", "employee_id"),
            ("部门", "department"),
            ("手机号", "phone"),
            ("邮箱", "email"),
            ("城市", "city"),
            ("称呼", "nickname"),
            ("职业", "occupation"),
            ("名字", "name"),
        )
        attribute = next((name for word, name in wanted if word in query_text), "")
        if not attribute:
            return
        cards = [
            hit
            for subject, attr, _value, hit in self._cards(hits)
            if subject == "user" and attr == attribute and "status=current" in hit.content
        ]
        self._lift(hits, cards)

    def _shield_private(self, query_text: str, intent: str, hits: list[_Hit]) -> None:
        if not hits or intent == "history" or re.search(r"忘掉了什么|忘记了什么|哪些隐私", query_text):
            return
        private = {"phone", "email", "id_card", "address", "medical"}
        hidden: list[str] = []
        for subject, attribute, value, hit in self._cards(hits):
            if subject == "user" and attribute in private and "status=retracted" in hit.content and value != "*":
                hidden.append(value)
        if not hidden:
            return
        hidden_set = set(hidden)
        for hit in hits:
            match = re.search(r"= (.+?) \(status=(current|previous|retracted)", hit.content)
            if match and match.group(1) in hidden_set and match.group(2) != "current":
                hit.score -= 20
            elif hit.kind in {"turn", "profile"} and any(value in hit.content for value in hidden_set):
                hit.score -= 20

    def _demote_hearsay(self, query_text: str, intent: str, hits: list[_Hit]) -> None:
        if not hits or intent == "history" or re.search(r"我(现在)?住", query_text) is None:
            return
        if re.search(r"导师|同事|朋友|同学|经理|学生|见面", query_text):
            return
        for hit in hits:
            if re.search(r"听说|据说|没有核实|未核实", hit.content):
                hit.score -= 20

    def _relation_names(self, query_text: str) -> list[str]:
        blocked = {
            "他们", "她们", "我们", "你们", "之间", "关系", "什么", "现在", "这个", "那个",
            "一个", "是否", "怎么", "描述", "介绍", "一下", "请", "有", "何种", "特点",
            "朋友", "熟人", "同事", "邻居", "同学", "导师", "老师", "配角",
        }
        text = re.sub(r"^(请)?(描述|介绍)一下", "", query_text)
        text = re.sub(r"[？?，,。！!]", "", text)
        names: list[str] = []
        for piece in re.split(r"有什么关系|有何关系|是什么关系|何种关系|什么关系|之间|和|与|跟|的|是", text):
            piece = piece.strip()
            if not piece or piece in blocked or piece in names or not 2 <= len(piece) <= 4:
                continue
            if any(word in piece for word in ("关系", "什么", "朋友", "描述", "介绍")):
                continue
            if not re.fullmatch(r"[\u4e00-\u9fff]+", piece):
                continue
            names.append(piece)
        return names[:3]

    def _promote_named_relation(self, query_text: str, hits: list[_Hit]) -> None:
        if not hits or "关系" not in query_text:
            return
        if re.search(r"关系(?:是)?(?:如何|怎样)", query_text):
            return
        names = self._relation_names(query_text)
        if len(names) >= 2:
            wanted = names
            shorts = [
                hit
                for hit in hits
                if len(hit.content) <= 180 and all(name in hit.content for name in wanted)
            ]
        elif len(names) == 1 and re.search(r"谁|我", query_text):
            name = names[0]
            shorts = [
                hit
                for hit in hits
                if len(hit.content) <= 180
                and name in hit.content
                and (f"是{name}" in hit.content or "的朋友是" in hit.content or "的同事是" in hit.content)
            ]
        else:
            return
        if not shorts:
            return
        best = max(hit.score for hit in hits)
        ordered = sorted(shorts, key=lambda hit: hit.score, reverse=True)[:2]
        step = self.settings.hop_residence_bonus
        for index, hit in enumerate(ordered):
            hit.score = max(hit.score, best + step * (len(ordered) - index))

    def _promote_social_object(self, query_text: str, hits: list[_Hit]) -> None:
        if not hits:
            return
        name = ""
        attribute = list(re.finditer(r"([\u4e00-\u9fff]{2,4})的(?:职业|爱好|国籍|外貌)", query_text))
        if attribute:
            name = attribute[-1].group(1)
        elif (liked := re.match(r"^([\u4e00-\u9fff]{2,3})(?:喜欢|对什么)", query_text)):
            name = liked.group(1)
        elif (intro := re.search(r"(?:介绍|描述)一下([\u4e00-\u9fff]{2,3})(?:的|[。？?，,]|$)", query_text)):
            name = intro.group(1)
        elif (who := re.match(r"^([\u4e00-\u9fff]{2,4})(?:是谁|是什么人|是什么关系|是做什么|是干什么|是什么角色|是什么行业)", query_text)):
            name = who.group(1)
        elif any(word in query_text for word in ("浪漫伴侣", "男朋友", "女朋友")):
            found = re.search(r"([\u4e00-\u9fff]{2,4})的(?:浪漫伴侣|男朋友|女朋友)", query_text)
            if found:
                owner = found.group(1)[1:] if found.group(1).startswith("是") else found.group(1)
                shorts = [
                    hit
                    for hit in hits
                    if len(hit.content) <= 200
                    and owner in hit.content
                    and any(token in hit.content for token in ("男朋友", "女朋友", "男友", "女友"))
                ]
                self._lift(hits, shorts)
            return
        if not name:
            return
        if any(f"{name}的职业是" in hit.content or f"{name}的头衔是" in hit.content for hit in hits):
            return
        shorts = [
            hit
            for hit in hits
            if len(hit.content) <= 220 and (f"是{name}。" in hit.content or f"是{name}，" in hit.content)
        ]
        self._lift(hits, shorts)

    def _lift(self, hits: list[_Hit], shorts: list[_Hit]) -> None:
        if not hits or not shorts:
            return
        best = max(hit.score for hit in hits)
        ordered = sorted(shorts, key=lambda hit: hit.score, reverse=True)[:2]
        step = self.settings.hop_residence_bonus
        for index, hit in enumerate(ordered):
            hit.score = max(hit.score, best + step * (len(ordered) - index))

    def _promote_attribute_alias(self, query_text: str, hits: list[_Hit]) -> None:
        if not hits:
            return
        groups = (
            ("雇主", ("工作单位",)),
            ("种族", ("民族",)),
            ("人种", ("民族",)),
            ("奖项", ("获奖", "奖项")),
            ("榜样", ("榜样", "Role Models")),
            ("哪个国家", ("国籍",)),
            ("学什么", ("教育背景",)),
            ("学过", ("教育背景",)),
        )
        needles: list[str] = []
        for cue, forms in groups:
            if cue in query_text:
                needles.extend(forms)
        if re.search(r"^[\u4e00-\u9fff]{2,4}(?:的奖项是|获得了哪[些个]奖项)", query_text):
            needles.append("的Awards是")
        appearance = re.match(r"^([\u4e00-\u9fff]{2,4})的外貌", query_text)
        if appearance:
            needles.append(f"{appearance.group(1)}的外貌是")
        achievement = re.match(r"^([\u4e00-\u9fff]{2,4})(?:取得了|取得|获得了|获得|有哪些|有什么|的)", query_text)
        if achievement and "成就" in query_text:
            needles.append(f"{achievement.group(1)}的成就是")
        if not needles:
            return
        shorts = [hit for hit in hits if len(hit.content) <= 180 and any(needle in hit.content for needle in needles)]
        if not shorts:
            return
        best = max(hit.score for hit in hits)
        ordered = sorted(shorts, key=lambda hit: hit.score, reverse=True)[:2]
        step = self.settings.hop_residence_bonus
        for index, hit in enumerate(ordered):
            hit.score = max(hit.score, best + step * (len(ordered) - index))

    def _promote_identity(self, query_text: str, hits: list[_Hit]) -> None:
        if not hits or re.search(r"的[\u4e00-\u9fff]{1,8}是谁", query_text):
            return
        identity = re.search(r"是谁[？?]?$|是什么人|什么身份|是做什么|做什么的", query_text)
        bare_match = re.search(r"^[\u4e00-\u9fff]{1,8}是什么[？?]?$", query_text)
        bare = bare_match is not None and "的" not in bare_match.group(0)
        subject = ""
        if identity:
            found = re.search(r"([\u4e00-\u9fff]{2,8})(?:是谁|是什么人|是什么身份|是做什么|做什么的)", query_text)
            subject = found.group(1) if found else ""
        elif bare and bare_match:
            subject = bare_match.group(0).split("是什么")[0]
        if not identity and not bare:
            return
        shorts = [
            hit
            for hit in hits
            if len(hit.content) <= 240
            and ("的职业是" in hit.content or "的头衔是" in hit.content)
            and (not subject or subject in hit.content)
        ]
        if not shorts:
            return
        best = max(hit.score for hit in hits)
        ordered = sorted(shorts, key=lambda hit: hit.score, reverse=True)[:2]
        step = self.settings.hop_residence_bonus
        for index, hit in enumerate(ordered):
            hit.score = max(hit.score, best + step * (len(ordered) - index))

    def _promote_met_person_home(self, query_text: str, hits: list[_Hit]) -> None:
        if not self.settings.scheme_hop or not hits:
            return
        zh = "见面" in query_text and re.search(r"住在|哪里", query_text) is not None
        en = re.search(r"\b(?:met|meet|meeting)\b", query_text, re.I) is not None and re.search(
            r"\b(?:live|lives|where)\b", query_text, re.I
        ) is not None
        if not zh and not en:
            return
        names: list[str] = []
        for hit in hits:
            patterns = []
            if zh:
                patterns.append(r"和([\u4e00-\u9fffA-Za-z]{1,20})见面")
            if en:
                patterns.append(r"\bmet ([A-Z][a-z]+)")
            for pattern in patterns:
                for match in re.finditer(pattern, hit.content, re.I if en else 0):
                    name = match.group(1)
                    if name not in names:
                        names.append(name)
        if not names:
            return
        best = max(hit.score for hit in hits)
        margin = self.settings.hop_residence_bonus
        for hit in hits:
            text = hit.content
            if not any(name in text for name in names):
                continue
            has_home = "residence" in text or "住在" in text or re.search(r"\blives in\b", text, re.I) is not None
            if not has_home:
                continue
            meeting_only = ("见面" in text or re.search(r"\bmet\b", text, re.I) is not None) and "residence" not in text
            if meeting_only and "住在" not in text and re.search(r"\blives in\b", text, re.I) is None:
                continue
            hit.score = max(hit.score, best + margin)

    def _promote_own_current_home(self, query_text: str, hits: list[_Hit]) -> None:
        if not hits or re.search(r"以前|曾经|之前|before|used to", query_text, re.I):
            return
        asks_own = ("我" in query_text and "住" in query_text and "现在" in query_text) or re.search(
            r"\bhome now\b|\blive now\b|\bwhere do i live\b", query_text, re.I
        )
        if not asks_own:
            return
        best = max(hit.score for hit in hits)
        margin = self.settings.hop_residence_bonus
        for hit in hits:
            if "user residence = " in hit.content and "status=current" in hit.content:
                hit.score = max(hit.score, best + margin)

    def purge_all(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                DELETE FROM memories;
                DELETE FROM adds;
                DELETE FROM term_df;
                DELETE FROM user_stats;
                INSERT INTO memories_fts(memories_fts) VALUES('delete-all');
                """
            )
            self._conn.commit()

    def _prepare_message(self, message: dict) -> dict:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ContractError(422, "message role must be user or assistant")
        content = message.get("content")
        text = flatten_content(content).strip()
        if not text and not _has_image(content):
            raise ContractError(422, "message content is empty")
        created_ms = message.get("timestamp")
        if created_ms is None:
            created_ms = int(time.time() * 1000)
        try:
            created_ms = int(created_ms)
        except (TypeError, ValueError) as exc:
            raise ContractError(422, "message timestamp must be unix milliseconds") from exc
        content_json = json.dumps(content, ensure_ascii=False) if isinstance(content, list) else None
        stored = text or "[image]"
        return {
            "role": role,
            "text": text or "[image]",
            "stored": f"[{_iso(created_ms)}] {role}: {stored}",
            "content_json": content_json,
            "created_ms": created_ms,
            "when": _iso(created_ms),
        }

    def _apply_fact(self, fact: Fact, user_id: str, session_id: str, request_id: str, role: str, created_ms: int, when: str, embedding) -> None:
        if fact.mode == "correct":
            self._correct_fact(
                fact,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                role=role,
                created_ms=created_ms,
                when=when,
            )
            return
        if fact.mode == "retract":
            self._retract(user_id, fact, created_ms)
            status = "retracted"
            active = 0
        elif fact.mode == "archive":
            status = "previous"
            active = 0
        else:
            status = "current"
            active = 1
        content = fact.render(status, when)
        public_id = self._insert(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            role=role,
            kind="fact",
            content=content,
            content_json=None,
            plain_text=content,
            created_ms=created_ms,
            embedding=embedding,
            attr_key=fact.attr_key,
            fact_value=fact.value,
            active=active,
            entities=fact.entities,
        )
        if fact.mode == "replace":
            self._supersede_previous(user_id, fact.attr_key, public_id, created_ms)

    def _correct_fact(self, fact: Fact, user_id: str, session_id: str, request_id: str, role: str, created_ms: int, when: str) -> None:
        old = fact.replaces
        rows = self._conn.execute(
            """
            SELECT id, content, fact_value, attr_key FROM memories
            WHERE user_id = ? AND active = 1 AND kind = 'fact'
            ORDER BY created_ms DESC
            """,
            (user_id,),
        ).fetchall()
        subject, attribute = "user", "residence"
        for row in rows:
            if old and old in (row["fact_value"] or ""):
                replaced = row["content"].replace("status=current", "status=retracted", 1)
                self._conn.execute(
                    "UPDATE memories SET active = 0, content = ? WHERE id = ?",
                    (replaced, row["id"]),
                )
                attr_key = row["attr_key"] or ""
                if "|" in attr_key:
                    subject, attribute = attr_key.split("|", 1)
                break
        corrected = Fact(
            subject=subject,
            attribute=attribute,
            value=fact.value,
            evidence=fact.evidence,
            mode="replace",
            entities=fact.entities,
        )
        vector = self.embedder.embed(corrected.render("current", when))
        self._apply_fact(
            corrected,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            role=role,
            created_ms=created_ms,
            when=when,
            embedding=vector,
        )

    def _supersede_previous(self, user_id: str, attr_key: str, public_id: str, created_ms: int) -> None:
        rows = self._conn.execute(
            """
            SELECT id, public_id, content FROM memories
            WHERE user_id = ? AND attr_key = ? AND active = 1 AND public_id != ? AND kind = 'fact'
            """,
            (user_id, attr_key, public_id),
        ).fetchall()
        for row in rows:
            replaced = row["content"].replace("status=current", "status=previous", 1)
            self._conn.execute(
                "UPDATE memories SET active = 0, superseded_by = ?, content = ? WHERE id = ?",
                (public_id, replaced, row["id"]),
            )

    def _retract(self, user_id: str, fact: Fact, created_ms: int) -> None:
        needle = fact.value.lower()
        rows = self._conn.execute(
            """
            SELECT id, content, fact_value FROM memories
            WHERE user_id = ? AND attr_key = ? AND active = 1 AND kind = 'fact'
            """,
            (user_id, fact.attr_key),
        ).fetchall()
        for row in rows:
            value = (row["fact_value"] or "").lower()
            if needle == "*" or needle in value or value in needle:
                replaced = row["content"].replace("status=current", "status=retracted", 1)
                self._conn.execute(
                    "UPDATE memories SET active = 0, content = ? WHERE id = ?",
                    (replaced, row["id"]),
                )

    def _insert(
        self,
        user_id: str,
        session_id: str,
        request_id: str,
        role: str,
        kind: str,
        content: str,
        content_json: str | None,
        plain_text: str,
        created_ms: int,
        embedding,
        attr_key: str | None,
        fact_value: str | None,
        active: int,
        entities: list[str] | None = None,
    ) -> str:
        tokens = tokenize(plain_text)
        if entities:
            tokens = tokens + [tok for tok in entities if tok not in tokens]
        entity_text = " ".join(entities or [])
        indexed = " ".join(tokens)
        doc_len = max(len(tokenize(plain_text)), 1)
        cursor = self._conn.execute(
            """
            INSERT INTO memories(
              public_id, user_id, session_id, request_id, role, kind, content, content_json,
              indexed_text, attr_key, fact_value, active, created_ms, entities, embedding, doc_len
            ) VALUES ('pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_id,
                request_id,
                role,
                kind,
                content,
                content_json,
                indexed,
                attr_key,
                fact_value,
                active,
                created_ms,
                entity_text,
                pack_vector(embedding),
                doc_len,
            ),
        )
        row_id = cursor.lastrowid
        public_id = f"m{row_id}"
        self._conn.execute("UPDATE memories SET public_id = ? WHERE id = ?", (public_id, row_id))
        fts_text = f"{user_token(user_id)} {indexed}"
        self._conn.execute(
            "INSERT INTO memories_fts(indexed_text, public_id) VALUES (?, ?)",
            (fts_text, public_id),
        )
        self._bump_stats(user_id, tokens, doc_len)
        return public_id

    def _bump_stats(self, user_id: str, tokens: list[str], doc_len: int) -> None:
        unique = set(tokens)
        for term in unique:
            self._conn.execute(
                """
                INSERT INTO term_df(user_id, term, df) VALUES (?, ?, 1)
                ON CONFLICT(user_id, term) DO UPDATE SET df = df + 1
                """,
                (user_id, term),
            )
        self._conn.execute(
            """
            INSERT INTO user_stats(user_id, n_docs, sum_len) VALUES (?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET n_docs = n_docs + 1, sum_len = sum_len + ?
            """,
            (user_id, doc_len, doc_len),
        )

    def _upsert_profile(self, user_id: str, session_id: str, created_ms: int) -> None:
        rows = self._conn.execute(
            """
            SELECT attr_key, fact_value, content FROM memories
            WHERE user_id = ? AND kind = 'fact' AND active = 1
            ORDER BY created_ms ASC
            """,
            (user_id,),
        ).fetchall()
        if not rows:
            self._conn.execute(
                "UPDATE memories SET content = ?, active = 0 WHERE user_id = ? AND kind = 'profile'",
                ("Active profile:", user_id),
            )
            return
        lines = ["Active profile:"]
        for row in rows[-40:]:
            lines.append(f"- {row['content'].split('. evidence:')[0]}")
        text = "\n".join(lines)
        existing = self._conn.execute(
            "SELECT id, public_id FROM memories WHERE user_id = ? AND kind = 'profile'",
            (user_id,),
        ).fetchone()
        vector = self.embedder.embed(text)
        if existing:
            tokens = tokenize(text)
            self._conn.execute(
                """
                UPDATE memories
                SET content = ?, indexed_text = ?, embedding = ?, doc_len = ?, created_ms = ?, active = 1
                WHERE id = ?
                """,
                (text, " ".join(tokens), pack_vector(vector), max(len(tokens), 1), created_ms, existing["id"]),
            )
            return
        self._insert(
            user_id=user_id,
            session_id=session_id,
            request_id="profile",
            role="assistant",
            kind="profile",
            content=text,
            content_json=None,
            plain_text=text,
            created_ms=created_ms,
            embedding=vector,
            attr_key="user|profile",
            fact_value=None,
            active=1,
        )

    def _candidate_hits(self, user_id: str, q_terms: list[str]) -> list[_Hit]:
        count = self._conn.execute("SELECT COUNT(*) AS n FROM memories WHERE user_id = ?", (user_id,)).fetchone()["n"]
        if count <= self.settings.full_scan_limit:
            rows = self._conn.execute("SELECT * FROM memories WHERE user_id = ?", (user_id,)).fetchall()
            return [self._hit(row, 0) for row in rows]
        match_terms = [term for term in q_terms if term.isalnum() or _cjk(term)][:24]
        if not match_terms:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY created_ms DESC LIMIT ?",
                (user_id, self.settings.candidate_limit),
            ).fetchall()
            return [self._hit(row, 0) for row in rows]
        clauses = " OR ".join(f'"{term}"' for term in match_terms)
        match = f'"{user_token(user_id)}" AND ({clauses})'
        found = self._conn.execute(
            "SELECT public_id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, self.settings.candidate_limit),
        ).fetchall()
        ids = [row["public_id"] for row in found]
        fact_rows = self._conn.execute(
            """
            SELECT public_id FROM memories
            WHERE user_id = ? AND ((kind = 'fact' AND active = 1) OR kind = 'profile')
            ORDER BY created_ms DESC LIMIT 220
            """,
            (user_id,),
        ).fetchall()
        for row in fact_rows:
            if row["public_id"] not in ids:
                ids.append(row["public_id"])
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE user_id = ? AND public_id IN ({placeholders})",
            (user_id, *ids),
        ).fetchall()
        return [self._hit(row, 0) for row in rows]

    def _second_hop(self, user_id: str, seed: list[_Hit], seen: set[str]) -> list[_Hit]:
        entities: list[str] = []
        for hit in seed:
            for token in hit.entities[:8]:
                if len(token) > 1 and token not in entities:
                    entities.append(token)
            if len(entities) >= 12:
                break
        if not entities:
            return []
        clauses = " OR ".join(f'"{token}"' for token in entities if token.isalnum() or _cjk(token))
        if not clauses:
            return []
        match = f'"{user_token(user_id)}" AND ({clauses})'
        try:
            found = self._conn.execute(
                "SELECT public_id FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT 40",
                (match,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        ids = [row["public_id"] for row in found if row["public_id"] not in seen]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE user_id = ? AND public_id IN ({placeholders})",
            (user_id, *ids),
        ).fetchall()
        return [self._hit(row, 1) for row in rows]

    def _hit(self, row: sqlite3.Row, hop: int) -> _Hit:
        tokens = [tok for tok in (row["indexed_text"] or "").split() if tok]
        return _Hit(
            public_id=row["public_id"],
            content=row["content"],
            content_json=row["content_json"],
            kind=row["kind"],
            active=int(row["active"]),
            created_ms=int(row["created_ms"]),
            entities=[tok for tok in (row["entities"] or "").split() if tok],
            tokens=tokens,
            tf=dict(Counter(tokens)),
            doc_len=int(row["doc_len"]),
            embedding=unpack_vector(row["embedding"]),
            hop=hop,
        )

    def _dfs(self, user_id: str, terms: set[str]) -> dict[str, int]:
        if not terms:
            return {}
        placeholders = ",".join("?" for _ in terms)
        rows = self._conn.execute(
            f"SELECT term, df FROM term_df WHERE user_id = ? AND term IN ({placeholders})",
            (user_id, *sorted(terms)),
        ).fetchall()
        return {row["term"]: int(row["df"]) for row in rows}

    def _features(self, hit: _Hit, max_bm25: float, query_vector, q_terms, q_entities, q_numbers, intent, min_ms, max_ms, option_sets) -> dict[str, float]:
        span = max(max_ms - min_ms, 1)
        recency = (hit.created_ms - min_ms) / span
        if intent == "history":
            temporal = 0.7 * (1 - hit.active) + 0.3 * (1 - recency)
        elif intent == "state":
            temporal = 0.75 * hit.active + 0.25 * recency
        else:
            temporal = 0.45 + 0.15 * hit.active
        if intent == "quote":
            kind = 1.0 if hit.kind == "turn" else 0.25
        elif intent == "broad":
            kind = 1.0 if hit.kind == "profile" else 0.35
        elif intent == "history":
            kind = 1.0 if hit.kind == "fact" and not hit.active else 0.4
        elif intent == "state":
            kind = 1.0 if hit.kind == "fact" and hit.active else 0.42
        else:
            kind = 0.7 if hit.kind == "fact" else 0.5
        if q_numbers:
            blob = hit.content
            literal = sum(1 for number in q_numbers if number in blob) / len(q_numbers)
        else:
            literal = 0.0
        option = 0.0
        if option_sets:
            doc_set = set(hit.tokens)
            option = max((jaccard(doc_set, option) for option in option_sets), default=0.0)
        profile = 1.0 if hit.kind == "profile" and intent == "broad" else 0.0
        dense = (cosine(query_vector, hit.embedding) + 1.0) / 2.0
        return {
            "bm25": hit.raw_bm25 / max_bm25,
            "dense": dense,
            "overlap": overlap_ratio(q_terms, hit.tokens),
            "entity": jaccard(q_entities, hit.entities or hit.tokens),
            "literal": literal,
            "temporal": temporal,
            "kind": kind,
            "hop": float(hit.hop),
            "option": option,
            "profile": profile,
        }

    def _mmr(self, pool: list[_Hit], top_k: int) -> list[_Hit]:
        if not pool:
            return []
        lo = min(item.score for item in pool)
        hi = max(item.score for item in pool)
        span = hi - lo or 1.0
        relevance = {item.public_id: (item.score - lo) / span for item in pool}
        token_sets = [set(item.tokens) for item in pool]
        selected: list[_Hit] = []
        selected_sets: list[set[str]] = []
        rest = list(range(len(pool)))
        lam = self.settings.mmr_lambda
        while rest and len(selected) < top_k:
            best_index = 0
            best_value = -1e9
            for index, pool_index in enumerate(rest):
                current = token_sets[pool_index]
                redundancy = 0.0
                if selected_sets and current:
                    for previous in selected_sets:
                        if not previous:
                            continue
                        overlap = len(current & previous)
                        score = overlap / (len(current) + len(previous) - overlap)
                        if score > redundancy:
                            redundancy = score
                value = lam * relevance[pool[pool_index].public_id] - (1 - lam) * redundancy
                if value > best_value:
                    best_value = value
                    best_index = index
            chosen = rest.pop(best_index)
            selected.append(pool[chosen])
            selected_sets.append(token_sets[chosen])
        return selected

    def _render(self, hit: _Hit, budget: int) -> tuple[object, int]:
        if not hit.content_json:
            return hit.content, budget
        try:
            parts = json.loads(hit.content_json)
        except json.JSONDecodeError:
            return hit.content, budget
        kept = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = ((part.get("image_url") or {}).get("url")) or ""
                size = len(url.encode("utf-8"))
                if size > IMAGE_LIMIT or size > budget:
                    continue
                budget -= size
            kept.append(part)
        if not any(_part_nonempty(part) for part in kept):
            return hit.content, budget
        return kept, budget


def _iso(created_ms: int) -> str:
    return datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _has_image(content) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)


def _part_nonempty(part) -> bool:
    if isinstance(part, str) and part.strip():
        return True
    if not isinstance(part, dict):
        return False
    if part.get("type") == "text":
        return bool(str(part.get("text") or "").strip())
    if part.get("type") == "image_url":
        return bool((part.get("image_url") or {}).get("url"))
    return False


def _cjk(token: str) -> bool:
    return bool(token) and all("\u4e00" <= ch <= "\u9fff" for ch in token)
