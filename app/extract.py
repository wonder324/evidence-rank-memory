from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from app.textutil import tokenize

REPLACE_ATTRS = {"residence", "occupation", "name", "phone", "email", "timezone"}


@dataclass
class Fact:
    subject: str
    attribute: str
    value: str
    evidence: str
    mode: str  # replace | append | retract | archive | correct
    entities: list[str] = field(default_factory=list)
    replaces: str = ""

    @property
    def attr_key(self) -> str:
        return f"{self.subject.strip().lower()}|{self.attribute}"

    def render(self, status: str, when: str) -> str:
        names = " ".join(self.entities)
        return (
            f"{self.subject} {self.attribute} = {self.value} "
            f"(status={status}, time={when}). "
            f"entities: {names}. evidence: {self.evidence}"
        )


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" \t\r\n.,;:!?。！？；，")
    return value[:240]


def _entities(*chunks: str) -> list[str]:
    found: list[str] = []
    seen = set()
    for chunk in chunks:
        for tok in tokenize(chunk):
            if len(tok) <= 1 or tok in seen:
                continue
            seen.add(tok)
            found.append(tok)
        for match in re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", chunk or ""):
            key = match.lower()
            if key not in seen:
                seen.add(key)
                found.append(key)
    return found[:24]


def _add(facts: list[Fact], subject: str, attribute: str, value: str, evidence: str, mode: str, replaces: str = "") -> None:
    value = _clean(value)
    subject = _clean(subject) or "user"
    if not value:
        return
    facts.append(
        Fact(
            subject=subject,
            attribute=attribute,
            value=value,
            evidence=_clean(evidence) or value,
            mode=mode,
            entities=_entities(subject, value, evidence),
            replaces=_clean(replaces),
        )
    )


_NOT_PERSON = {"我以前", "我现在", "我曾经", "以前", "现在", "如今", "曾经", "过去", "原来", "之前", "目前", "后来"}


def _not_person(name: str) -> bool:
    return _clean(name) in _NOT_PERSON


PATTERNS: list[tuple[re.Pattern[str], str, str, str]] = [
    (re.compile(r"\bmy name is ([^.!?\n]+)", re.I), "user", "name", "replace"),
    (re.compile(r"\bi am ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"), "user", "name", "replace"),
    (re.compile(r"我叫([\u4e00-\u9fffA-Za-z0-9·]{1,32})"), "user", "name", "replace"),
    (re.compile(r"我的名字是([\u4e00-\u9fffA-Za-z0-9·]{1,32})"), "user", "name", "replace"),
    (re.compile(r"我以前的名字是([\u4e00-\u9fffA-Za-z0-9·]{1,32})"), "user", "name", "archive"),
    (re.compile(r"我改名为([\u4e00-\u9fffA-Za-z0-9·]{1,32})"), "user", "name", "replace"),
    (re.compile(r"\bi (?:now )?live in ([^.!?\n]+)", re.I), "user", "residence", "replace"),
    (re.compile(r"\bi moved to ([^.!?\n]+)", re.I), "user", "residence", "replace"),
    (re.compile(r"\bi'm from ([^.!?\n]+)", re.I), "user", "residence", "replace"),
    (re.compile(r"我(?:现在)?住在([^，。！？\n]{1,40})"), "user", "residence", "replace"),
    (re.compile(r"我搬到(?:了)?([^，。！？\n]{1,40})"), "user", "residence", "replace"),
    (re.compile(r"(?:后来)?我搬去(?:了)?([^，。！？\n]{1,40})"), "user", "residence", "replace"),
    (re.compile(r"\bi used to work as (?:an? )?([^.!?\n]+?)\.?$", re.I), "user", "occupation", "archive"),
    (re.compile(r"\bi work as (?:an? )?([^.!?\n]+?)(?: now)?\.?$", re.I), "user", "occupation", "replace"),
    (re.compile(r"\bi'm a ([^.!?\n]+)", re.I), "user", "occupation", "replace"),
    (re.compile(r"我的职业是([^，。！？\n]{1,40})"), "user", "occupation", "replace"),
    (re.compile(r"我是一名([^，。！？\n]{1,40})"), "user", "occupation", "replace"),
    (re.compile(r"\bmy (?:favorite|favourite) ([a-z ]{2,24}) is ([^.!?\n]+)", re.I), "user", "preference", "append"),
    (re.compile(r"\bi (?:like|love|prefer|enjoy) ([^.!?\n]+)", re.I), "user", "preference", "append"),
    (re.compile(r"我喜欢([^，。！？\n]{1,40})"), "user", "preference", "append"),
    (re.compile(r"\bi (?:don't|do not) like ([^.!?\n]+)", re.I), "user", "preference", "retract"),
    (re.compile(r"我不喜欢([^，。！？\n]{1,40})"), "user", "preference", "retract"),
    (re.compile(r"\bnever ([^.!?\n]+)", re.I), "user", "rule", "append"),
    (re.compile(r"\balways ([^.!?\n]+)", re.I), "user", "rule", "append"),
    (re.compile(r"不要([^，。！？\n]{1,40})"), "user", "rule", "append"),
    (re.compile(r"必须([^，。！？\n]{1,40})"), "user", "rule", "append"),
    (re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) lives in ([^.!?\n]+)"), "name", "residence", "replace"),
    (re.compile(r"([\u4e00-\u9fff]{2,8})住在([^，。！？\n]{1,40})"), "name", "residence", "replace"),
]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def extract_facts(text: str) -> list[Fact]:
    from app.textutil import looks_like_code

    if not text or looks_like_code(text):
        return []
    facts: list[Fact] = []
    for sentence in _sentences(text):
        used_to = re.search(
            r"\bi used to live in ([^.!?\n]+?) but now i live in ([^.!?\n]+)",
            sentence,
            re.I,
        )
        if used_to:
            _add(facts, "user", "residence", used_to.group(1), sentence, "archive")
            _add(facts, "user", "residence", used_to.group(2), sentence, "replace")
        zh_move = re.search(
            r"我以前住在([^，。！？\n]{1,40})[，,]\s*现在住在([^，。！？\n]{1,40})",
            sentence,
        )
        if zh_move:
            _add(facts, "user", "residence", zh_move.group(1), sentence, "archive")
            _add(facts, "user", "residence", zh_move.group(2), sentence, "replace")
        past_home = re.search(r"我以前住在([^，。！？\n]{1,40})", sentence)
        if past_home:
            _add(facts, "user", "residence", past_home.group(1), sentence, "archive")
        once_home = re.search(
            r"我曾经住在([^，。！？\n]{1,40})(?:[，,]\s*后来搬到了([^，。！？\n]{1,40}))?",
            sentence,
        )
        if once_home:
            _add(facts, "user", "residence", once_home.group(1), sentence, "archive")
            if once_home.group(2):
                _add(facts, "user", "residence", once_home.group(2), sentence, "replace")
        now_home = re.search(r"(?:我)?现在住在([^，。！？\n]{1,40})", sentence)
        if now_home:
            _add(facts, "user", "residence", now_home.group(1), sentence, "replace")
        pref_shift = re.search(
            r"我以前喜欢([^，。！？\n]{1,40})[，,]\s*现在喜欢([^，。！？\n]{1,40})",
            sentence,
        )
        if pref_shift and os.environ.get("SCHEME_PREF", "1").strip() in {"1", "true", "yes"}:
            _add(facts, "user", "preference", pref_shift.group(1), sentence, "archive")
            _add(facts, "user", "preference", pref_shift.group(2), sentence, "replace")
        instead = re.search(
            r"\bi (?:now )?prefer ([^.!?\n]+?) instead of ([^.!?\n]+)",
            sentence,
            re.I,
        )
        if instead:
            _add(facts, "user", "preference", instead.group(2), sentence, "retract")
            _add(facts, "user", "preference", instead.group(1), sentence, "append")
        schedule = re.search(
            r"\b(?:meet|met|meeting|appointment) (?:with )?([^.!?\n]{0,60}?)(?:\bat\b|\bon\b) ([^.!?\n]+)",
            sentence,
            re.I,
        )
        if schedule and _clean(schedule.group(2)):
            who = _clean(schedule.group(1)) or "someone"
            _add(facts, "user", "event", f"met {who} at {schedule.group(2)}", sentence, "append")
        zh_meet = re.search(r"我(周[一二三四五六日天]|星期[一二三四五六日天])(?:和|与)([\u4e00-\u9fffA-Za-z]{1,20})见面", sentence)
        if zh_meet:
            day = zh_meet.group(1)
            who = zh_meet.group(2)
            order = "一二三四五六日天".find(day[-1]) + 1
            _add(facts, "user", "event", f"在{day}和{who}见面", sentence, "append")
            _add(facts, "user", "met", f"{order}:{day}:{who}", sentence, "append")
        _extract_links(facts, sentence)
        for pattern, subject, attribute, mode in PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            if attribute == "preference" and subject == "user" and pattern.pattern.startswith(r"\bmy (?:favorite"):
                _add(facts, "user", "preference", f"{match.group(1).strip()}: {match.group(2)}", sentence, "append")
                continue
            if subject == "name":
                if match.lastindex and match.lastindex >= 2 and not _not_person(match.group(1)):
                    _add(facts, match.group(1), attribute, match.group(2), sentence, mode)
                continue
            value = match.group(1)
            if mode == "append" and attribute == "rule":
                value = sentence
            _add(facts, subject, attribute, value, sentence, mode)
    deduped: list[Fact] = []
    seen = set()
    for fact in facts:
        if fact.attribute == "rule" and (fact.value.startswith("同事") or fact.evidence.startswith("同事")):
            fact.subject = "同事"
        key = (fact.attr_key, fact.value.lower(), fact.mode)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


_FIELD = {
    "工号": "employee_id",
    "部门": "department",
    "手机号": "phone",
    "邮箱": "email",
    "城市": "city",
    "身份证号": "id_card",
    "住址": "address",
    "病历": "medical",
}


_REL = {
    "导师": "mentor",
    "同事": "colleague",
    "朋友": "friend",
    "同学": "classmate",
    "经理": "manager",
    "学生": "student",
    "mentor": "mentor",
    "colleague": "colleague",
    "friend": "friend",
}


def _extract_links(facts: list[Fact], sentence: str) -> None:
    corrected = re.search(r"(?:记错了|更正一下|更正)[^。]{0,16}不是([^，,]{1,20})[，,]\s*是([^，。]{1,20})", sentence)
    if corrected:
        _add(facts, "user", "other", corrected.group(2), sentence, "correct", replaces=corrected.group(1))
    forgotten = re.search(r"(?:忘掉|忘记|不要再记得?)(?:我)?不喜欢([^，。！？\n]{1,40})", sentence)
    if forgotten:
        _add(facts, "user", "preference", forgotten.group(1), sentence, "retract")
    past_job = re.search(r"我的职业曾经是([^，。！？\n]{1,40})", sentence)
    if past_job:
        _add(facts, "user", "occupation", past_job.group(1), sentence, "archive")
    now_job = re.search(r"我现在的职业是([^，。！？\n]{1,40})", sentence)
    if now_job:
        _add(facts, "user", "occupation", now_job.group(1), sentence, "replace")
    mine = re.search(r"我的(导师|同事|朋友|同学|经理)是([\u4e00-\u9fffA-Za-z]{1,20})", sentence)
    if mine:
        _add(facts, "user", _REL[mine.group(1)], mine.group(2), sentence, "append")
    linked = re.search(r"([\u4e00-\u9fff]{2,8})的(同事|朋友|导师|学生)是([\u4e00-\u9fffA-Za-z]{1,20})", sentence)
    if linked and not _not_person(linked.group(1)):
        _add(facts, linked.group(1), _REL[linked.group(2)], linked.group(3), sentence, "append")
    owned = re.search(r"\b([A-Z][a-z]+)'s (colleague|friend|mentor) is ([A-Z][a-z]+)", sentence)
    if owned:
        _add(facts, owned.group(1), _REL[owned.group(2).lower()], owned.group(3), sentence, "append")
    my_en = re.search(r"\bmy (mentor|colleague|friend) is ([A-Z][a-z]+)", sentence, re.I)
    if my_en:
        _add(facts, "user", _REL[my_en.group(1).lower()], my_en.group(2), sentence, "append")
    met_en = re.search(r"\bI met ([A-Z][a-z]+)", sentence)
    if met_en:
        _add(facts, "user", "met", met_en.group(1), sentence, "append")
    first_stop = re.search(r"我先去了([\u4e00-\u9fff]{2,8})", sentence)
    if first_stop:
        _add(facts, "user", "event", f"1:{first_stop.group(1)}", sentence, "append")
    next_stop = re.search(r"我然后去了([\u4e00-\u9fff]{2,8})", sentence)
    if next_stop:
        _add(facts, "user", "event", f"2:{next_stop.group(1)}", sentence, "append")
    last_stop = re.search(r"我最后去了([\u4e00-\u9fff]{2,8})", sentence)
    if last_stop:
        _add(facts, "user", "event", f"3:{last_stop.group(1)}", sentence, "append")
    numbered = re.search(r"行程第([1-9])站是([^，。！？\n]{1,24})", sentence)
    if numbered:
        _add(facts, "user", "event", f"{numbered.group(1)}:{numbered.group(2)}", sentence, "append")
    own_field = re.search(
        r"我的(工号|部门|手机号|邮箱|城市|身份证号)(?:更新为|改成|是)([^，。！？\n]{1,40})",
        sentence,
    )
    if own_field:
        clause = f"我的{own_field.group(1)}是{own_field.group(2)}"
        _add(facts, "user", _FIELD[own_field.group(1)], own_field.group(2), clause, "replace")
    hidden = re.search(
        r"(?:这是隐私[，,]?)?(?:请)?(?:忘掉|忘记|不要记住)我的(手机号|邮箱|身份证号|住址|病历)",
        sentence,
    )
    if hidden:
        _add(facts, "user", _FIELD[hidden.group(1)], "*", sentence, "retract")
    allergy = re.search(r"我(?:过敏[，,]?)?不能吃([^，。！？\n]{1,40})", sentence)
    if allergy:
        _add(facts, "user", "allergy", allergy.group(1), sentence, "append")
    nickname = re.search(r"请叫我([^，。！？\n]{1,24})", sentence)
    if nickname:
        _add(facts, "user", "nickname", nickname.group(1), sentence, "replace")
    habit = re.search(r"我的(?:写作习惯|说话习惯)：([^。！？\n]{1,80})", sentence)
    if habit:
        _add(facts, "user", "rule", sentence, sentence, "append")
    other_job = re.search(r"^([\u4e00-\u9fff]{2,8})的职业是([^，。！？\n]{1,40})", sentence)
    if other_job and other_job.group(1) not in {"我的", "现在", "曾经"}:
        _add(facts, other_job.group(1), "occupation", other_job.group(2), sentence, "replace")
