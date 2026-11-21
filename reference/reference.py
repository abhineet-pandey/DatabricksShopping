#!/usr/bin/env python3
"""Independent, straightforward reference for rules.md (slow, for testing).

Usage: reference.py DATA_DIR OUTPUT_DIR
Writes OUTPUT_DIR/user_daily.csv and OUTPUT_DIR/campaign_daily.csv in the
fixtures' expected format (day as YYYY-MM-DD, empty top_category for null).
Shares no code with the native pipeline.
"""
import calendar
import datetime
import glob
import json
import os
import re
import sys
from collections import defaultdict

DAY = 86_400_000
GAP = 1_800_000
REACH = 7 * DAY
ISO = re.compile(rb"^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(Z|[+-][0-9]{2}:[0-9]{2})$")
DIGITS = re.compile(rb"^[0-9]+$")


def split(line, n):
    """Strict dialect: returns decoded fields or None if malformed."""
    if line == b"":
        return None
    parts = line.split(b",")
    if len(parts) != n:
        return None
    out = []
    for p in parts:
        if b'"' in p:
            if len(p) >= 2 and p[0:1] == b'"' and p[-1:] == b'"' and b'"' not in p[1:-1]:
                p = p[1:-1]
            else:
                return None
        out.append(p)
    return out


def rows(path, n):
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines.pop()
    for line in lines[1:]:
        r = split(line, n)
        if r is not None:
            yield r


def parse_ts(s):
    if s == b"":
        return None
    if DIGITS.match(s):
        return int(s)
    m = ISO.match(s)
    if not m:
        return None
    y, mo, d, h, mi, se = (int(x) for x in m.groups()[:6])
    if not (1 <= mo <= 12) or h > 23 or mi > 59 or se > 59 or d < 1:
        return None
    if y == 0:
        return None  # year 0000 is far outside every window either way
    if d > calendar.monthrange(y, mo)[1]:
        return None
    secs = calendar.timegm((y, mo, d, h, mi, se, 0, 0, 0))
    z = m.group(7)
    if z != b"Z":
        oh, om = int(z[1:3]), int(z[4:6])
        if oh > 23 or om > 59:
            return None
        off = oh * 3600 + om * 60
        secs -= off if z[0:1] == b"+" else -off
    return secs * 1000


def main():
    data, out = sys.argv[1], sys.argv[2]
    man = json.load(open(os.path.join(data, "manifest.json")))
    w0 = parse_ts(man["window_start"].encode())
    w1 = parse_ts(man["window_end"].encode())

    products = {p: c for p, c in rows(os.path.join(data, "products.csv"), 2)}
    campaigns = {c: ch for c, ch in rows(os.path.join(data, "campaigns.csv"), 2)}

    # links: union-find over byte strings
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in sorted(glob.glob(os.path.join(data, "links", "*.csv"))):
        for a, b in rows(f, 2):
            if a == b"" or b == b"":
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    members = defaultdict(list)
    for x in list(parent):
        members[find(x)].append(x)
    name = {}
    for r, ms in members.items():
        m = min(ms)
        for x in ms:
            name[x] = m

    def person(u):
        return name.get(u, u)

    # events: dedup on event_id keeping min ingest, then validate
    best = {}
    for f in sorted(glob.glob(os.path.join(data, "events", "*.csv"))):
        for r in rows(f, 7):
            eid = r[0]
            if eid == b"":
                continue
            ing = parse_ts(r[3])
            if ing is None:
                ing = float("inf")
            cur = best.get(eid)
            if cur is None or ing < cur[0]:
                best[eid] = (ing, r)
    events = defaultdict(list)  # person -> [(ts, price, category)]
    for eid, (ing, r) in best.items():
        _, user, ts, _, typ, prod, price = r
        if user == b"" or prod == b"":
            continue
        t = parse_ts(ts)
        if t is None or not (w0 <= t < w1):
            continue
        if typ not in (b"view", b"cart", b"purchase"):
            continue
        p = 0
        cat = None
        if typ == b"purchase":
            if not DIGITS.match(price):
                continue
            p = int(price)
            if not (1 <= p <= 10_000_000):
                continue
            cat = products.get(prod, b"unknown")
        if ing - t > DAY:
            continue
        events[person(user)].append((t, p, cat))

    clicks = defaultdict(dict)  # person -> click_id -> (ts, campaign)
    for f in sorted(glob.glob(os.path.join(data, "clicks", "*.csv"))):
        for cid, user, camp, ts in rows(f, 4):
            if cid == b"" or user == b"" or camp not in campaigns:
                continue
            t = parse_ts(ts)
            if t is None or not (w0 - REACH <= t < w1):
                continue
            clicks[person(user)][cid] = (t, camp)

    user_rows = {}
    camp_rows = defaultdict(lambda: [0, 0, 0, 0])
    for pid, evs in events.items():
        evs.sort(key=lambda e: e[0])
        sessions = []
        for e in evs:
            if sessions and e[0] - sessions[-1][-1][0] <= GAP:
                sessions[-1].append(e)
            else:
                sessions.append([e])
        my_clicks = sorted((t, cid, camp) for cid, (t, camp) in clicks.get(pid, {}).items())
        for s in sessions:
            start = s[0][0]
            day = datetime.datetime.fromtimestamp(start // 1000, datetime.timezone.utc).date()
            ur = user_rows.setdefault((pid, day), [0, 0, 0, defaultdict(int)])
            ur[0] += 1
            ur[1] += len(s)
            rev = sum(e[1] for e in s)
            ur[2] += rev
            for e in s:
                if e[1] > 0:
                    ur[3][e[2]] += e[1]
            eligible = [c for c in my_clicks if start - REACH <= c[0] <= start]
            if eligible:
                latest = max(c[0] for c in eligible)
                win = min(c[1] for c in eligible if c[0] == latest)
                camp = next(c[2] for c in eligible if c[0] == latest and c[1] == win)
            else:
                camp = b"organic"
            cr = camp_rows[(camp, day)]
            cr[0] += 1
            cr[1] += 1 if rev > 0 else 0
            cr[2] += rev
            for c in {c[2] for c in eligible}:
                camp_rows[(c, day)][3] += 1

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "user_daily.csv"), "wb") as f:
        f.write(b"user_id,day,sessions,events,revenue_cents,top_category\n")
        for (pid, day), (n, ev, rev, cats) in sorted(user_rows.items()):
            top = b""
            if cats:
                top = min(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            f.write(b"%s,%s,%d,%d,%d,%s\n" % (pid, day.isoformat().encode(), n, ev, rev, top))
    with open(os.path.join(out, "campaign_daily.csv"), "wb") as f:
        f.write(b"campaign_id,channel,day,sessions,purchasing_sessions,revenue_cents,assisted_sessions\n")
        for (camp, day), (n, pn, rev, a) in sorted(camp_rows.items()):
            ch = b"organic" if camp == b"organic" else campaigns[camp]
            f.write(b"%s,%s,%s,%d,%d,%d,%d\n" % (camp, ch, day.isoformat().encode(), n, pn, rev, a))


if __name__ == "__main__":
    main()
