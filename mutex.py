"""mutex.py: exhaustive two-thread interleaving checker for Homework 6."""

from collections import deque

INCREMENTS = 3
VISITS = 2
SIZE = 1  # B-d: reduce the bounded buffer from 2 slots to 1
ITEMS = 3

def put(sh, name, value, nxt):
    sh[name] = value
    return nxt

def again(sh, me, first, done):
    sh[f"left{me}"] -= 1
    return first if sh[f"left{me}"] > 0 else done

def sem_wait(name, nxt):
    def action(sh, me):
        if sh[name] == 0:
            return None
        sh[name] -= 1
        return nxt
    return (f"semWait({name})", action)

def sem_signal(name, nxt):
    return (f"semSignal({name})", lambda sh, me: put(sh, name, sh[name] + 1, nxt))

def append_item(nxt):
    return ("CS: append an item", lambda sh, me: put(sh, "items", sh["items"] + 1, nxt))

def take_item(nxt):
    return ("CS: take an item", lambda sh, me: put(sh, "items", sh["items"] - 1, nxt))

def freeze(sh):
    return tuple(sorted(sh.items()))

def explore(threads, init):
    start = ((0,) * len(threads), freeze(init))
    came_from = {start: None}
    moves = {}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        pcs, frozen = state
        moves[state] = []
        for t, program in enumerate(threads):
            if pcs[t] == len(program):
                continue
            sh = dict(frozen)
            nxt = program[pcs[t]][1](sh, t)
            if nxt is None:
                continue
            new = (pcs[:t] + (nxt,) + pcs[t + 1:], freeze(sh))
            moves[state].append(new)
            if new not in came_from:
                came_from[new] = (state, t)
                queue.append(new)
    return came_from, moves

def trace(came_from, threads, end):
    steps = []
    while came_from[end] is not None:
        before, t = came_from[end]
        steps.append((t, threads[t][before[0][t]][0], end[1]))
        end = before
    return steps[::-1]

def show(steps):
    for t, text, frozen in steps:
        shared = "  ".join(f"{k}={v}" for k, v in frozen if not k.startswith(("left", "reg")))
        print(f"      T{t}  {text:<42} {shared}")

def status(threads, state):
    words = []
    for t, program in enumerate(threads):
        pc = state[0][t]
        if pc == len(program):
            words.append(f"T{t} has finished")
            continue
        sh = dict(state[1])
        nxt = program[pc][1](sh, t)
        if nxt is None:
            how = "is blocked at"
        elif nxt == pc and freeze(sh) == state[1]:
            how = "spins forever on"
        else:
            how = "can still take"
        words.append(f"T{t} {how} '{program[pc][0]}'")
    return "; ".join(words)

def in_cs(program, pc):
    return pc < len(program) and program[pc][0].startswith("CS:")

def check(name, model):
    threads, init = model["threads"], model["init"]
    came_from, moves = explore(threads, init)
    states = list(came_from)
    print(f"  {name}: {len(states)} reachable states")
    done = [s for s in states if all(pc == len(p) for pc, p in zip(s[0], threads))]
    if "result" in model:
        values = sorted({dict(s[1])[model["result"]] for s in done})
        print(f"    possible final values of {model['result']}: " + ", ".join(map(str, values)))
        return
    both_in = [s for s in states if all(in_cs(p, pc) for p, pc in zip(threads, s[0]))]
    if both_in:
        print("    both threads in the critical section at once: POSSIBLE. The shortest way:")
        show(trace(came_from, threads, both_in[0]))
    else:
        print("    both threads in the critical section at once: never")
    if "broken" in model:
        bad = [s for s in states if model["broken"](dict(s[1]))]
        if bad:
            print(f"    {model['broken'](dict(bad[0][1]))}: POSSIBLE. The shortest way:")
            show(trace(came_from, threads, bad[0]))
        else:
            print("    buffer overflow or underflow: never")
    behind = {}
    for state, outs in moves.items():
        for out in outs:
            behind.setdefault(out, []).append(state)
    can_finish = set(done)
    queue = deque(done)
    while queue:
        state = queue.popleft()
        for before in behind.get(state, []):
            if before not in can_finish:
                can_finish.add(before)
                queue.append(before)
    stuck = [s for s in states if s not in can_finish]
    if stuck:
        traps = [s for s in stuck if all(out == s for out in moves[s])]
        end = (traps or stuck)[0]
        where = "a dead end (a state no step can change)" if traps else "a stuck state"
        print(f"    can always still finish: NO ({len(stuck)} of {len(states)} states are stuck). The shortest way to {where}:")
        show(trace(came_from, threads, end))
        print(f"      stuck: {status(threads, end)}")
    else:
        print("    can always still finish: yes")

def increment_race():
    program = [
        ("reg = count", lambda sh, me: put(sh, f"reg{me}", sh["count"], 1)),
        ("reg = reg + 1", lambda sh, me: put(sh, f"reg{me}", sh[f"reg{me}"] + 1, 2)),
        ("count = reg", lambda sh, me: put(sh, "count", sh[f"reg{me}"], 3)),
        ("repeat if increments remain", lambda sh, me: again(sh, me, 0, 4)),
    ]
    init = {"count": 0, "reg0": 0, "reg1": 0, "left0": INCREMENTS, "left1": INCREMENTS}
    return {"threads": [program, program], "init": init, "result": "count"}

def look_then_flag():
    program = [
        ("while flag[other]: wait", lambda sh, me: 0 if sh[f"flag{1-me}"] else 1),
        ("flag[me] = True", lambda sh, me: put(sh, f"flag{me}", True, 2)),
        ("CS: critical section", lambda sh, me: 3),
        ("flag[me] = False", lambda sh, me: put(sh, f"flag{me}", False, 4)),
        ("repeat if visits remain", lambda sh, me: again(sh, me, 0, 5)),
    ]
    init = {"flag0": False, "flag1": False, "left0": VISITS, "left1": VISITS}
    return {"threads": [program, program], "init": init}

def flag_then_look():
    program = [
        ("flag[me] = True", lambda sh, me: put(sh, f"flag{me}", True, 1)),
        ("while flag[other]: wait", lambda sh, me: 1 if sh[f"flag{1-me}"] else 2),
        ("CS: critical section", lambda sh, me: 3),
        ("flag[me] = False", lambda sh, me: put(sh, f"flag{me}", False, 4)),
        ("repeat if visits remain", lambda sh, me: again(sh, me, 0, 5)),
    ]
    init = {"flag0": False, "flag1": False, "left0": VISITS, "left1": VISITS}
    return {"threads": [program, program], "init": init}

def strict_turns():
    program = [
        ("while turn != me: wait", lambda sh, me: 0 if sh["turn"] != me else 1),
        ("CS: critical section", lambda sh, me: 2),
        ("turn = other", lambda sh, me: put(sh, "turn", 1 - me, 3)),
        ("repeat if visits remain", lambda sh, me: again(sh, me, 0, 4)),
    ]
    init = {"turn": 0, "left0": 1, "left1": VISITS}
    return {"threads": [program, program], "init": init}

def peterson():
    program = [
        ("flag[me] = True", lambda sh, me: put(sh, f"flag{me}", True, 1)),
        ("turn = other", lambda sh, me: put(sh, "turn", 1 - me, 2)),
        ("while flag[other] and turn == other: wait", lambda sh, me: 2 if (sh[f"flag{1-me}"] and sh["turn"] == 1 - me) else 3),
        ("CS: critical section", lambda sh, me: 4),
        ("flag[me] = False", lambda sh, me: put(sh, f"flag{me}", False, 5)),
        ("repeat if visits remain", lambda sh, me: again(sh, me, 0, 6)),
    ]
    init = {"flag0": False, "flag1": False, "turn": 0, "left0": VISITS, "left1": VISITS}
    return {"threads": [program, program], "init": init}

LOCKS = [("Look, then flag", look_then_flag), ("Flag, then look", flag_then_look), ("Strict turns", strict_turns), ("Peterson's algorithm", peterson)]

def bounded_buffer():
    producer = [
        sem_wait("e", 1), sem_wait("s", 2), append_item(3),
        sem_signal("s", 4), sem_signal("n", 5),
        ("repeat if items remain", lambda sh, me: again(sh, me, 0, 6)),
    ]
    consumer = [
        sem_wait("n", 1), sem_wait("s", 2), take_item(3),
        sem_signal("s", 4), sem_signal("e", 5),
        ("repeat if items remain", lambda sh, me: again(sh, me, 0, 6)),
    ]
    init = {"s": 1, "n": 0, "e": SIZE, "items": 0, "left0": ITEMS, "left1": ITEMS}
    broken = lambda sh: ("buffer overflow" if sh["items"] > SIZE else "buffer underflow" if sh["items"] < 0 else None)
    return {"threads": [producer, consumer], "init": init, "broken": broken}

def main():
    print(f"Experiment 1: two threads each add 1 to count {INCREMENTS} times (count = count + 1 is load, add, store)")
    check("Increment race", increment_race())
    print("\nExperiment 2: four ways to guard a critical section")
    for name, build in LOCKS:
        model = build()
        visits = f"visits: T0 {model['init']['left0']}, T1 {model['init']['left1']}"
        check(f"{name} ({visits})", model)
    print(f"\nExperiment 3: a producer and a consumer pass {ITEMS} items through a buffer that holds {SIZE}")
    check("Bounded buffer", bounded_buffer())

if __name__ == "__main__":
    main()
