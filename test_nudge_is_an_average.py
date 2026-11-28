"""Tests for nudge_is_an_average.py. Each test prints PASS or FAIL."""

from nudge_is_an_average import (backward, forward, loss_of, nudge_ratio,
                                 nudge_weight, slope_at)

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


# --- Avi's notebook numbers ---
check("forward: h 6, y 3", forward(2, 3, 0.5) == (6, 3))
check("loss (3 - 1)^2 = 4", loss_of(3, 1) == 4)
check("nudge 0.01: new loss 4.0401", close(loss_of(3.01, 1), 4.0401))
check("nudge 0.01: ratio 4.01", close(nudge_ratio(3, 1, 0.01), 4.01))
check("nudge 0.001: new loss 4.004001", close(loss_of(3.001, 1), 4.004001))
check("nudge 0.001: ratio 4.001", close(nudge_ratio(3, 1, 0.001), 4.001))
check("slope at current y = 3 is 4", slope_at(3, 1) == 4)
check("slope at nudged y = 3.01 is 4.02 (the slip)", close(slope_at(3.01, 1), 4.02))
check("slope at nudged y = 3.001 is 4.002 (the slip)", close(slope_at(3.001, 1), 4.002))
check("w2 nudge 0.51: y 3.06", close(forward(2, 3, 0.51)[1], 3.06))
check("w2 nudge 0.01: ratio 24.36", close(nudge_weight(2, 3, 0.5, 1, "w2", 0.01), 24.36))

# --- the blames ---
b = backward(2, 3, 0.5, 1)
check("y's blame 4", b["y"] == 4)
check("w2's blame 4 x 6 = 24", b["w2"] == 24)
check("h's blame 4 x 0.5 = 2", b["h"] == 2)
check("w1's blame 2 x 2 = 4", b["w1"] == 4)

# --- the big claim: nudge ratio = average of start slope and landing slope ---
for y in [3, -2, 0.5, 10]:
    for target in [1, 0, -3]:
        for n in [0.5, 0.1, 0.01, -0.2]:
            avg = (slope_at(y, target) + slope_at(y + n, target)) / 2
            check(f"y {y}, target {target}, nudge {n}: ratio = average",
                  close(nudge_ratio(y, target, n), avg))

# --- shrinking nudges close in on the derivative ---
gaps = [abs(nudge_ratio(3, 1, n) - 4) for n in [0.1, 0.01, 0.001]]
check("gap shrinks 0.1 -> 0.01 -> 0.001", gaps[0] > gaps[1] > gaps[2])
check("nudge from below undershoots (3.99)", close(nudge_ratio(3, 1, -0.01), 3.99))

# --- backward = nudge on other machines ---
machines = [(2, 3, 0.5, 1), (1, -2, 3, 4), (-1, 0.5, -4, 2), (3, 1, 1, 3)]
for m in machines:
    back = backward(*m)
    check(f"machine {m}: w1 backward = nudge",
          close(back["w1"], nudge_weight(*m, "w1", 1e-7), 1e-4))
    check(f"machine {m}: w2 backward = nudge",
          close(back["w2"], nudge_weight(*m, "w2", 1e-7), 1e-4))

# --- each partner sees the other ---
check("double h (x 4) doubles w2's blame ratio to y's",
      backward(4, 3, 0.5, 1)["w2"] / backward(4, 3, 0.5, 1)["y"] == 12)
check("on target: every blame is 0",
      all(v == 0 for v in backward(2, 3, 0.5, 3).values()))

print(f"\n{passed} passed, {failed} failed")
