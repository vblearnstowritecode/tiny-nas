"""Tests for two_blames_per_hop.py. Each test prints PASS or FAIL."""

from two_blames_per_hop import (backward, blame_at_end, error_of, forward,
                                nudge_a, nudge_b, nudge_output)

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


def close(x, y, tol=1e-3):
    return abs(x - y) < tol


# --- the lesson machine: input 1, a 2, b 0.5, target 3 ---
check("forward: middle 2, output 1", forward(1, 2, 0.5) == (2, 1))
check("error (3 - 1)^2 = 4", error_of(1, 3) == 4)
check("Avi's fill-in: output 1.1 -> error 3.61", close(error_of(1.1, 3), 3.61, 1e-9))
check("Avi's fill-in: nudge 0.1 -> -3.9", close(nudge_output(1, 3, 0.1), -3.9, 1e-9))

lesson = backward(1, 2, 0.5, 3)
check("end blame -4", lesson["output"] == -4)
check("b's blame = -4 x middle 2 = -8", lesson["b"] == -8)
check("middle's blame = -4 x b 0.5 = -2", lesson["middle"] == -2)
check("a's blame = -2 x input 1 = -2", lesson["a"] == -2)

# --- blame at the end is local: it depends on where the output is ---
check("output 1 -> blame -4 (loud)", blame_at_end(1, 3) == -4)
check("output 2 -> blame -2", blame_at_end(2, 3) == -2)
check("output 3 -> blame 0 (hit the target)", blame_at_end(3, 3) == 0)
check("output 4 -> blame +2 (overshot, push down)", blame_at_end(4, 3) == 2)

# --- smaller nudges get closer to the exact blame ---
gaps = [abs(nudge_output(1, 3, n) - (-4)) for n in [0.1, 0.01, 0.001]]
check("output nudge: gap shrinks 0.1 -> 0.01 -> 0.001",
      gaps[0] > gaps[1] > gaps[2])
gaps_a = [abs(nudge_a(1, 2, 0.5, 3, n) - (-2)) for n in [0.1, 0.01, 0.001]]
check("a nudge: gap shrinks too", gaps_a[0] > gaps_a[1] > gaps_a[2])

# --- the bet: nudging a (b frozen) matches the backward walk ---
machines = [
    (1, 2, 0.5, 3),     # the lesson machine
    (2, 1, 3, 5),       # bigger input
    (1, -1, 2, 4),      # negative weight a
    (3, 0.5, -2, 1),    # negative weight b
    (-2, 1.5, 0.7, 0),  # negative input
    (0.5, 4, 0.25, 0.5),  # already at the target
    (10, 0.1, 0.1, 7),  # tiny weights, big input
]
for m in machines:
    back = backward(*m)
    check(f"machine {m}: a's blame, backward = nudge",
          close(back["a"], nudge_a(*m, nudge=1e-6)))
    check(f"machine {m}: b's blame, backward = nudge",
          close(back["b"], nudge_b(*m, nudge=1e-6)))

# --- at the target, every blame goes silent ---
silent = backward(0.5, 4, 0.25, 0.5)
check("on target: all blames are 0", all(v == 0 for v in silent.values()))

# --- freeze rule: nudging b leaves the middle alone ---
check("nudge b: middle stays 2", forward(1, 2, 0.5 + 0.1)[0] == forward(1, 2, 0.5)[0])
check("nudge a: middle moves", forward(1, 2 + 0.1, 0.5)[0] != forward(1, 2, 0.5)[0])

# --- the swap: weight's blame uses the number IN, messenger uses the weight ---
zero_middle = backward(1, 0, 0.5, 3)   # a = 0 -> middle 0
check("middle 0 -> b's blame 0 (nothing flows through b)", zero_middle["b"] == 0)
check("middle 0 -> messenger still loud (b = 0.5)", zero_middle["middle"] != 0)

zero_b = backward(1, 2, 0, 3)          # b = 0 -> blocks the messenger
check("b = 0 -> messenger is 0", zero_b["middle"] == 0)
check("b = 0 -> a's blame is 0 (blame can't get past b)", zero_b["a"] == 0)
check("b = 0 -> b's own blame is NOT 0", zero_b["b"] != 0)

zero_input = backward(0, 2, 0.5, 3)
check("input 0 -> a's blame 0", zero_input["a"] == 0)

# --- doubling b doubles the messenger (same end blame) ---
m1 = backward(1, 2, 0.5, 3)["middle"] / blame_at_end(1, 3)
m2 = backward(1, 1, 1.0, 3)["middle"] / blame_at_end(1, 3)  # also output 1
check("messenger / end blame = b (0.5 vs 1.0)", m1 == 0.5 and m2 == 1.0)

print(f"\n{passed} passed, {failed} failed")
