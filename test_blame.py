"""Tests for blame.py. Each test prints PASS or FAIL."""

from blame import blame, loss, measured_blame, one_step, train

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


def close(a, b, tol=0.01):
    return abs(a - b) < tol


# --- loss: the bowl ---
check("weight 1: loss 16 (miss 4)", loss(1) == 16)
check("weight 2.5: loss 1 (miss 1)", loss(2.5) == 1)
check("weight 3: loss 0 (perfect)", loss(3) == 0)
check("overshoot counts as bad: weight 4 has loss 4, same as weight 2",
      loss(4) == loss(2) == 4)
check("loss is never negative", all(loss(w / 10) >= 0 for w in range(-100, 100)))

# --- blame: the formula ---
check("weight 1: blame -16 (= -2 x 4 x 2)", blame(1) == -16)
check("weight 2.5: blame -4", blame(2.5) == -4)
check("at the bottom (weight 3): blame 0", blame(3) == 0)
check("too big (weight 4): blame +8, says turn DOWN", blame(4) == 8)
check("far away = steeper: |blame at 1| is 4x |blame at 2.5|",
      blame(1) == 4 * blame(2.5))

# --- blame: formula matches a hand measurement ---
for w in [-5, 0, 1, 2.5, 3, 4, 10]:
    check(f"weight {w}: formula matches the nudge measurement",
          close(blame(w), measured_blame(w)))

# --- one step ---
check("weight 1, slice 0.05 -> 1.8 (Avi's worked step)",
      close(one_step(1, 0.05), 1.8, 1e-9))
check("a step from below moves the weight UP", one_step(1, 0.05) > 1)
check("a step from above moves the weight DOWN", one_step(5, 0.05) < 5)
check("at the bottom, a step does nothing", one_step(3, 0.05) == 3)
check("one small step lowers the loss", loss(one_step(1, 0.05)) < loss(1))

# --- training ---
history = train(1, 0.05, 30)
check("30 small steps reach weight 3", close(history[-1], 3, 0.001))
check("loss falls on every step",
      all(loss(b) < loss(a) for a, b in zip(history, history[1:])))
steps = [b - a for a, b in zip(history, history[1:])]
check("steps shrink as we get close (the bowl flattens)",
      all(s2 < s1 for s1, s2 in zip(steps, steps[1:])))
check("starting above (weight 10) also reaches 3",
      close(train(10, 0.05, 40)[-1], 3, 0.001))

# --- the problem a small slice fixes ---
big = train(1, 1, 3)
check("slice 1 (all the blame): 1 -> 17 -> -95 -> 689", big == [1, 17, -95, 689])
check("slice 1: loss gets WORSE each step",
      all(loss(b) > loss(a) for a, b in zip(big, big[1:])))

print(f"\n{passed} passed, {failed} failed")
