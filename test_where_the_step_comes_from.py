"""Tests for where_the_step_comes_from.py. Each test prints PASS or FAIL."""

from where_the_step_comes_from import (
    BEND, START, blame, bottom_by_search, bottom_of_guessed_bowl,
    guessed_bowl, line_guess, loss, taylor_guess,
)
from blame import one_step

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


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


# --- starting point ---
check("start at weight 1: loss 16", loss(START) == 16)
check("start at weight 1: slope -16", blame(START) == -16)
check("bend is 8 (= 2 x input x input)", BEND == 8)

# --- bend really is how fast the slope changes ---
for w in [-3, 0, 1, 2.5, 3, 7]:
    check(f"weight {w}: slope grows by 8 per 1 of weight",
          close(blame(w + 1) - blame(w), BEND))

# --- the straight line: good nearby ---
check("step 0: line guess = loss now", line_guess(0) == 16)
check("step 0.1: line says 14.40, truth 14.44 (Avi's check)",
      close(line_guess(0.1), 14.4) and close(loss(START + 0.1), 14.44))
check("step 0.001: line almost exact",
      abs(line_guess(0.001) - loss(START + 0.001)) < 1e-5)
check("line's error grows with distance (0.1 < 1 < 10)",
      abs(line_guess(0.1) - loss(1.1)) < abs(line_guess(1) - loss(2))
      < abs(line_guess(10) - loss(11)))

# --- the straight line: lies far away (the problem) ---
check("step 10: line says -144, impossible for a squared number",
      close(line_guess(10), -144) and line_guess(10) < 0)
check("step 10: truth is 256, the loss went UP by 240",
      loss(START + 10) == 256 and loss(START + 10) - 16 == 240)
check("line has no bottom: a bigger step always looks better",
      line_guess(100) < line_guess(10) < line_guess(1))

# --- line + bend is exact for our bowl ---
for step in [-10, -2, -0.5, 0, 0.1, 1, 2, 5, 10]:
    check(f"step {step}: line + bend = true loss",
          close(taylor_guess(step), loss(START + step)))

# --- bottom of the true bowl = divide = one step to the answer ---
true_step = -blame(START) / BEND
check("true bowl's bottom: step 16 / 8 = 2", true_step == 2)
check("that lands on weight 3, loss 0", loss(START + true_step) == 0)
check("same as Day 4's divide: miss / input = 4 / 2 = 2", true_step == (6 - 2) / 2)
for start in [-4, 0, 2.5, 5, 10]:
    s = -blame(start) / BEND
    check(f"from weight {start}: blame / bend lands on 3 in one step",
          close(start + s, 3))

# --- guessed bowl: formula matches a brute-force search ---
for rate in [0.01, 0.05, 0.1, 1 / 8, 0.5, 1]:
    check(f"rate {rate:g}: formula bottom matches search",
          abs(bottom_of_guessed_bowl(rate) - bottom_by_search(rate)) <= 0.001)

# --- the formula = gradient descent ---
check("rate 0.05 -> step 0.8 -> weight 1.8 (Avi's worked step)",
      close(START + bottom_of_guessed_bowl(0.05), 1.8))
for rate in [0.01, 0.05, 0.2, 1]:
    check(f"rate {rate}: same weight as Day 3's one_step",
          close(START + bottom_of_guessed_bowl(rate), one_step(START, rate)))
check("step points opposite the slope (slope -16, step +)",
      bottom_of_guessed_bowl(0.05) > 0)
check("guessed bowl at step 0 = loss now (no move, no risk)",
      guessed_bowl(0, 0.05) == 16)
check("guessed bowl treats +2 and -2 the same risk (beyond the line)",
      close(guessed_bowl(2, 0.05) - line_guess(2), guessed_bowl(-2, 0.05) - line_guess(-2)))

# --- what the learning rate means ---
check("rate = 1/bend (1/8) lands exactly on 3",
      close(START + bottom_of_guessed_bowl(1 / BEND), 3))
check("rate too small (0.05): safe, loss falls but not to 0",
      0 < loss(START + bottom_of_guessed_bowl(0.05)) < 16)
check("rate too big (1): overshoots to 17, loss 784 (Day 3's blow-up)",
      close(START + bottom_of_guessed_bowl(1), 17) and loss(17) == 784)
check("any rate below 2/bend (0.25) lowers the loss",
      all(loss(START + bottom_of_guessed_bowl(r / 100)) < 16 for r in range(1, 25)))
check("rate 0.25 exactly: jumps to 5, same loss (bounces)",
      close(START + bottom_of_guessed_bowl(0.25), 5) and loss(5) == 16)
check("rate above 0.25 makes the loss worse",
      all(loss(START + bottom_of_guessed_bowl(r / 100)) > 16 for r in range(26, 101)))

print(f"\n{passed} passed, {failed} failed")
