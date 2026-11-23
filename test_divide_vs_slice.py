"""Tests for divide_vs_slice.py. Each test prints PASS or FAIL."""

from divide_vs_slice import (
    miss, gradient, total_loss, divide_step, slice_step,
    train, batch_step, train_batch,
)

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


# --- the building blocks (A = input 2, target 6; B = input 1, target 5) ---
check("A at weight 1: miss 4", miss(1, 2, 6) == 4)
check("B at weight 1: miss 4 too", miss(1, 1, 5) == 4)
check("A at weight 1: gradient -16 (same as Day 3)", gradient(1, 2, 6) == -16)
check("B at weight 1: gradient -8 (same miss, half the input)", gradient(1, 1, 5) == -8)
check("total loss at 3.4 is 0.64 + 2.56 = 3.2", close(total_loss(3.4), 3.2, 1e-9))

# --- the division rule ---
check("divide on A from weight 1 -> 3", divide_step(1, 2, 6) == 3)
check("divide on B from weight 1 -> 5", divide_step(1, 1, 5) == 5)
for start in [-10, 0, 1, 4, 100]:
    check(f"divide always lands on target / input (start {start})",
          close(divide_step(start, 2, 6), 3, 1e-9))
divided = train(divide_step, 1, 10)
check("divide bounces 3, 5, 3, 5 ... forever",
      divided == [3, 5] * 10)
check("divide never settles: last two weights are 2 apart",
      abs(divided[-1] - divided[-2]) == 2)
check("divide: a tiny input (0.01) makes a giant jump (1 -> 600)",
      close(divide_step(1, 0.01, 6), 600, 1e-6))
try:
    divide_step(1, 0, 6)
    check("divide: input 0 crashes (divide by zero)", False)
except ZeroDivisionError:
    check("divide: input 0 crashes (divide by zero)", True)

# --- the slice rule ---
check("slice on A from weight 1 -> 1.8 (Day 3's step)",
      close(slice_step(1, 2, 6), 1.8, 1e-9))
check("slice: a tiny input (0.01) gives a tiny step, not a jump",
      abs(slice_step(1, 0.01, 6) - 1) < 0.01)
check("slice: input 0 is safe (gradient 0, no move)", slice_step(1, 0, 6) == 1)
sliced = train(slice_step, 1, 200)
check("slice never reaches A's answer 3 or B's answer 5 late in training",
      all(3.2 < w < 3.5 for w in sliced[-20:]))
check("slice settles into a tiny wobble (under 0.2 wide)",
      max(sliced[-20:]) - min(sliced[-20:]) < 0.2)
check("slice: after A the weight dips (3.26), after B it rises (3.43)",
      close(sliced[-2], 3.261) and close(sliced[-1], 3.435))
check("smaller slice -> smaller wobble",
      (lambda h: max(h[-20:]) - min(h[-20:]))(
          train(lambda w, i, t: slice_step(w, i, t, 0.005), 1, 2000))
      < max(sliced[-20:]) - min(sliced[-20:]))
check("too big a slice (1, all the blame) blows up, like Day 3",
      abs(train(lambda w, i, t: slice_step(w, i, t, 1), 1, 10)[-1]) > 1000)

# --- batch: average both gradients, then one step ---
check("batch from weight 1: average of -16 and -8 is -12 -> 1.6",
      close(batch_step(1), 1.6, 1e-9))
check("batch settles at 3.4", close(train_batch(1, 200)[-1], 3.4, 1e-6))
for start in [-20, 0, 3, 5, 50]:
    check(f"batch reaches 3.4 from any start ({start})",
          close(train_batch(start, 300)[-1], 3.4, 1e-6))
check("batch: no wobble at the end",
      abs(train_batch(1, 300)[-1] - train_batch(1, 301)[-1]) < 1e-9)

# --- why 3.4: the tugs cancel, and it's the bottom of the total bowl ---
check("at 3.4, A's gradient is +3.2", close(gradient(3.4, 2, 6), 3.2, 1e-9))
check("at 3.4, B's gradient is -3.2", close(gradient(3.4, 1, 5), -3.2, 1e-9))
check("at 3.4, the two gradients cancel to 0",
      close(gradient(3.4, 2, 6) + gradient(3.4, 1, 5), 0, 1e-9))
best = min((w / 100 for w in range(0, 1000)), key=total_loss)
check("brute force: lowest total loss on a grid is at 3.4", close(best, 3.4, 1e-9))
check("Avi's bet: midpoint 4 has a higher loss than 3.4",
      total_loss(4) > total_loss(3.4))
check("the compromise leans toward A (the bigger input)",
      abs(3.4 - 3) < abs(3.4 - 5))

# --- edge cases: other sets of pictures ---
agree = [(2, 6), (1, 3)]  # both want weight 3
check("pictures that agree: divide stops bouncing (stays at 3)",
      train(divide_step, 1, 5, agree)[-4:] == [3, 3, 3, 3])
check("pictures that agree: slice settles exactly at 3",
      close(train(slice_step, 1, 200, agree)[-1], 3, 1e-6))
equal = [(1, 3), (1, 5)]  # same input -> equal say
check("equal inputs: compromise is the true midpoint 4",
      close(train_batch(1, 500, equal)[-1], 4, 1e-6))
flipped = [(1, 3), (2, 10)]  # B now has the bigger input
check("flip the inputs: compromise now leans toward B's answer 5",
      train_batch(1, 500, flipped)[-1] > 4)
negative = [(-2, -6), (1, 5)]  # A with a negative input still wants 3
check("negative input: same answer 3.4 (only input squared matters)",
      close(train_batch(1, 500, negative)[-1], 3.4, 1e-6))
check("one picture alone: slice and divide agree on 3",
      close(train(slice_step, 1, 100, [(2, 6)])[-1],
            train(divide_step, 1, 1, [(2, 6)])[-1], 1e-6))

print(f"\n{passed} passed, {failed} failed")
