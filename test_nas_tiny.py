"""Tests for nas_tiny.py. Each test prints PASS or FAIL."""

from nas_tiny import (all_floor_plans, best_trade_offs, count_weights,
                      load_data, train_and_score, winner_for_budget)

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


# --- the search space: how many floor plans? ---

plans = all_floor_plans()
check("Rule B gives 3 + 9 + 27 = 39 plans", len(plans) == 39)
check("no plan appears twice", len(set(plans)) == 39)
check("1 layer: 3 plans", sum(1 for p in plans if len(p) == 1) == 3)
check("2 layers: 9 plans", sum(1 for p in plans if len(p) == 2) == 9)
check("3 layers: 27 plans", sum(1 for p in plans if len(p) == 3) == 27)
check("every width is 4, 16 or 64", all(w in (4, 16, 64) for p in plans for w in p))
check("mixed widths are allowed, e.g. 64-4", (64, 4) in plans)
check("order matters: 4-64 and 64-4 are different plans", (4, 64) in plans and (64, 4) in plans)
check("toy: widths [4,16], depths [1,2] -> 6 plans",
      len(all_floor_plans([4, 16], [1, 2])) == 6)
check("one width, 3 depths -> 3 plans", len(all_floor_plans([8], [1, 2, 3])) == 3)
check("6 layers x 9 options = 9^6 = 531,441 plans",
      len(all_floor_plans(list(range(9)), [6])) == 531_441)

# --- size: counting weights ---

check("16 -> 64x16 + 16x10 = 1,184", count_weights((16,)) == 1184)
check("4 -> 64x4 + 4x10 = 296", count_weights((4,)) == 296)
check("64-64 -> 4096 + 4096 + 640 = 8,832", count_weights((64, 64)) == 8832)
check("smallest plan is 1 layer of 4 (296), not 4-4-4 (328)",
      min(count_weights(p) for p in plans) == count_weights((4,)) == 296
      and count_weights((4, 4, 4)) == 328)
check("wide first layer costs most: 64-4-4 (4,408) vs 4-64-4 (808)",
      count_weights((64, 4, 4)) == 4408 and count_weights((4, 64, 4)) == 808)
check("64-64-64 is the biggest plan (12,928)",
      max(count_weights(p) for p in plans) == count_weights((64, 64, 64)) == 12928)
check("order changes size: 4-64 (1,152) != 64-4 (4,392)",
      count_weights((4, 64)) == 64*4 + 4*64 + 64*10 and
      count_weights((64, 4)) == 64*64 + 64*4 + 4*10 and
      count_weights((4, 64)) != count_weights((64, 4)))
check("no hidden layer at all -> 64x10 = 640", count_weights(()) == 640)

# --- best trade-offs ---

A = ("A", 100, 0.80)
B = ("B", 200, 0.70)
C = ("C", 300, 0.90)
check("toy: B dropped (A is smaller AND better)", best_trade_offs([A, B, C]) == [A, C])
check("input order doesn't matter", best_trade_offs([C, B, A]) == [A, C])
check("bigger but equal score is dropped",
      best_trade_offs([("X", 100, 0.9), ("Y", 500, 0.9)]) == [("X", 100, 0.9)])
check("same size: only the better one kept",
      best_trade_offs([("X", 100, 0.7), ("Y", 100, 0.9)]) == [("Y", 100, 0.9)])
check("strictly rising: all kept",
      len(best_trade_offs([("a", 1, 0.1), ("b", 2, 0.2), ("c", 3, 0.3)])) == 3)
check("bigger always worse: only the smallest kept",
      best_trade_offs([("a", 1, 0.9), ("b", 2, 0.5), ("c", 3, 0.1)]) == [("a", 1, 0.9)])
check("empty list -> empty list", best_trade_offs([]) == [])

# --- winner per budget ---

check("budget 250 -> A (only A fits)", winner_for_budget([A, B, C], 250) == A)
check("budget 1000 -> C (most accurate)", winner_for_budget([A, B, C], 1000) == C)
check("budget exactly 300 -> C fits", winner_for_budget([A, B, C], 300) == C)
check("budget 50 -> None (nothing fits)", winner_for_budget([A, B, C], 50) is None)
check("tie in accuracy -> smaller plan wins",
      winner_for_budget([("big", 500, 0.9), ("small", 100, 0.9)], 1000)[0] == "small")

# --- real data + one real training (slow part, a few seconds) ---

x_train, x_test, y_train, y_test = load_data()
check("1,797 pictures in total", len(x_train) + len(x_test) == 1797)
check("test set is ~25% (450)", len(x_test) == 450)
check("each picture is 64 numbers", x_train.shape[1] == 64)
check("pixels scaled to 0..1", x_train.min() >= 0 and x_train.max() <= 1)
check("all 10 digits appear in the test set", set(y_test) == set(range(10)))
data = (x_train, x_test, y_train, y_test)
acc_small = train_and_score((4,), data, seeds=[0])
acc_big = train_and_score((64,), data, seeds=[0])
check(f"width 4 beats a random guess of 10% ({acc_small:.1%})", acc_small > 0.5)
check(f"width 64 reads most digits ({acc_big:.1%})", acc_big > 0.9)
check("same seed -> same score (repeatable)",
      train_and_score((16,), data, seeds=[0]) == train_and_score((16,), data, seeds=[0]))

print(f"\n{passed} passed, {failed} failed")
