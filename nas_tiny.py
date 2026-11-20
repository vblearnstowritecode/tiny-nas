"""
Day 1: a real, tiny Neural Architecture Search (NAS).

The task: read handwritten digits (0-9) from 8x8 grayscale pictures.
The question: which floor plan (how many layers, how wide) is best?

    64 pixels -> [hidden layers] -> 10 scores (one per digit)
                       ^
           NAS chooses only this middle part

Outer loop (NAS):      try every floor plan
Inner loop (training): learn the weights of one floor plan
"""

import warnings
from itertools import product

from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

INPUTS = 64          # 8 x 8 pixels
OUTPUTS = 10         # digits 0..9
WIDTHS = [4, 16, 64] # each hidden layer picks one of these
DEPTHS = [1, 2, 3]   # number of hidden layers
SEEDS = [0, 1, 2]    # train each plan 3 times, average out the luck


def all_floor_plans(widths=WIDTHS, depths=DEPTHS):
    """Every plan where each layer picks its own width (Rule B).

    Toy: widths [4, 16], depths [1, 2]
         -> (4,) (16,) (4,4) (4,16) (16,4) (16,16)  = 2 + 4 = 6 plans
    """
    plans = []
    for depth in depths:
        for plan in product(widths, repeat=depth):
            plans.append(plan)
    return plans


def count_weights(plan, inputs=INPUTS, outputs=OUTPUTS):
    """Size of a plan = number of weights (we skip the small biases).

    Toy: plan (16,) -> 64x16 + 16x10 = 1024 + 160 = 1,184
    """
    sizes = [inputs] + list(plan) + [outputs]
    total = 0
    for a, b in zip(sizes, sizes[1:]):
        total += a * b
    return total


def load_data():
    """1,797 digit pictures. 75% to train, 25% kept hidden for the test."""
    digits = load_digits()
    pixels = digits.data / 16.0   # shades 0..16 -> 0..1
    return train_test_split(pixels, digits.target, test_size=0.25, random_state=0)


def train_and_score(plan, data, seeds=SEEDS):
    """Inner loop: train this plan a few times, return its average test accuracy."""
    x_train, x_test, y_train, y_test = data
    scores = []
    for seed in seeds:
        model = MLPClassifier(hidden_layer_sizes=plan, max_iter=1000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))  # share of unseen pictures it got right
    return sum(scores) / len(scores)


def best_trade_offs(results):
    """Keep a plan only if no smaller-or-equal plan scores as high or higher.

    results: list of (plan, size, accuracy)
    Toy: A (size 100, 0.80), B (size 200, 0.70), C (size 300, 0.90)
         -> B is dropped: A is smaller AND more accurate.  Keep A, C.
    """
    by_size = sorted(results, key=lambda r: (r[1], -r[2]))
    kept = []
    best_so_far = -1.0
    for plan, size, accuracy in by_size:
        if accuracy > best_so_far:
            kept.append((plan, size, accuracy))
            best_so_far = accuracy
    return kept


def winner_for_budget(results, max_size):
    """Most accurate plan that fits in the budget. None if nothing fits."""
    fits = [r for r in results if r[1] <= max_size]
    if not fits:
        return None
    # highest accuracy; on a tie, the smaller plan wins
    return max(fits, key=lambda r: (r[2], -r[1]))


def save_chart(results, kept, path="trade_off.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([r[1] for r in results], [r[2] for r in results],
               color="#9aa5b1", label="all 39 floor plans")
    ax.plot([r[1] for r in kept], [r[2] for r in kept],
            color="#d1495b", marker="o", label="best trade-offs")
    ax.set_xscale("log")
    ax.set_xlabel("size (number of weights, log scale)")
    ax.set_ylabel("accuracy on unseen pictures")
    ax.set_title("Tiny NAS: 39 floor plans for reading digits")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plan_name(plan):
    return "-".join(str(w) for w in plan)


if __name__ == "__main__":
    data = load_data()
    plans = all_floor_plans()
    print(f"Searching {len(plans)} floor plans, {len(SEEDS)} trainings each...\n")

    # The outer loop: this IS the search.
    results = []
    for plan in plans:
        size = count_weights(plan)
        accuracy = train_and_score(plan, data)
        results.append((plan, size, accuracy))
        print(f"  {plan_name(plan):>10}   {size:>6,} weights   {accuracy:.1%}")

    kept = best_trade_offs(results)
    print("\nBest trade-offs (nothing smaller scores higher):")
    for plan, size, accuracy in kept:
        print(f"  {plan_name(plan):>10}   {size:>6,} weights   {accuracy:.1%}")

    print("\nWinner per budget:")
    for label, budget in [("phone", 1_000), ("laptop", 3_000), ("data center", 10**9)]:
        plan, size, accuracy = winner_for_budget(results, budget)
        limit = "no limit" if budget == 10**9 else f"<= {budget:,}"
        print(f"  {label:<12} ({limit:>8})   {plan_name(plan):>8}   {size:>6,} weights   {accuracy:.1%}")

    save_chart(results, kept)
    print("\nChart saved to trade_off.png")
