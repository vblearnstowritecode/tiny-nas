"""Day 4: dividing bounces forever, small slices settle on a compromise.

Same machine as Day 3:  output = input x weight
But now there are TWO pictures, and they disagree:

  A: input 2, target 6  -> alone, A wants weight 6 / 2 = 3
  B: input 1, target 5  -> alone, B wants weight 5 / 1 = 5

No single weight makes both happy. What does training do?
"""

PICTURES = [
    (2, 6),  # A: (input, target)
    (1, 5),  # B: (input, target)
]

SLICE = 0.05  # the learning rate: how much of the gradient we use per step


def miss(weight, input, target):
    """How far short we are. Example: weight 1, A -> 6 - 2 x 1 = 4"""
    return target - input * weight


def gradient(weight, input, target):
    """Day 3's blame: -2 x miss x input.
    Example: weight 1, A -> -2 x 4 x 2 = -16
    """
    return -2 * miss(weight, input, target) * input


def total_loss(weight, pictures=PICTURES):
    """Add up every picture's squared miss.
    Example: weight 3.4 -> A (-0.8)^2 + B (1.6)^2 = 0.64 + 2.56 = 3.2
    """
    total = 0
    for input, target in pictures:
        total += miss(weight, input, target) ** 2
    return total


def divide_step(weight, input, target):
    """Avi's division rule: fix this picture's miss exactly, in one jump.
    Example: weight 1, A -> 1 + 4 / 2 = 3   (always lands on target / input)
    """
    return weight + miss(weight, input, target) / input


def slice_step(weight, input, target, slice_size=SLICE):
    """Gradient descent: a small tug, against the gradient.
    Example: weight 1, A -> 1 - 0.05 x (-16) = 1.8
    """
    return weight - slice_size * gradient(weight, input, target)


def train(step_rule, weight, rounds, pictures=PICTURES):
    """Show A, B, A, B, ... Each round = one look at every picture.
    Returns the weight after each picture.
    """
    history = []
    for _ in range(rounds):
        for input, target in pictures:
            weight = step_rule(weight, input, target)
            history.append(weight)
    return history


def batch_step(weight, pictures=PICTURES):
    """The Day 2 way (mini-batch): average the gradients of ALL pictures,
    then take ONE step.
    Example: weight 1 -> A says -16, B says -8 -> average -12
             -> 1 - 0.05 x (-12) = 1.6
    """
    total = 0
    for input, target in pictures:
        total += gradient(weight, input, target)
    average = total / len(pictures)
    return weight - SLICE * average


def train_batch(weight, steps, pictures=PICTURES):
    """Take many batch steps. Returns the weight after each step."""
    history = []
    for _ in range(steps):
        weight = batch_step(weight, pictures)
        history.append(weight)
    return history


if __name__ == "__main__":
    print("Two pictures that disagree: A wants weight 3, B wants weight 5.")
    print("Start at weight 1. Show A, B, A, B, ...\n")

    divided = train(divide_step, 1, 4)
    sliced = train(slice_step, 1, 40)

    print("1) Divide (jump to each picture's answer):")
    print("   " + ", ".join(f"{w:g}" for w in divided) + ", ...  never settles")

    print("\n2) Slice 0.05 (small tug from each picture):")
    print("   after   picture  weight")
    for i in [0, 1, 2, 3, 4, 5]:
        print(f"   {i + 1:5d}     {'AB'[i % 2]}     {sliced[i]:.3f}")
    print("     ...")
    for i in [76, 77, 78, 79]:
        print(f"   {i + 1:5d}     {'AB'[i % 2]}     {sliced[i]:.3f}")
    print("   It settles into a tiny wobble: A tugs it down, B tugs it up.")

    batched = train_batch(1, 60)
    final = batched[-1]
    print("\n3) Batch (average both gradients, then one step):")
    print("   step 1: " + f"{batched[0]:.3f}" + "   step 60: " + f"{final:.3f}")

    print(f"\n4) Why there? At weight {final:.1f}, the two tugs cancel:")
    for name, (input, target) in zip("AB", PICTURES):
        m = miss(final, input, target)
        g = gradient(final, input, target)
        print(f"   {name}: miss {m:+.1f}  gradient -2 x {m:+.1f} x {input} = {g:+.1f}")
