"""Day 3: what "blame" is, on a machine with one weight.

The machine:  output = input x weight
We want output = target. How should we turn the weight?

Example: input 2, target 6, weight 1
  output = 2 x 1 = 2,  miss = 6 - 2 = 4,  loss = 4 x 4 = 16
"""

INPUT = 2
TARGET = 6


def loss(weight):
    """How wrong we are: (target - output) squared. Always 0 or more."""
    miss = TARGET - INPUT * weight
    return miss * miss


def blame(weight):
    """Derivative of the loss with respect to the weight (the "gradient").

    loss = (target - input x weight)^2
    chain rule: blame = 2 x miss x (-input) = -2 x miss x input
    Example: weight 1 -> miss 4 -> blame = -2 x 4 x 2 = -16
    Negative blame means: raise the weight and the loss falls.
    """
    miss = TARGET - INPUT * weight
    return -2 * miss * INPUT


def measured_blame(weight, nudge=0.001):
    """Same thing, measured by hand: nudge the weight, see how the loss moves."""
    return (loss(weight + nudge) - loss(weight)) / nudge


def one_step(weight, slice_size):
    """Move a little, against the blame (downhill).
    Example: weight 1, blame -16, slice 0.05 -> 1 - 0.05 x (-16) = 1.8
    """
    return weight - slice_size * blame(weight)


def train(weight, slice_size, steps):
    """Take many small steps. Returns the weight after each step."""
    history = [weight]
    for _ in range(steps):
        weight = one_step(weight, slice_size)
        history.append(weight)
    return history


if __name__ == "__main__":
    print("1) Does the formula match a hand measurement?")
    for w in [1, 2.5, 3, 4]:
        print(f"   weight {w}: formula {blame(w):7.2f}   measured {measured_blame(w):7.2f}")

    print("\n2) Training with a small slice (0.05). The right weight is 3.")
    print("   step  weight   miss    blame")
    for step, w in enumerate(train(1, 0.05, 12)):
        miss = TARGET - INPUT * w
        print(f"   {step:4d}  {w:6.3f}  {miss:5.2f}  {blame(w):7.2f}")

    print("\n3) Using ALL the blame (slice 1) jumps way past 3:")
    for step, w in enumerate(train(1, 1, 3)):
        print(f"   step {step}: weight {w:g}")
