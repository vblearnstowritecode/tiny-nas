"""Day 5: where "new weight = old - learning rate x blame" comes from.

Same machine as Days 3-4:  output = input x weight,  input 2, target 6.
We stand at weight 1:  loss 16, blame (slope) -16.

The idea in three lines:
  1. The slope gives a straight-line guess of the loss.  Good nearby, bad far away.
  2. Taylor adds the bend (the curve the line can't see).  Bottom of that bowl:
         step = - blame / bend
  3. Gradient descent can't afford the bend, so it uses a fixed guess:
         step = - learning rate x blame        (learning rate stands in for 1 / bend)
"""

INPUT = 2
TARGET = 6
START = 1


def loss(weight):
    """(target - output) squared.  Example: weight 1 -> (6 - 2)^2 = 16."""
    miss = TARGET - INPUT * weight
    return miss * miss


def blame(weight):
    """Slope of the loss.  Example: weight 1 -> -2 x 4 x 2 = -16."""
    miss = TARGET - INPUT * weight
    return -2 * miss * INPUT


# How fast the slope itself changes (the second derivative).
# loss = (6 - 2w)^2, slope = -4 x (6 - 2w), so the slope grows by 8 for every 1 of weight.
# It is the same everywhere because our loss is a perfect bowl.
BEND = 2 * INPUT * INPUT   # = 8


def line_guess(step):
    """Straight-line guess of the loss after moving by `step`.
    Example: step 0.1 -> 16 + (-16) x 0.1 = 14.4
    """
    return loss(START) + blame(START) * step


def taylor_guess(step):
    """Line plus the bend.  For a perfect bowl this is exact.
    Example: step 10 -> 16 - 160 + 0.5 x 8 x 100 = 256
    """
    return loss(START) + blame(START) * step + 0.5 * BEND * step * step


def guessed_bowl(step, learning_rate):
    """Line plus a GUESSED bend of 1 / learning_rate (what gradient descent uses)."""
    return loss(START) + blame(START) * step + 0.5 * (1 / learning_rate) * step * step


def bottom_of_guessed_bowl(learning_rate):
    """Set the slope of guessed_bowl to 0:  blame + step / learning_rate = 0.
    So  step = - learning_rate x blame.   Example: 0.05 -> -0.05 x -16 = 0.8
    """
    return -learning_rate * blame(START)


def bottom_by_search(learning_rate):
    """Same answer found the slow way: try many steps, keep the lowest."""
    steps = [i / 1000 for i in range(-20000, 20001)]   # -20.000 ... 20.000
    return min(steps, key=lambda s: guessed_bowl(s, learning_rate))


if __name__ == "__main__":
    print(f"Standing at weight {START}: loss {loss(START)}, slope {blame(START)}\n")

    print("1) The straight line is good nearby, and lies far away:")
    print("    step   line says   truth   line + bend")
    for step in [0.1, 0.5, 1, 2, 5, 10]:
        print(f"   {step:5}   {line_guess(step):8.2f}  {loss(START + step):7.2f}"
              f"   {taylor_guess(step):8.2f}")
    print("   A line has no bottom, so it can't pick a step. The bend gives it one.\n")

    print("2) Bottom of the true bowl:  step = -blame / bend")
    true_step = -blame(START) / BEND
    print(f"   step = 16 / {BEND} = {true_step:g}  ->  weight {START + true_step:g}"
          f"  (loss {loss(START + true_step):g}, done in one step)\n")

    print("3) Gradient descent guesses the bend as 1 / learning rate:")
    print("   rate    guessed bend   step   new weight   new loss")
    for rate in [0.05, 1 / 8, 1]:
        step = bottom_of_guessed_bowl(rate)
        print(f"   {rate:5.3f}   {1 / rate:10.1f}   {step:5.2f}   {START + step:9.2f}"
              f"   {loss(START + step):8.2f}")
    print("   0.05 guesses too steep a bowl: short, safe step (Day 3's 1.8).")
    print("   1/8 guesses the true bend: lands on 3 at once.")
    print("   1 guesses too flat a bowl: overshoots to 17 (Day 3's blow-up).")
