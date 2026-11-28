"""Day 8: a nudge measures the AVERAGE slope, the derivative measures it AT a point.

The machine:

    x --[x w1]--> h --[x w2]--> y      target
    2      3      6     0.5     3        1

    loss = (y - target)^2 = (3 - 1)^2 = 4

Blame on y, two ways:
    derivative at y = 3:   2 x (3 - 1) = 4        (use the CURRENT y, not the nudged one)
    nudge y by 0.01:       (4.0401 - 4) / 0.01 = 4.01

Why 4.01? The slope is 4 where we start (y = 3) and 4.02 where we land (y = 3.01).
The nudge ratio is the average of the two: 4.01.
"""


def forward(x, w1, w2):
    """Left to right. Returns the hidden number h and the output y."""
    h = x * w1
    y = h * w2
    return h, y


def loss_of(y, target):
    """The loss: (y - target) squared."""
    return (y - target) ** 2


def slope_at(y, target):
    """Derivative of (y - target)^2 at THIS y: 2 x (y - target)."""
    return 2 * (y - target)


def nudge_ratio(y, target, nudge):
    """Push y up by a little. How much does the loss move, per unit of push?"""
    return (loss_of(y + nudge, target) - loss_of(y, target)) / nudge


def backward(x, w1, w2, target):
    """Right to left. Every blame, built from the one on its right."""
    h, y = forward(x, w1, w2)
    y_blame = slope_at(y, target)   # 4
    w2_blame = y_blame * h          # 4 x 6   = 24  (w2 sees h)
    h_blame = y_blame * w2          # 4 x 0.5 = 2   (h sees w2)
    w1_blame = h_blame * x          # 2 x 2   = 4   (w1 sees x)
    return {"y": y_blame, "w2": w2_blame, "h": h_blame, "w1": w1_blame}


def nudge_weight(x, w1, w2, target, which, nudge):
    """Nudge one weight, freeze the other, measure how the loss moves."""
    _, before = forward(x, w1, w2)
    if which == "w1":
        _, after = forward(x, w1 + nudge, w2)
    else:
        _, after = forward(x, w1, w2 + nudge)
    return (loss_of(after, target) - loss_of(before, target)) / nudge


if __name__ == "__main__":
    x, w1, w2, target = 2, 3, 0.5, 1
    h, y = forward(x, w1, w2)

    print("1) Forward:")
    print(f"   x {x} --[x {w1}]--> h {h} --[x {w2}]--> y {y}")
    print(f"   loss = ({y} - {target})^2 = {loss_of(y, target)}")

    print("\n2) Blame on y: nudge ratio vs the slope at start and at landing")
    print("   nudge    slope at start   slope at landing   average   nudge ratio")
    for n in [0.1, 0.01, 0.001]:
        start = slope_at(y, target)
        land = slope_at(y + n, target)
        avg = (start + land) / 2
        print(f"   {n:<6}   {start:14.3f}   {land:16.3f}   {avg:7.4f}"
              f"   {nudge_ratio(y, target, n):11.4f}")
    print("   The nudge ratio is always the average. As the nudge shrinks, all")
    print("   three squeeze toward 4. The derivative is that limit, at y = 3.")

    b = backward(x, w1, w2, target)
    print("\n3) Walk back. In y = w2 x h, each partner's slope is the OTHER's value:")
    print(f"   w2's blame = y's blame {b['y']} x h {h}   = {b['w2']}")
    print(f"   h's blame  = y's blame {b['y']} x w2 {w2} = {b['h']}")
    print(f"   w1's blame = h's blame {b['h']} x x {x}   = {b['w1']}")

    print("\n4) Check with nudges (freeze the other weight):")
    print("   nudge     w1's blame   w2's blame")
    for n in [0.01, 0.001, 0.0001]:
        print(f"   {n:<7}   {nudge_weight(x, w1, w2, target, 'w1', n):10.4f}"
              f"   {nudge_weight(x, w1, w2, target, 'w2', n):10.4f}")
    print(f"   backward  {b['w1']:10.4f}   {b['w2']:10.4f}")
