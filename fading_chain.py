"""Day 6: blame fades (or explodes) as it walks back through a chain.

The machine: a chain of one-weight links. Each link multiplies.

    start -> [x w1] -> [x w2] -> ... -> [x wN] -> end

Walking back, each link multiplies the blame by its own weight.
Why? A link does  out = w x in.  If "in" goes up by 1, "out" goes up by w.
So the error cares about "in" w times as much as it cares about "out".

Example: 3 links of 0.5, blame at the end = 1
    end 1  ->  0.5  ->  0.25  ->  0.125 at the start
"""


def blame_backward(weights, end_blame=1):
    """Walk back from the end once. Returns the blame at every spot.

    Spot 0 is the start, the last spot is the end.
    Example: weights [0.5, 0.5], end blame 1 -> [0.25, 0.5, 1]
    """
    blames = [end_blame]
    carried = end_blame
    for w in reversed(weights):  # last link first
        carried = carried * w
        blames.append(carried)
    blames.reverse()  # so spot 0 is the start again
    return blames


def run_chain(weights, start_value, from_spot=0):
    """Run the chain forward from one spot. Returns the end value."""
    value = start_value
    for w in weights[from_spot:]:
        value = value * w
    return value


def blame_forward(weights, spot, nudge=0.001):
    """Same blame, measured forward: bump one spot, rerun, watch the end.

    This needs a separate rerun for every spot.
    """
    before = run_chain(weights, 1, spot)
    after = run_chain(weights, 1 + nudge, spot)
    return (after - before) / nudge


def forward_hops(n_links):
    """Each spot walks to the end on its own: n + (n-1) + ... + 1."""
    return n_links * (n_links + 1) // 2


def backward_hops(n_links):
    """One walk back crosses each link once."""
    return n_links


if __name__ == "__main__":
    print("1) Ten links of 0.5. Blame at the end = 1. Walk back:")
    for spot, b in enumerate(blame_backward([0.5] * 10)):
        print(f"   spot {spot:2d}: blame {b:.4f}")
    print("   The first layer gets 1/1000th of the blame. It barely learns.")

    print("\n2) Ten links of 2. Walk back:")
    for spot, b in enumerate(blame_backward([2] * 10)):
        print(f"   spot {spot:2d}: blame {b:g}")
    print("   The first layer gets 1,024x the blame. Its step blows up.")

    print("\n3) Ten links of 1:")
    print(f"   start blame {blame_backward([1] * 10)[0]:g}  (stays steady)")

    print("\n4) Forward nudge vs backward walk (3 links: 1, 0.5, 0.5):")
    weights = [1, 0.5, 0.5]
    back = blame_backward(weights)
    for spot in range(len(weights)):
        print(f"   spot {spot}: backward {back[spot]:.3f}"
              f"   forward {blame_forward(weights, spot):.3f}")

    print("\n5) Hops needed to get every spot's blame:")
    print("   links   forward   backward")
    for n in [3, 10, 100, 1000]:
        print(f"   {n:5d}  {forward_hops(n):8d}  {backward_hops(n):8d}")
