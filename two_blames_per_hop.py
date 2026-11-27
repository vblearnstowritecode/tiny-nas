"""Day 7: each hop back makes two blames.

The machine (two weights, a and b):

    input --[x a]--> middle --[x b]--> output      target
      1       2        2       0.5       1           3

    error = (target - output)^2 = (3 - 1)^2 = 4

Walking back, each hop takes the blame on its right and makes TWO numbers:

    weight's blame  = blame on the right x number flowing IN   (to nudge it)
    middle's blame  = blame on the right x the weight          (the messenger)

Every blame answers one question:
    "Nudge this one thing a little, freeze the rest. How much does the ERROR move?"
"""


def forward(input_value, a, b):
    """Walk 1, left to right. Returns the middle and output (we SAVE them)."""
    middle = input_value * a
    output = middle * b
    return middle, output


def error_of(output, target):
    """The error: (target - output) squared."""
    return (target - output) ** 2


def blame_at_end(output, target):
    """Chain rule on (target - output)^2: 2 x (target - output) x (-1).

    Example: target 3, output 1 -> 2 x 2 x (-1) = -4
    """
    return -2 * (target - output)


def backward(input_value, a, b, target):
    """Walk 2, right to left. Returns every blame, using the saved middle."""
    middle, output = forward(input_value, a, b)

    output_blame = blame_at_end(output, target)   # -4

    # hop 1: across weight b
    b_blame = output_blame * middle               # -4 x 2   = -8
    middle_blame = output_blame * b               # -4 x 0.5 = -2  (messenger)

    # hop 2: across weight a
    a_blame = middle_blame * input_value          # -2 x 1   = -2

    return {"output": output_blame, "b": b_blame,
            "middle": middle_blame, "a": a_blame}


def nudge_a(input_value, a, b, target, nudge):
    """Nudge a, freeze b. The middle and output ripple. Watch the error."""
    _, before = forward(input_value, a, b)
    _, after = forward(input_value, a + nudge, b)
    return (error_of(after, target) - error_of(before, target)) / nudge


def nudge_b(input_value, a, b, target, nudge):
    """Nudge b, freeze a (so the middle stays put). Watch the error."""
    _, before = forward(input_value, a, b)
    _, after = forward(input_value, a, b + nudge)
    return (error_of(after, target) - error_of(before, target)) / nudge


def nudge_output(output, target, nudge):
    """Nudge the output itself. Watch the error."""
    return (error_of(output + nudge, target) - error_of(output, target)) / nudge


if __name__ == "__main__":
    x, a, b, target = 1, 2, 0.5, 3

    middle, output = forward(x, a, b)
    print("1) Forward walk (save the middle, we need it later):")
    print(f"   input {x} --[x {a}]--> middle {middle} --[x {b}]--> output {output}")
    print(f"   error = ({target} - {output})^2 = {error_of(output, target)}")

    blames = backward(x, a, b, target)
    print("\n2) Blame at the end: -2 x miss")
    print(f"   exact: {blames['output']}")
    for n in [0.1, 0.01, 0.001]:
        print(f"   nudge output by {n:<5}: {nudge_output(output, target, n):.4f}")

    print("\n3) Walk back. Each hop makes two blames:")
    rows = [
        ("b's blame     ", f"{blames['output']} x middle {middle}", blames["b"], "to nudge b"),
        ("middle's blame", f"{blames['output']} x b {b}", blames["middle"], "messenger"),
        ("a's blame     ", f"{blames['middle']} x input {x}", blames["a"], "to nudge a"),
    ]
    for name, sum_text, value, job in rows:
        print(f"   {name} = {sum_text:<15} = {value:<5}  ({job})")

    print("\n4) Check: freeze the others, nudge one weight, measure the error.")
    print("   nudge     a's blame   b's blame")
    for n in [0.1, 0.01, 0.001]:
        print(f"   {n:<6}   {nudge_a(x, a, b, target, n):9.4f}"
              f"   {nudge_b(x, a, b, target, n):9.4f}")
    print(f"   backward {blames['a']:9.4f}   {blames['b']:9.4f}")
    print("   Same answers. One walk back gave both.")
