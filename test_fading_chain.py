"""Tests for fading_chain.py. Each test prints PASS or FAIL."""

from fading_chain import (backward_hops, blame_backward, blame_forward,
                          forward_hops, run_chain)

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


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


# --- the worked example from the lesson ---
check("2 links of 0.5 -> [0.25, 0.5, 1]",
      blame_backward([0.5, 0.5]) == [0.25, 0.5, 1])
check("3 links of 0.5 -> start blame 0.125",
      blame_backward([0.5] * 3)[0] == 0.125)
check("one spot per link plus the end",
      len(blame_backward([0.5] * 10)) == 11)
check("the end keeps the end blame", blame_backward([0.5] * 10)[-1] == 1)

# --- fading ---
fade = blame_backward([0.5] * 10)
check("10 links of 0.5 -> start blame about 0.001", close(fade[0], 1 / 1024))
check("fading: blame shrinks at every step back",
      all(earlier < later for earlier, later in zip(fade, fade[1:])))
check("50 links of 0.5 -> start blame below one in a million billion",
      blame_backward([0.5] * 50)[0] < 1e-15)
check("links of 0.9 fade slower than links of 0.5",
      blame_backward([0.9] * 10)[0] > blame_backward([0.5] * 10)[0])
check("even 0.9 fades a lot over 100 links (below 0.0001)",
      blame_backward([0.9] * 100)[0] < 0.0001)

# --- exploding ---
boom = blame_backward([2] * 10)
check("10 links of 2 -> start blame 1024", boom[0] == 1024)
check("exploding: blame grows at every step back",
      all(earlier > later for earlier, later in zip(boom, boom[1:])))

# --- steady ---
check("10 links of 1 -> start blame stays 1",
      all(b == 1 for b in blame_backward([1] * 10)))
check("0.5 then 2 cancel out -> start blame 1",
      blame_backward([0.5, 2, 0.5, 2])[0] == 1)

# --- the starting blame doesn't change the fade ---
check("end blame 8, 3 links of 0.5 -> 8, 4, 2, 1 (lesson example)",
      blame_backward([0.5] * 3, end_blame=8) == [1, 2, 4, 8])
check("end blame 0.1, 3 links of 0.5 -> 0.0125 at the start",
      close(blame_backward([0.5] * 3, end_blame=0.1)[0], 0.0125))
check("negative end blame keeps its sign all the way back",
      all(b < 0 for b in blame_backward([0.5] * 5, end_blame=-4)))
check("a negative link flips the sign",
      blame_backward([-1], end_blame=3)[0] == -3)

# --- edge cases ---
check("no links: blame is just the end blame", blame_backward([]) == [1])
check("a zero link kills all blame before it",
      blame_backward([0.5, 0, 0.5])[:2] == [0, 0])
check("blame after a zero link is untouched",
      blame_backward([0.5, 0, 0.5])[2] == 0.5)

# --- forward run ---
check("run 3 links of 0.5 from 1 -> 0.125", run_chain([0.5] * 3, 1) == 0.125)
check("run from spot 2 only uses the last link",
      run_chain([0.5, 0.5, 0.5], 1, from_spot=2) == 0.5)

# --- forward nudge gives the same blame as backward ---
for weights in ([1, 0.5, 0.5], [0.5] * 10, [2] * 10, [0.3, -1.5, 2, 0.8]):
    back = blame_backward(weights)
    same = all(close(blame_forward(weights, s), back[s], 1e-4)
               for s in range(len(weights)))
    check(f"forward nudge matches backward walk for {weights}", same)

# --- hop counts ---
check("3 links: forward 6 hops, backward 3",
      forward_hops(3) == 6 and backward_hops(3) == 3)
check("100 links: forward 5050 hops, backward 100",
      forward_hops(100) == 5050 and backward_hops(100) == 100)
check("1 link: both need 1 hop", forward_hops(1) == backward_hops(1) == 1)
check("the gap grows with depth",
      all(forward_hops(n) / backward_hops(n) < forward_hops(n + 1) / backward_hops(n + 1)
          for n in range(1, 50)))

print(f"\n{passed} passed, {failed} failed")
