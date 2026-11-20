# tiny-nas

Neural Architecture Search, worked out with small examples. One idea per day.

I'm learning from MIT EfficientML Lecture 7 (Neural Architecture Search, Part I).
This repo is my notes in code form, written for someone new to the topic.

The idea in one line: instead of a person designing a neural network by hand,
let a computer search for a good design.

## Day 1: a real, tiny search, and the biggest model doesn't win

Pruning and quantization shrink a model someone already designed. NAS picks
the design itself.

```
quantization   →  same floor plan, cheaper bricks
pruning        →  same floor plan, remove some bricks
NAS            →  pick the floor plan: how many layers, how wide, what type
```

**Why a computer?** The choices multiply. With 6 layers and 9 options each,
there are 9⁶ = 531,441 designs. At 1 hour of training each, that's 60 years.

**Why not just training?** Training learns the weights *inside* a fixed
floor plan. It can't add a layer. So NAS is a loop around training:

```
NAS (outer loop):    pick a floor plan
   training (inner): learn its weights
   score it:         accuracy + size
   repeat
```

### The search

Task: read handwritten digits (0-9) from 8×8 grayscale pictures (1,797 of
them, built into scikit-learn). Each picture becomes 64 numbers.

```
64 pixels → [hidden layers] → 10 scores → pick the highest
                 ↑
     NAS chooses only this part: 1-3 layers, each 4, 16 or 64 wide
     3 + 9 + 27 = 39 floor plans
```

Each plan trains 3 times (to average out luck) and is scored on 450
pictures it never saw.

![trade-off chart](trade_off.png)

```
plan        weights   accuracy
4               296     89.3%
16            1,184     96.8%   ← the knee: small, nearly as good as the best
64-64         8,832     97.6%   ← best
64-64-64     12,928     97.3%   ← biggest, NOT the best
4-64-4          808     61.1%   ← worst
```

There's no single "best model". There's a best model **per budget**:
phone (≤ 1,000 weights) gets `4-16` at 89.8%, laptop (≤ 3,000) gets `16-64`
at 97.2%, no limit gets `64-64` at 97.6%.

Honest note: 0.3 points is about 1 picture out of 450. At the top, the big
plans are basically tied.

### Two stylized facts from our search

Patterns that held in our data, not laws:

1. **A pipe too narrow anywhere caps the whole model.** Every plan with a
   width-4 layer (best 94.7%) lost to every plan without one (worst 95.9%).
   Squeeze the picture into 4 numbers and the detail is gone. A wide layer
   after it can't bring it back. Once every layer is wide enough (16 vs 64),
   extra width barely helps.

2. **Deeper didn't help here.** Same width, 1 → 2 → 3 layers: flat or worse.
   This is task-specific. Deeper models do win on harder problems. That's why
   NAS exists: the rules change from task to task, so you search.

The model here is fully connected: it sees 64 numbers, not a picture, and
doesn't know which pixels are neighbors. Convolution builds that in. Layer
type is another choice NAS can make.

Try it: `python nas_tiny.py` (about 4 minutes). Tests: `python test_nas_tiny.py`.
