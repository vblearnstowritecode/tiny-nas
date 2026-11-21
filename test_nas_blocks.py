"""Tests for nas_blocks.py. Each test prints PASS or FAIL."""

import torch

from nas_blocks import (BLOCKS, Block, build_model, count_macs, count_weights,
                        load_data, train_once)

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


# --- fully connected: MACs = weights (Part 1) ---
# A 1x1 glass at a single spot is the same as a fully connected layer.

fc_64_to_16 = [(1, 64, 16, 1)]
check("64 in -> 16 out: 1,024 weights", count_weights(fc_64_to_16) == 1024)
check("64 in -> 16 out at 1 spot: 1,024 MACs (MACs = weights)",
      count_macs(fc_64_to_16, spots=1) == 1024)

# --- convolution: weights get reused at every spot (Part 2) ---

one_glass = [(3, 1, 1, 1)]
check("one 3x3 glass: 9 weights", count_weights(one_glass) == 9)
check("one 3x3 glass on 8x8: 9 x 64 = 576 MACs", count_macs(one_glass) == 576)
check("one 3x3 glass on 16x16: still 9 weights", count_weights(one_glass) == 9)
check("one 3x3 glass on 16x16: 9 x 256 = 2,304 MACs",
      count_macs(one_glass, spots=256) == 2304)

sixteen_glasses = [(3, 1, 16, 1)]
check("16 glasses, 1 page in: 144 weights", count_weights(sixteen_glasses) == 144)
check("16 glasses, 1 page in: 9 x 64 x 16 = 9,216 MACs", count_macs(sixteen_glasses) == 9216)

normal = BLOCKS["normal 3x3"]
check("normal 3x3, 16 -> 16: 2,304 weights", count_weights(normal) == 2304)
check("normal 3x3, 16 -> 16: 147,456 MACs", count_macs(normal) == 147_456)
check("doubling pages in AND out -> 4x the work",
      count_macs([(3, 32, 32, 1)]) == 4 * count_macs([(3, 16, 16, 1)]))

# --- groups and depthwise (Part 3) ---

check("2 groups: 73,728 MACs (half)", count_macs([(3, 16, 16, 2)]) == 73_728)
check("32 pages, 4 groups -> 4x less work",
      count_macs([(3, 32, 32, 1)]) == 4 * count_macs([(3, 32, 32, 4)]))
check("16 groups (depthwise): 9,216 MACs", count_macs([(3, 16, 16, 16)]) == 9216)
check("1x1 mixer, 16 -> 16: 16,384 MACs", count_macs([(1, 16, 16, 1)]) == 16_384)
separable = BLOCKS["depthwise + 1x1"]
check("depthwise + 1x1: 9,216 + 16,384 = 25,600 MACs", count_macs(separable) == 25_600)
check("depthwise + 1x1: 400 weights", count_weights(separable) == 400)
check("depthwise + 1x1 is about 5.8x cheaper",
      round(count_macs(normal) / count_macs(separable), 1) == 5.8)

# --- bottleneck (Part 4), per spot, with Avi's 64 -> 16 -> 64 numbers ---

check("squeeze 64 -> 16: 1,024 MACs per spot", count_macs([(1, 64, 16, 1)], spots=1) == 1024)
check("think 3x3 at 16: 2,304 MACs per spot", count_macs([(3, 16, 16, 1)], spots=1) == 2304)
check("expand 16 -> 64: 1,024 MACs per spot", count_macs([(1, 16, 64, 1)], spots=1) == 1024)
check("one big 3x3 at 64: 36,864 MACs per spot", count_macs([(3, 64, 64, 1)], spots=1) == 36_864)
small_bottleneck = [(1, 64, 16, 1), (3, 16, 16, 1), (1, 16, 64, 1)]
check("bottleneck 64->16->64: 4,352 per spot, about 8.5x less",
      count_macs(small_bottleneck, spots=1) == 4352 and
      round(36_864 / 4352, 1) == 8.5)
resnet = [(1, 2048, 512, 1), (3, 512, 512, 1), (1, 512, 2048, 1)]
check("lecture's ResNet 2048->512->2048: also about 8.5x less",
      round(count_macs([(3, 2048, 2048, 1)]) / count_macs(resnet), 1) == 8.5)
check("our bottleneck 16->4->16: 17,408 MACs", count_macs(BLOCKS["bottleneck"]) == 17_408)
check("the bypass adds no weights",
      count_weights(BLOCKS["bottleneck + bypass"]) == count_weights(BLOCKS["bottleneck"]))

# --- our counts match what PyTorch actually builds ---

x = torch.rand(5, 16, 8, 8)
for name, layers in BLOCKS.items():
    block = Block(layers)
    pytorch_weights = sum(p.numel() for p in block.parameters())
    check(f"{name}: our weight count = PyTorch's ({pytorch_weights:,})",
          pytorch_weights == count_weights(layers))

    # Measure MACs from the real convolutions: weights x spots in each output.
    measured = 0
    out = x
    for conv in block.convs:
        out = conv(out)
        measured += conv.weight.numel() * out.shape[2] * out.shape[3]
    check(f"{name}: measured MACs = our count ({measured:,})", measured == count_macs(layers))

    check(f"{name}: 16 pages of 8x8 in -> 16 pages of 8x8 out",
          Block(layers)(x).shape == (5, 16, 8, 8))

# --- the bypass: lost detail comes back (Avi: "the block learns the delta") ---

def zeroed(block):
    """Set every weight to 0, so the block itself learns nothing."""
    with torch.no_grad():
        for conv in block.convs:
            conv.weight.zero_()
    return block

pages = torch.rand(2, 16, 8, 8)   # all values 0..1, so ReLU keeps them as they are
no_bypass = zeroed(Block(BLOCKS["bottleneck"], bypass=False))
with_bypass = zeroed(Block(BLOCKS["bottleneck"], bypass=True))
check("all-zero block, no bypass: every detail is lost (all zeros)",
      torch.all(no_bypass(pages) == 0).item())
check("all-zero block, with bypass: the input comes back unchanged",
      torch.equal(with_bypass(pages), pages))

# --- the full model ---

for name in BLOCKS:
    scores = build_model(name)(torch.rand(3, 1, 8, 8))
    check(f"{name}: model gives 10 scores per picture", scores.shape == (3, 10))

# --- real data + short real trainings (slow part, under a minute) ---

torch.set_num_threads(1)
data = load_data()
x_train, x_test, y_train, y_test = data
check("1,797 pictures in total", len(x_train) + len(x_test) == 1797)
check("test set is 450 pictures, same split as Day 1", len(x_test) == 450)
check("each picture is 1 page of 8x8", tuple(x_train.shape[1:]) == (1, 8, 8))
check("pixels scaled to 0..1", x_train.min() >= 0 and x_train.max() <= 1)
check("all 10 digits appear in the test set", set(y_test.tolist()) == set(range(10)))

acc = train_once("normal 3x3", data, seed=0, epochs=10)
check(f"10 passes: normal 3x3 beats a 10% random guess ({acc:.1%})", acc > 0.5)
acc_sep = train_once("depthwise + 1x1", data, seed=0, epochs=10)
check(f"10 passes: depthwise + 1x1 beats a 10% random guess ({acc_sep:.1%})", acc_sep > 0.5)
check("same seed -> same score (repeatable)",
      train_once("bottleneck", data, seed=0, epochs=3) ==
      train_once("bottleneck", data, seed=0, epochs=3))

print(f"\n{passed} passed, {failed} failed")
