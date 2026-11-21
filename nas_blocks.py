"""
Day 2: the price tag of a layer, and four ways to build the same block.

Every block takes 16 pages in and gives 16 pages out, on an 8x8 grid.
(A "page" is one channel: one 8x8 grid of numbers.)
We swap only the middle block and see what it costs and how well it reads digits.

    8x8 digit -> [3x3, 1 -> 16 pages] -> [ MIDDLE BLOCK ] -> average -> 10 scores
                                              ^
                                     we swap only this part

Two price tags per block:
    weights -> storage (how big is the file?)
    MACs    -> work    (how many multiply-adds per picture?)
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

PAGES = 16         # pages going in and out of every middle block
SPOTS = 8 * 8      # every glass visits all 64 spots of the 8x8 grid
SEEDS = [0, 1, 2, 3, 4]  # train each block 5 times, average out the luck
EPOCHS = 100             # passes over the training pictures
BATCH = 32               # pictures per small step

# Each layer is (glass size, pages in, pages out, groups).
#   groups = 1      -> every glass looks through all pages in
#   groups = pages  -> depthwise: each glass looks through one page only
BLOCKS = {
    "normal 3x3": [
        (3, 16, 16, 1),
    ],
    "depthwise + 1x1": [
        (3, 16, 16, 16),  # each page alone, looks at neighbors
        (1, 16, 16, 1),   # each spot alone, mixes all pages
    ],
    "bottleneck": [
        (1, 16, 4, 1),    # squeeze 16 -> 4 pages
        (3, 4, 4, 1),     # think at 4 pages
        (1, 4, 16, 1),    # expand back 4 -> 16 pages
    ],
}
# Same layers as "bottleneck", plus the bypass that adds the input back.
BLOCKS["bottleneck + bypass"] = BLOCKS["bottleneck"]
HAS_BYPASS = {"bottleneck + bypass"}


def count_weights(layers):
    """Storage: each glass is size x size x (pages it looks through).

    Toy: (3, 16, 16, 1) -> 16 glasses x 3x3x16 = 2,304 weights
    """
    total = 0
    for size, pages_in, pages_out, groups in layers:
        pages_each_glass_sees = pages_in // groups
        total += pages_out * size * size * pages_each_glass_sees
    return total


def count_macs(layers, spots=SPOTS):
    """Work: every weight gets used once at every spot.

    Toy: (3, 16, 16, 1) on 64 spots -> 2,304 weights x 64 = 147,456 MACs
    """
    return count_weights(layers) * spots


class Block(nn.Module):
    """Turns a list of layers into real PyTorch convolutions."""

    def __init__(self, layers, bypass=False):
        super().__init__()
        convs = []
        for size, pages_in, pages_out, groups in layers:
            convs.append(nn.Conv2d(pages_in, pages_out, kernel_size=size,
                                   padding=size // 2, groups=groups, bias=False))
        self.convs = nn.ModuleList(convs)
        self.bypass = bypass

    def forward(self, x):
        out = x
        for i, conv in enumerate(self.convs):
            out = conv(out)
            is_last = (i == len(self.convs) - 1)
            if not is_last:
                out = torch.relu(out)
        if self.bypass:
            out = out + x          # the original pages skip around the block
        return torch.relu(out)


def build_model(name):
    """stem (1 -> 16 pages) -> middle block -> average each page -> 10 scores"""
    return nn.Sequential(
        nn.Conv2d(1, PAGES, kernel_size=3, padding=1),
        nn.ReLU(),
        Block(BLOCKS[name], bypass=(name in HAS_BYPASS)),
        nn.AdaptiveAvgPool2d(1),   # each 8x8 page -> its average, 1 number
        nn.Flatten(),              # 16 numbers
        nn.Linear(PAGES, 10),      # 10 scores, one per digit
    )


def load_data():
    """Same 1,797 digits and same split as Day 1, shaped as 1-page 8x8 pictures."""
    digits = load_digits()
    pixels = digits.data / 16.0
    x_train, x_test, y_train, y_test = train_test_split(
        pixels, digits.target, test_size=0.25, random_state=0)

    def as_pictures(x):
        return torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 8, 8)

    return (as_pictures(x_train), as_pictures(x_test),
            torch.tensor(y_train), torch.tensor(y_test))


def train_once(name, data, seed, epochs=EPOCHS):
    """Train one model from scratch. Return its accuracy on unseen pictures."""
    x_train, x_test, y_train, y_test = data
    torch.manual_seed(seed)
    model = build_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        order = torch.randperm(len(x_train))
        for start in range(0, len(x_train), BATCH):
            batch = order[start:start + BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()     # gradients: which way to turn each knob
            optimizer.step()    # turn each knob a little

    with torch.no_grad():
        guesses = model(x_test).argmax(dim=1)
    return (guesses == y_test).float().mean().item()


def train_and_score(name, data, seeds=SEEDS):
    """Train a few times. Return (average, worst, best) accuracy."""
    scores = [train_once(name, data, seed) for seed in seeds]
    return sum(scores) / len(scores), min(scores), max(scores)


if __name__ == "__main__":
    torch.set_num_threads(1)   # small model: one thread is fast enough and repeatable
    data = load_data()
    normal_macs = count_macs(BLOCKS["normal 3x3"])
    print(f"Four middle blocks, {len(SEEDS)} trainings each (about 10 minutes)...\n")
    print(f"  {'block':<20} {'weights':>7} {'MACs':>9}   {'work':<10} accuracy (worst-best)")
    for name in BLOCKS:
        layers = BLOCKS[name]
        macs = count_macs(layers)
        work = "baseline" if macs == normal_macs else f"{normal_macs / macs:.1f}x less"
        average, worst, best = train_and_score(name, data)
        print(f"  {name:<20} {count_weights(layers):>7,} {macs:>9,}   {work:<10} "
              f"{average:.1%}  ({worst:.1%}-{best:.1%})", flush=True)
