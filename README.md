# Reinforcement Learning Logo Turtle

This package wraps Python's built-in Logo Turtle.

The problem with the built-in turtle is that it's not easy to get the drawn canvas as a numeric array. It's possible after lots of effort, but then you'd find that it was very slow.

This package wraps/overrides some of the built-in turtle methods to keep a (*approximate) copy of the drawn canvas as a NumPy array. I say approximate because the arrow I use as the turtle itself (a 10x10 icon) roughly looks like the one used in the native package, but isn't the same. It doesn't matter.

# What's this for?

Python's turtle makes a *great* testing ground for reinforcement learning / deep-Q learning algorithms, or it would if we could easily get at the array/data/rendered image underlying the window in which the turtle shows. So, this makes that array available.

## Usage

```
from rlturtle import RLTurtle
tortoise = RLTurtle(200,200, n_history_frames=2)
tortoise.forward(10)
tortoise.plot()
tortoise.left(90)
tortoise.forward(10)
current_frame = tortoise.canvas
prev_frame = tortoise.prev_frames[1]
older_frame = tortoise.prev_frames[0]

...

model.act(torch.cat((current_frame, prev_frame, older_frame)))
```