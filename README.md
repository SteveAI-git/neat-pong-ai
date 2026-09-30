# NEAT Pong — Real-Time NeuroEvolution Demo

A minimal, presentation-ready demo of **NeuroEvolution of Augmenting
Topologies (NEAT)** learning to play Pong. A population of 50 neural
networks evolves, generation by generation, to track the ball and score
against a rule-based opponent — with the current best player rendered
live on screen.

## Project Structure

```
pong-neat/
├── config-feedforward.txt   # NEAT hyperparameters (population, mutation rates, network shape)
├── game.py                  # Paddle & Ball physics, rule-based opponent AI, smoothed movement control
├── main.py                  # NEAT training loop, fitness function, rendering, checkpointing
├── replay.py                # Load a saved genome and watch it play, no training
├── best_genome.pkl          # Created automatically the first time you stop training
└── README.md                # This file
```

## Installation & Setup

Requires Python 3.8+.

```bash
pip install pygame neat-python
```

Then run:

```bash
python main.py
```

A window will open and training starts immediately — no extra setup or
dataset needed. Console output (via NEAT's `StdOutReporter`) shows
per-generation fitness statistics alongside the visual demo.

Training runs **indefinitely** by design (`fitness_threshold` is set
unreachably high in the config) — close the window or press `Ctrl+C` in
the terminal whenever you want to stop.

## How It Works, In Brief

- **The green paddle** (right) is controlled by a NEAT neural network.
  Every genome in the population of 50 plays its own independent game
  simultaneously in the background — all 50 games are being simulated
  every frame, whether or not you can see them.
- **The red paddle** (left) is a simple rule-based opponent with
  human-like imperfection (~65% reaction accuracy, ±12px deadzone) —
  beatable, but not a pushover.
- Only **one match is drawn on screen at a time** — the "spotlight"
  game. The spotlight is only handed off to a different genome when
  the one currently being watched actually loses (misses the ball); it
  is *not* re-picked every frame. This keeps what you watch smooth and
  continuous — one full match at a time, with a clean handoff to the
  next contender only when a match genuinely ends — instead of
  flickering between many genomes' games.
- **Fitness signal:**
  - Small continuous reward for keeping the paddle Y-aligned with the ball.
  - `+100` for hitting the ball.
  - `+200` for scoring a goal against the opponent.
  - `-50` and elimination for that genome's round when the AI misses the ball.
- **The fitness graph** in the bottom-right corner plots two lines per
  generation: best fitness (green) and average population fitness
  (white). The average line is the stronger evidence of the two — it
  rising shows the *whole population* getting better, not just one
  lucky genome.
- **A "★ NEW ALL-TIME BEST!"** banner flashes across the top whenever
  the genome currently on screen beats every genome from every prior
  generation — an unambiguous, hard-to-miss "it just got better" signal.
- **A short pause between generations** shows that generation's final
  best/average fitness before breeding the next one, so transitions are
  a visible beat instead of an instant, easy-to-miss jump.

## Why It's Smoother and Faster Now

Two specific refinements address jittery movement and slow wall-clock
training:

- **Hysteresis-based paddle control** (`decide_ai_direction` in
  `game.py`): instead of a single "output > 0.1 → up" threshold, the
  paddle needs a *larger* swing in the network's output to reverse
  direction than it needed to start moving. This is the same trick used
  in thermostats to stop rapid on/off chattering — it removes
  paddle-vibration near-zero outputs immediately, from generation 1, not
  just after many generations of training.
- **`SIM_SPEED` in `main.py`** (default `2`): the game simulates this
  many logic steps per rendered frame, so training covers more
  in-game time per second of wall clock while still drawing a full,
  steady 60 FPS. It plays like watching a brisk 2x-speed match — still
  fully smooth, just faster to sit through. Set it to `1` for real-time
  speed during a careful walkthrough, or `3`–`4` to blaze through early
  generations faster (at the cost of being less watchable).

## Saving & Replaying Your Best AI

Training saves the best genome NEAT has found **automatically** — no
extra steps needed:

- **On every exit path**: closing the window, pressing Ctrl+C in the
  terminal, or (if you let it run that long) reaching the fitness
  threshold. This is why the code tracks NEAT's own
  `population.best_genome` rather than only the return value of
  `population.run()` — since training runs indefinitely by design,
  `population.run()` normally never returns on its own.
- **Periodically during a long run** (every 10 generations) as a safety
  net against crashes or power loss, saved silently in the background.

The genome is written to `best_genome.pkl` in the project folder. To
watch it play — no training, no population, just that one trained
network against the rule-based opponent:

```bash
python replay.py
```

or point it at a specific saved file:

```bash
python replay.py my_saved_genome.pkl
```

`replay.py` reuses the exact same `decide_ai_direction()` movement logic
from training, so what you see in replay is genuinely how that genome
plays — not a different, unrelated control scheme.

## Tuning Ideas (Optional Extensions)

- Increase `pop_size` in `config-feedforward.txt` for a broader search
  (slower per generation, but more genetic diversity).
- Adjust `opponent_ai_move`'s `accuracy`/`deadzone` in `game.py` to make
  the rule-based opponent easier or harder to beat.
- Tune `SIM_SPEED` in `main.py` to trade watchability for training speed.
- Widen/narrow `hysteresis` in `decide_ai_direction` (`game.py`) to make
  paddle movement even smoother (wider) or more responsive (narrower).
