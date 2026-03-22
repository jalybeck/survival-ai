# Survival AI

Survival AI is an educational reinforcement learning sandbox built in plain Python.

The project intentionally implements the AI pieces by hand instead of using PyTorch, TensorFlow, JAX, Stable-Baselines, or similar frameworks. The goal is not only to produce behavior, but to make the behavior inspectable: observations, hidden activations, action scores, rewards, episode memory, and weights can all be examined directly.

`pygame` is used only as the visualization layer.

## Core Idea

Three agents share one policy network and compete in a deterministic grid world with:

- movement
- melee combat
- ranged combat
- heal pickups
- melee weapon pickups
- ranged weapon pickups
- line-of-sight limited perception
- episode-based policy-gradient learning

The same application can be used as:

- a visual sandbox
- a live on-screen training environment
- a headless trainer
- a debugging tool for neural network behavior

## Why This Repository Exists

This repository is designed for learning:

- how an observation vector is constructed
- how a feedforward neural network works internally
- how policy-gradient reinforcement learning updates weights
- how reward shaping changes behavior
- how RL agents exploit loopholes if the reward function allows it
- how architecture and hyperparameters influence training

The emphasis is on clarity and experimentation rather than raw training speed.

## Requirements

- Python 3.12 is recommended on Windows
- `pygame` is required for visual modes

Install `pygame` with:

```powershell
py -3.12 -m pip install pygame
```

## Running the Project

With no arguments, the application starts in visible live-learning policy mode:

```powershell
py -3.12 -m survival_ai.main
```

Current default startup behavior:

- mode: `policy`
- policy playback mode: `sample`
- max ticks per episode: `1000`
- if no valid `weights.json` exists, the app starts from fresh random weights and writes new weights after completed episodes

Other useful commands:

```powershell
py -3.12 -m survival_ai.main debug
py -3.12 -m survival_ai.main policy
py -3.12 -m survival_ai.main train 1000
py -3.12 -m survival_ai.main policy --seed auto --tick 500
py -3.12 -m survival_ai.main train 2000 --seed 42 --tick 1000
```

### Runtime Modes

- `debug`
  - scripted agents
  - no policy control
  - useful for validating mechanics and renderer behavior
- `policy`
  - the shared neural network chooses actions
  - visible live learning is enabled
  - updated weights are saved after each completed episode
- `train [episodes]`
  - headless self-play training
  - no pygame window
  - weights are saved periodically and at the end

### Visual Controls

- `Space`: advance one tick
- `A`: toggle auto-advance
- `R`: reset the current visible episode
- `Tab`: switch the debug/inspection agent
- `O`: print the current observation and scores to the terminal
- `M`: cycle policy playback mode in `policy` mode
- click `>>` / `<<`: expand or collapse the network panel

## World Model

The arena is a deterministic 2D grid with:

- border walls
- fixed interior obstacles
- deterministic agent spawns
- deterministic item spawns

Each simulation tick resolves in a fixed order:

1. clear per-step flags
2. resolve movement
3. resolve item interactions
4. resolve melee and ranged attacks
5. update reward-shaping state
6. determine deaths and episode termination

This deterministic step order is important because RL behavior depends heavily on update ordering.

## Observation System

Agents do not see the full world state. They observe:

- their own normalized position
- their own normalized health
- immediate wall flags
- immediate visible-adjacent enemy flags
- nearest visible enemy direction and distance
- nearest visible item direction and distance
- self damage indicators
- whether attack or shoot is currently possible
- visible agent count
- visible item count
- visible cell ratio
- inventory and equipped weapon state

The debug UI also shows:

- the local symbolic observation grid
- the full feature vector
- hidden-layer activations
- output scores
- per-tick changes in activations

### Line of Sight

Visibility uses an integer line stepping method similar to Bresenham line tracing. A target cell is visible if a straight line from the agent to that cell does not cross a wall tile.

## Neural Network

The policy network is a plain-Python multilayer perceptron implemented in `survival_ai/network.py`.

Given an input vector \(x\), each dense layer computes:

\[
z = Wx + b
\]

where:

- \(x\) is the input vector coming from the previous layer
- \(W\) is the weight matrix
- \(b\) is the bias vector
- \(z\) is the pre-activation output of the layer before any activation function is applied

More concretely:

- each weight in \(W\) says how strongly one input value influences one neuron
- each bias in \(b\) is a small learned offset added after the weighted sum
- the bias helps a neuron shift its activation threshold instead of always depending only on input-weight products

Hidden layers apply ReLU:

\[
\text{ReLU}(z) = \max(0, z)
\]

`ReLU` stands for `Rectified Linear Unit`.

It is a very simple activation rule:

- if the incoming value is positive, keep it
- if the incoming value is zero or negative, output `0`

Examples:

- `ReLU(2.4) = 2.4`
- `ReLU(0.0) = 0.0`
- `ReLU(-1.7) = 0.0`

In practice this means:

- positive neuron signals are allowed through
- negative signals are clipped to zero
- many hidden neurons will often be exactly `0`, which is normal in ReLU networks

Output layers remain linear:

\[
\text{logits} = z
\]

These logits are converted into action probabilities using softmax over legal actions only:

\[
p(a_i \mid s) = \frac{e^{\ell_i}}{\sum_{j \in \mathcal{A}_{legal}} e^{\ell_j}}
\]

where:

- \(\ell_i\) is the raw output score for action \(i\)
- \(\mathcal{A}_{legal}\) is the current set of legal actions

### Shared Policy

All agents share the same network weights.

That means:

- one policy function
- different behaviors emerge from different observations
- all agents contribute training data to the same parameter set

This is useful for self-play because it keeps the system symmetric and easier to study.

## Reinforcement Learning Algorithm

Training is currently episode-based REINFORCE with discounted returns.

For each step \(t\), the return is:

\[
G_t = r_t + \gamma G_{t+1}
\]

where:

- \(r_t\) is the immediate reward at step \(t\)
- \(\gamma\) is `DISCOUNT_FACTOR`

Returns are normalized before the update to reduce variance.

The policy-gradient-style update uses:

\[
\nabla \log \pi(a_t \mid s_t) \cdot G_t
\]

In code terms:

- actions are sampled from the current policy
- the chosen action probability is reinforced or weakened depending on normalized return
- gradients are accumulated over the whole episode
- the network is updated once at episode end

This is intentionally simple and educational, not state-of-the-art.

## Reward Model

The reward is shaped from multiple components:

- survive tick reward
- reward for dealing damage
- penalty for taking damage
- death penalty
- win reward
- exploration reward for entering new tiles
- approach reward for moving closer to visible enemies
- reward for entering attack range
- item pickup reward
- heal use reward
- weapon use reward
- drop penalty
- idle penalty
- oscillation penalty
- no-winner penalty on timeout/draw

The total per-step reward is the sum of these components:

\[
R_t = R_{survive} + R_{deal} + R_{take} + R_{death} + R_{win} + R_{explore} + R_{approach} + R_{range} + R_{pickup} + R_{heal} + R_{weapon} + R_{drop} + R_{idle} + R_{loop} + R_{draw}
\]

This design is intentionally exposed and editable because reward shaping is one of the main learning goals of the repository.

## Live Training vs Headless Training

Two training styles are supported:

- headless training in `train`
- visible live learning in `policy`

In both cases:

- step data is collected every tick
- weights are updated after the episode ends, not after every tick

That means the current algorithm is episodic, not bootstrapped step-wise RL.

## Architecture and Visualization

The right-side network panel shows:

- input values
- hidden layer activations
- output values
- action names aligned with outputs

It also colors values by change since the previous tick:

- green: increased
- red: decreased
- white: unchanged

If the selected debug agent dies, the panel freezes on that final state and labels the view as `DEAD`.

## Configuration Knobs

All major tuning values live in `survival_ai/config.py`.

### World and Episode Settings

| Name | Default | Meaning |
|---|---:|---|
| `MAP_WIDTH` | `15` | Grid width in tiles |
| `MAP_HEIGHT` | `15` | Grid height in tiles |
| `NUM_AGENTS` | `3` | Number of agents spawned into the arena |
| `VISION_RADIUS` | `5` | Visibility radius used for local observation and LOS queries |
| `MAX_HEALTH` | `10` | Maximum health per agent |
| `MELEE_DAMAGE` | `2` | Base melee damage without a melee weapon |
| `MAX_EPISODE_LENGTH` | `200` | Default episode cap when no explicit `--tick` override is supplied |

### Item and Combat Settings

| Name | Default | Meaning |
|---|---:|---|
| `HEAL_ITEM_AMOUNT` | `4` | Health restored by a heal item |
| `MELEE_WEAPON_DAMAGE` | `4` | Damage dealt while a melee weapon is equipped |
| `MELEE_WEAPON_ATTACK_CHARGES` | `3` | Number of melee attacks before the melee weapon is consumed |
| `RANGED_WEAPON_DAMAGE` | `1` | Damage dealt by a ranged shot |
| `RANGED_WEAPON_SHOT_CHARGES` | `4` | Number of shots before the ranged weapon is consumed |
| `RANGED_WEAPON_RANGE` | `5` | Maximum shot distance in tiles |

### Window, Timing, and Debug UI

| Name | Default | Meaning |
|---|---:|---|
| `CELL_SIZE` | `48` | Pixel size of one grid cell |
| `TOP_PANEL_HEIGHT` | `138` | Height of the top UI panel |
| `BOTTOM_PANEL_HEIGHT` | `380` | Height of the bottom UI panel |
| `WINDOW_EXTRA_WIDTH` | `280` | Additional width beside the centered game board |
| `FPS` | `6` | Render-loop frame cap for visual modes |
| `RESET_DELAY_MS` | `1400` | Delay before an ended visible episode auto-resets |
| `TRACER_DURATION_MS` | `450` | Duration of ranged shot tracer lines |
| `NETWORK_PANEL_WIDTH` | `980` | Width reserved for the expandable right-side network panel |
| `NETWORK_PANEL_BUTTON_WIDTH` | `44` | Toggle button width |
| `NETWORK_PANEL_BUTTON_HEIGHT` | `28` | Toggle button height |

### Colors

| Name | Meaning |
|---|---|
| `BACKGROUND_COLOR` | Global background color |
| `GRID_LINE_COLOR` | Grid line color |
| `FLOOR_COLOR` | Walkable tile color |
| `WALL_COLOR` | Wall tile color |
| `TEXT_COLOR` | Main text color |
| `VISIBLE_OVERLAY_COLOR` | LOS/visibility overlay tint |
| `RANGED_TRACER_COLOR` | Ranged shot tracer color |
| `NETWORK_PANEL_BACKGROUND` | Right panel background |
| `NETWORK_PANEL_BORDER` | Right panel border |
| `AGENT_COLORS` | Per-agent display colors |

### Learning Hyperparameters

| Name | Default | Meaning |
|---|---:|---|
| `LEARNING_RATE` | `0.01` | Step size for gradient-based weight updates |
| `DISCOUNT_FACTOR` | `0.95` | How strongly future rewards influence current returns |
| `HIDDEN_LAYER_SIZES` | `(24, 24)` | Hidden layer widths of the shared MLP |

### Reward-Shaping Parameters

| Name | Default | Meaning |
|---|---:|---|
| `SURVIVE_TICK_REWARD` | `0.01` | Reward for remaining alive for one more tick |
| `DEAL_DAMAGE_REWARD` | `1.0` | Base reward for successfully damaging another agent |
| `TAKE_DAMAGE_PENALTY` | `-1.0` | Base penalty for taking damage |
| `DEATH_PENALTY` | `-5.0` | Penalty on death |
| `WIN_REWARD` | `5.0` | Reward for winning the episode |
| `NEW_TILE_REWARD` | `0.005` | Reward for visiting a previously unseen tile this episode |
| `IDLE_PENALTY` | `-0.01` | Penalty for staying on the same tile |
| `OSCILLATION_PENALTY` | `-0.03` | Penalty for short A-B-A movement loops |
| `NO_WINNER_PENALTY` | `-2.0` | Penalty applied to surviving agents on timeout/draw |
| `APPROACH_VISIBLE_AGENT_REWARD` | `0.02` | Reward for reducing distance to the nearest visible enemy |
| `ENTER_ATTACK_RANGE_REWARD` | `0.05` | Reward for newly entering melee range |
| `ITEM_PICKUP_REWARD` | `0.20` | Reward for a valid item pickup event |
| `HEAL_ITEM_USE_REWARD` | `0.35` | Reward for successfully using a heal item |
| `WEAPON_ITEM_USE_REWARD` | `0.20` | Reward for equipping a non-recycled weapon item |
| `DROP_ITEM_PENALTY` | `-0.02` | Penalty for dropping an item |

### Runtime Defaults

| Name | Default | Meaning |
|---|---:|---|
| `DEFAULT_TRAIN_EPISODES` | `200` | Fallback episode count for `train` mode |
| `WEIGHTS_PATH` | `"survival_ai/saves/weights.json"` | Main save file for network parameters |
| `DEFAULT_SEED` | `7` | Default random seed |
| `DEFAULT_START_MODE` | `"policy"` | Mode used when the app starts with no CLI mode argument |
| `DEFAULT_START_TICKS` | `1000` | Max ticks used when the app starts with no CLI args |
| `DEFAULT_POLICY_MODE` | `"sample"` | Initial policy playback mode in visible policy mode |

## File-by-File Guide

### Root

| File | Purpose |
|---|---|
| `README.md` | Project overview, algorithms, usage, and configuration reference |
| `.gitignore` | Git ignore rules for Python artifacts and runtime weight files |

### Main Package: `survival_ai/`

| File | Purpose |
|---|---|
| `survival_ai/__init__.py` | Package marker |
| `survival_ai/main.py` | CLI parsing, runtime mode dispatch, visible loop, and live policy training integration |
| `survival_ai/config.py` | Central configuration and tuning constants |
| `survival_ai/actions.py` | Action enum and directional/action helper groupings |
| `survival_ai/world.py` | Deterministic simulation core: movement, combat, item use, deaths, and episode progression |
| `survival_ai/mapgen.py` | Arena generation and wall storage |
| `survival_ai/entity.py` | Agent and item entity state, inventory logic, weapon consumption, and per-step flags |
| `survival_ai/items.py` | Item blueprint classes and item-entity factory |
| `survival_ai/agent.py` | Scripted controller, random controller, and neural policy controller |
| `survival_ai/observation.py` | LOS, visible cells, local symbolic grid, and numeric feature vector construction |
| `survival_ai/network.py` | Pure-Python MLP, forward pass, backward pass, softmax helpers, and save/load |
| `survival_ai/memory.py` | Episode memory structures for policy learning |
| `survival_ai/reward.py` | Per-step reward calculation and reward-component formatting |
| `survival_ai/trainer.py` | Episode collection, return computation, policy-gradient updates, and persistence |
| `survival_ai/render.py` | `pygame` renderer, HUD, network panel, and tracer drawing |
| `survival_ai/utils.py` | Small shared helpers such as bounds checks and Manhattan distance |

### Notes and Saves

| File | Purpose |
|---|---|
| `survival_ai/notes/experiments.md` | Free-form experiment notes and observations |
| `survival_ai/saves/weights.json` | Main runtime save file for learned network weights |

## Validation

After changing Python code, compile the package:

```powershell
py -3.12 -m compileall survival_ai
```

This catches syntax and import issues quickly.

## Design Principles

- implement AI logic manually where practical
- prefer readable code over opaque abstractions
- keep rendering separate from training and simulation logic
- expose state and intermediates for debugging
- make reward shaping and policy behavior easy to inspect

## Current Limitations

- training is intentionally simple and not optimized for speed
- reward shaping is still experimental
- policy quality is sensitive to seeds, rewards, and episode length
- this is a research/learning sandbox, not a production RL framework

## Suggested Experiments

- widen the network: `HIDDEN_LAYER_SIZES = (32, 32)` or `(48, 48)`
- deepen the network: `HIDDEN_LAYER_SIZES = (24, 24, 24)`
- change `DISCOUNT_FACTOR` to study short-term vs long-term behavior
- change `LEARNING_RATE` to study stability
- alter reward terms and watch how agents exploit new incentives
- compare `greedy` vs `sample` playback in policy mode

## License

MIT License

Copyright (c) 2026 Survival AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
