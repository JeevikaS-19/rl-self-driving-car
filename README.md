# RL Self-Driving Car — MetaDrive

A deep reinforcement learning self-driving agent built in MetaDrive (3D), trained with SAC.

## Problem

Most self-driving systems are built and tested for structured, rule-following traffic. Indian roads don't work like that; lane discipline is loose, right-of-way is negotiated in real time, and traffic is dense and mixed (cars, bikes, pedestrians, unstructured junctions). Inspired by Swaayatt Robots' work on autonomy in exactly this kind of unstructured environment, I wanted to understand how deep RL actually handles this problem, not just read about it.

This project started as a complete beginner in ML, built first-principles/Socratic style, alongside working through Stanford's CS234 RL lectures.

## Architecture

**Perception:** LiDAR data projected into an occupancy grid (Phase 1), stacked across 4 frames, passed through a CNN with attention into a 2048-dim latent vector.

**Actor (SAC):** takes the latent vector plus scaled-down navigation info (10-dim, scaled ×0.1 so it doesn't overwhelm the CNN features).

**Critic (SAC, dueling, twin):** takes the latent, nav_info, action, *and* the count of nearby agents (16-dim) — a piece of information the actor never sees. This is an asymmetric privileged critic: the critic gets more context than the actor to train against, which the actor doesn't get at inference time.

Other key choices: auto-tuned entropy temperature (target entropy -2.0), gamma compressed to 0.90 to fight baseline-dominance in an earlier flat critic (worth revisiting now that's fixed), a 200k-capacity replay buffer with 95% suppression of stationary/braking transitions to avoid overfitting to "do nothing," and the buffer seeded with ~37.5k expert demonstration transitions from MetaDrive's built-in autopilot.

## What actually broke, and what fixed it

Three root causes were found across a long debugging arc:

1. **Flat critic** — the critic couldn't distinguish state value from action advantage. Fixed by moving to a dueling critic (V(s) + centered advantage), confirmed via a ~20x increase in Q-value spread.
2. **Missing nav_info** — navigation info wasn't actually wired into the actor/critic inputs for a full training run, due to a copy-paste gap.
3. **Single-scenario training** — the simulator was training on one scenario repeatedly instead of a diverse set. Setting `num_scenarios=200` was the single biggest lever: it produced the first run with genuinely reactive, curve-responsive steering and real braking-behind-traffic behavior.

Ruled out along the way: comfort penalty (pathology persisted even at zero), demo-data label imbalance, and demo-data temporal clustering — none of these explained the steering issues being debugged.

## Results (750k-step checkpoint, 10-episode evaluation)

- Mean 132 steps survived per episode (median 112.5)
- Mean route completion: 33.7%
- 0/10 episodes reached the destination
- 10/10 episodes showed at least one genuine braking-for-traffic event

Reactive, safety-relevant behavior generalized. Full-route navigation did not.

## Where it stopped

After resuming training at a lower learning rate (3e-4 → 1e-4) to reduce optimizer shock, the car reliably hit higher speeds (15-22+ m/s), but steering became visibly jittery specifically above ~15 m/s — an untested hypothesis is that the policy never learned to scale down steering correction at higher speed.

At this point, results weren't improving fast enough to justify continuing to iterate on a single-agent setup. That question, how do you get *negotiation* between agents instead of just individual reactive driving, is what led directly to [Project Bee](link-to-project-bee-repo), a multi-agent approach inspired by bee-colony coordination instead.

## Blog writeup

Full reasoning, math, and debugging process documented in the ["Self-driving cars" series](https://local-minima.hashnode.dev/series/self-driving-rl) on my blog.
