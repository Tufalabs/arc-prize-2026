You are constructing a possibility space for deterministic visual puzzle games intended for the RE-ARC game-generation pipeline.

## Fixed implementation envelope

All games must:

- render on a 64×64 canvas using only integer colors 0–15 and no text;
- update deterministically only after a player action;
- make the settled frame sufficient to determine the complete gameplay state;
- represent all strategically relevant state persistently and visibly in the rendered frame;
- use a small set of deterministic interaction rules whose relevant effects are visually legible and inferable through play;
- be implementable as a finite discrete state machine without hidden history, randomness, continuous physics, arbitrary hidden predicates, or subtle latent modes;
- make available interactions and their consequences visually legible;
- plausibly support 7–9 authored levels, with later depth arising primarily from composition and interaction among previously introduced rules rather than larger boards, repetitive execution, or simple increases in quantity.

## Game category specification

**Replace only this section when exploring a different game category.**

[GAME CATEGORY SPECIFICATION GOES HERE]


## Task

Identify **exactly 10** broad, high-value, and nonredundant axes along which games within the specified game category can differ.

Prioritize axes that create the greatest variation in:

- core mechanics;
- interaction structure;
- state transformations;
- reasoning demands;
- dependencies between player actions and subsequent possibilities.

Omit narrow or secondary distinctions.

Each axis must:

- describe one independently controllable operational aspect of the game;
- be useful when generating an initial game idea and later compiling it into an implementation-ready specification;
- admit at least three concrete, meaningfully different, implementable values;
- change visible player-facing behavior without requiring a different renderer or overall game category;
- be independently selectable to a meaningful degree from the other axes.

Exclude:

- fixed implementation choices listed above;
- properties already fixed by the game category;
- themes or cosmetic variations;
- instance-specific parameters or level-specific configuration choices;
- emergent measurements such as difficulty, solution length, or branching factor;
- axes primarily based on hidden information or action history;
- axes whose mechanics depend on hidden, arbitrary, or visually ungrounded classifications of the state;
- distinctions based mainly on representation rather than player-facing operational rules;
- redundant axes whose values are substantially determined by another axis.

For each axis provide:

1. **Axis name**
2. **What operational rule it controls**
3. **Several concrete values**
4. **Why those values create meaningfully different puzzle reasoning**

Group related axes when useful, but keep each axis independently selectable.
