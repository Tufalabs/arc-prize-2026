# LP85 - Looping Chains

Official task URL: https://arcprize.org/tasks/lp85

Local implementation: `re_arc/environment_files/lp85/305b61c3/lp85.py`

Zip source: `Games/lp85 ...md` plus `Games/lp85/image*.png`

## Core Game

Looping Chains is a click-only synchronization puzzle. Objects ride on closed loops, belts, or chained tracks. Clicking buttons advances or reverses specific loops, and the player must align moving objects with target boxes. As the levels progress, multiple loops overlap, share objects, rotate, teleport at edges, or have hidden coupling. The win condition is that all tracked objects occupy their matching targets at the same time.

To reconstruct it, model each loop as an ordered cycle of positions. Buttons should advance one or more objects along those cycles by deterministic step counts or directions. Some objects belong to multiple logical chains, so later levels need coupled updates and simultaneous validation.

## Level Breakdown

- Level 1: Introduces the basic loop. Click the control until the yellow object sits on the target box.

- Level 2: Adds multiple overlapping belts. The player must account for how advancing one chain affects shared or crossing positions.

- Level 3: Adds rotation and multiple targets. Objects must be aligned with several boxes at once, not one at a time.

- Level 4: Adds edge teleportation. Objects that leave one edge of the grid re-enter through another linked edge while still needing to match multiple targets.

- Level 5: Uses subset relationships between loops. One chain's positions are contained in or constrained by another, so the player must align relative phase.

- Level 6: Removes a simple reverse option. One control advances in a direction, another moves along a star-shaped route, and an extra side button controls the last exchange.

- Level 7: Adds a hidden relationship between objects. The player must experiment to learn which control aligns the unseen or indirectly connected partner.

- Level 8: Final synchronization level. Three objects must arrive at their targets at the same time even though they are not aligned in the starting view.

## Reconstruction Notes

- Use click-only buttons tagged to specific loops.
- Represent loops as cycles with current phase.
- Validation must be simultaneous across all targets.
- Later levels should include coupled controls where one click changes more than one object.
- Edge teleportation is still a loop transition, not a separate free move.
