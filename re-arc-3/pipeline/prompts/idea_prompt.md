You receive exactly one short seed description, usually one or two sentences,
for a deterministic visual puzzle game in the RE-ARC game-generation pipeline.
Expand that seed into exactly one complete game idea.

Keep the entire response between 350 and 650 words, including the reproduced
seed, and never exceed 700 words. Prefer a shorter faithful idea over filling
space. This stage selects and clarifies the design; it does not exhaustively
document it.

The supplied seed is authoritative. Preserve its title when one is supplied,
its objects, concrete player verbs, central causal rule, objective, and intended
identity. Elaborate missing consequences, visual language, strategic
discoveries, and broad progression conservatively. Do not replace the seed
with a cleaner, more elaborate, or only loosely related game. Do not add an
independent mechanic merely to manufacture depth.

This is a game idea, not a complete specification. Do not prescribe exact
level layouts, coordinates, action strings, state-machine update order,
animation sequences, palette assignments, or detailed implementations. Include
only visual details needed to distinguish gameplay objects and persistent
state. Begin the response by reproducing the supplied seed verbatim so the
downstream spec stage can distinguish source material from elaboration.

The downstream game interface supports only `UP`, `DOWN`, `LEFT`, `RIGHT`,
`SPACE`, and clicking a cell; a game may use any subset. Every proposed player
action must map directly to that interface. Do not invent a second directional
action family such as four movement controls plus four pulling or dragging
controls, and do not invent separate Turn Left and Turn Right buttons. Map a
necessary local interaction conservatively to `SPACE`, visible facing plus
`SPACE`, direct contextual contact, or a spatially corresponding directional
input. Default to a click-free design. Use cell clicks only when the supplied
seed explicitly makes direct clicking central and no faithful mapping to the
directional controls or `SPACE` exists; do not introduce clicking merely for
convenience. A level reset is external recovery, not a gameplay action. State
the chosen control mapping briefly and do not introduce control modes unless
the supplied seed truly requires one and its active state is persistently
visible.

The game will be played by a human or black-box visual model that initially knows only:

- the available actions;
- the current visual observation;
- the visible results of previous actions.

The player will not have access to the written rules. Every important rule and strategic possibility must therefore be discoverable through interaction.

## Central objective

The expanded idea should define one focused, visually legible formal system
with simple deterministic rules and several non-obvious strategic consequences
that follow naturally from the supplied seed.

The game should be easy to simulate once understood, but capable of producing
deep later puzzles through consequences and interactions inherent in the
supplied rules.

Do not equate depth with adding many devices or object types.

## Perceptual concreteness gate

Prefer games built around concrete visible objects undergoing familiar,
spatially local transformations: moving, rotating, pushing, carrying,
stacking, painting, filling, blocking, matching, or colliding.

If the supplied seed's primary manipulated state is an inferred global relation
rather than a directly inspectable object state, identify the exact conflict
and propose the smallest faithful repair. Do not silently replace it with an
unrelated idea. Disfavored primary mechanics include:

- graph connectivity or loop/component membership;
- inside/outside or containment classification;
- parity, phase, mode, or another abstract global label;
- action-history-dependent behavior;
- restoring several hidden or visually subtle state variables;
- rules whose effects can only be understood by comparing distant regions or
  remembering previous frames.

Such relations may appear as secondary consequences only when they are
completely externalized through persistent, unmistakable visual objects or
status indicators.

A player must be able to point to the object being manipulated, describe the
core action with a concrete verb, and predict its immediate local effect from
the current settled frame.

## Important definitions

An operational rule or mechanic describes what the supplied system does.

A strategic discovery or derived technique is a non-obvious consequence of
those already established rules. Derive discoveries from the particular
objects, actions, state transitions, and objective in the supplied seed. Do not
steer the idea toward a generic template such as delaying goals, undoing
progress, revisiting states, preserving return paths, or using temporary
blockers unless that behavior follows directly and necessarily from the seed's
core rule.

The strongest progression should introduce relatively few operational rules but several strategic discoveries.

## Epistemic solvability

At every point, a blind player must have a reasonable way to form a useful hypothesis and choose an informative experiment.

This does not mean the solution should be obvious. It means difficulty must come from reasoning with visible evidence rather than hidden information, arbitrary guessing, unexplained exceptions, or ambiguous presentation.

A useful discovery should follow this pattern:

1. The player understands the rules demonstrated so far.
2. A reasonable greedy or obvious plan reaches a visible impasse.
3. The state provides evidence that a different use of the established rules may work.
4. A low-cost experiment reveals the new strategic technique.
5. The technique is necessary for progress and is reused later in another context.

Use this pattern as a design test, not as a checklist to reproduce for every
discovery. The obvious plan must be reasonable given the player’s prior
evidence. Do not create arbitrary gotchas.

## Core design requirements

The expanded idea must:

- Use one coherent game family rather than a collection of unrelated devices.
- Have a small action space with clear and consistent effects.
- Keep every strategically relevant state variable visibly readable
  after all animation ends. Animation may emphasize a change but may never be
  the only evidence of the resulting state.
- Make the settled frame sufficient to determine the current gameplay
  state. Do not require the player to remember an earlier frame or reconstruct
  action history.
- Give every independent win-condition requirement its own persistent,
  spatially associated visual cue.
- Make controls describe their actual semantic effect. Do not encode abstract
  operations such as Forward, Back, Restore, or Change Mode as misleading
  directional controls.
- Give every available action a meaningful, visibly distinguishable role.
  Do not expose inert or contextlessly silent actions.
- Keep the foundational objective to one primary visible condition.
  Add additional conditions only after each has been taught independently.
- Provide immediate visible feedback when an attempted action is blocked or fails a condition.
- Keep foundational exploration safe and recoverable, without tight timers, budgets, or irreversible traps.
- Introduce at most one new causal unknown at a time during the foundational curriculum.
- Ensure every visible gameplay object has a necessary communicative or mechanical role.
- Support several meaningful strategic discoveries without requiring additional object types for each one.
- Reuse every important discovery in a contrasting context after introducing it.
- Produce later difficulty through consequential uses of the supplied rules,
  not traversal padding or repeated identical actions.

Clicks are allowed only when clickable objects are visually unmistakable and clicking is a natural interpretation of their appearance. Prefer direct controls when clicking would create unnecessary ambiguity.

## Necessity-driven strategic progression

Identify 3–4 strategic discoveries that the supplied rule naturally supports.

At least 2 of them must emerge from new uses or consequences of existing rules
without introducing a new object type or changing the operational rules.

Describe each discovery in one compact bullet of no more than two sentences.
Name the insight, the visible evidence that makes it discoverable, and why it
can become necessary or be reused. Do not separately enumerate the greedy plan,
impasse, experiment, necessity, and reuse for every discovery.

Do not merely list features such as “add walls,” “add another color,” or “add more objects.” Explain the new reasoning those elements force.

A strong discovery changes the player’s understanding of how to use the
supplied system. Do not import a discovery pattern merely because it is common
in puzzle games.

## Progression capacity

The expanded idea should plausibly support 7–9 authored levels, but do not
choose or announce an exact level count. The downstream game-idea JSON supplies
the authoritative `num_levels` value.

Describe progression only at a broad level and in no more than five sentences
total. Cover the following arc without turning every phase into a detailed
level plan:

- Foundational phase: clean scenes establish the supplied objective and atomic
  causal rules.
- Early composition: the first consequences specific to the supplied rule
  become necessary.
- Middle phase: established consequences are reused in contrasting contexts
  that arise naturally from the same system.
- Late phase: multiple previously taught consequences interact without adding
  unrelated mechanics.
- Final phase: the level tests the fullest coherent use of the supplied rule
  family.

Do not target a prescribed action count at the idea stage. Depth should come
from consequential decisions and interactions inherent in the supplied
mechanic, not oversized maps, long corridors, repetitive actions, or an
arbitrary solution-length target.

Do not specify the exact first levels. Explain what must be learned and how the reasoning develops.
