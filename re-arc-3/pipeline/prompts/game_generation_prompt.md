Generate game ideas for small deterministic puzzle games. Each idea should be a compact game concept, not a full level specification. The goal is to produce ideas that can later be expanded into implementation-ready specs.

These ideas will later be passed to a separate spec-generation prompt that expands each idea into exact rules, visuals, and authored levels. Therefore, each idea must give the spec generator a strong mechanic to amplify. The idea should contain enough concrete mechanic vocabulary that the later spec can preserve it: the main objects, player actions, visible state, update consequence, objective, and the kinds of level progression the mechanic supports. Avoid vague ideas where the spec generator would need to invent the real game.

A good game idea should be built around one clear formal mechanic or interface, not a collection of unrelated devices. It should be learnable from a small number of interactions and remain readable in a small 64×64 visual implementation.

The core mechanic must have high role elasticity. This means the same rule should naturally support many different strategic roles depending on context, timing, selection, ordering, layout, resource pressure, object placement, symbolic state, or interaction with other systems. The game should not become interesting only by adding many unrelated object types.

Prefer mechanics where player actions create visible, inspectable consequences that affect future possibilities. The player should feel that they are changing the puzzle state in a meaningful way, not merely triggering isolated effects. Consequences may include changes to pattern state, rule interpretation, cursor position, selection, program state, resource flow, object placement, connectivity, reachability, timing, visibility, force, topology, or relationships between components.

The mechanic should involve visible state. Strong state variables include position, orientation, polarity, height, tension, color, connection, trail, light, weight, attachment, shape, selected item, program token, symbol identity, resource amount, target pattern, cursor state, or phase, as long as they are easy to inspect. Hidden or abstract state is acceptable only when the game provides clear visual evidence for learning and reasoning about it.

The idea should create interactions between parts of the system. A strong game is rarely just about changing one independent variable until it matches a target. The core mechanic should allow one choice, object, token, rule, cursor, actor, resource, or region to change what is possible for another: blocking, redirecting, transforming, selecting, activating, delaying, separating, combining, constraining, proving, routing, or invalidating it. This interaction should come from the main rule, not from unrelated secondary devices.

The mechanic should support a clean level progression. Before outputting an idea, mentally check whether it could survive expansion into 6–9 authored levels. The core mechanic should naturally support tutorial levels, middle levels with interaction and conflict, and later levels with dependency networks. Progression should come from new consequences, dependencies, layout or representation changes, timing or resource pressures, interactions between system parts, or reversals of earlier assumptions, not from simply adding more nouns. Later levels should combine earlier mechanics into dependency networks rather than longer linear recipes.

Wrong moves should fail visibly and teach the player something. The player should be able to understand why a plan failed by inspecting the board, interface, trace, target, or replay. Symbolic or validation-heavy games are acceptable when the evidence for success and failure is visible and learnable.

If clicking is necessary, the spec must make the clickable objects visually obvious, state exactly which objects or regions can be clicked, explain why clicking belongs to the core mechanic, and include early levels where the click affordance is taught through a visible local impasse. Avoid designs where the player must guess which arbitrary object on the board is clickable.

The idea must be visually robust. It should compress cleanly into simple, high-contrast shapes, symbols, panels, actors, or modules. Important state should be readable at a glance or through a compact visual grammar. Dense symbolic, panel-based, or exact-pattern games are allowed, but they must use clear grouping, large hit regions, consistent notation, and visual feedback that fits a 64x64 implementation. A slightly simpler mechanic that renders clearly is better than a clever mechanic that becomes visually confusing.

Reject ideas where the mechanic has only one main use, even if it is clear and intuitive. Avoid one-use gimmicks, pure path drawing with no interaction, or delivery tasks with no interesting system. Symbolic lock-and-key puzzles, attribute constraints, machine calibration, setup-and-execute simulations, and checklist-like validation are allowed when the player reasons through visible structure and when the same rule keeps changing meaning through context.

Secondary mechanics are allowed, but only if they amplify the core formal system. They should make the original rule play a new role rather than replace it with a separate mini-game. If the idea needs many special cases, modes, or object types to become interesting, the core mechanic is probably too weak.

Harder levels should introduce new ways to use existing mechanics before adding new mechanics.

The description should give the spec generator hooks for level design: what the first interaction teaches, what can go wrong, how one system part can affect another, how the mechanic can become a tool, constraint, or liability, and what kind of final dependency puzzle the game could support.

Output a valid JSON array of game idea objects. Do not include commentary outside the JSON.

Each object must have exactly these fields:

"title": a short 2–4 word title.
"description": one compact paragraph, usually around 140 words.
"num_levels": integer
