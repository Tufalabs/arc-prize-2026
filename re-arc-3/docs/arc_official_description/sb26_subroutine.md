# Subroutine

## Core Game

Subroutine is a setup-and-execute programming puzzle. The playfield is a 64x64 logical grid with a dark gray background. The upper area contains target boxes and one or more colored component frames. The lower tray, below the horizontal divider at y=53, contains movable tokens. The player edits the component frames by clicking a tray token, then clicking a destination slot. Pressing space runs the checker.

There are two token types:

- A solid colored square is a literal. When execution reaches it, it must match the next target color and then it fills that target box.
- A hollow colored square is a pointer. When execution reaches it, it calls the component whose frame border has the same color. The pointer itself produces no target color.

Component frames are one-row programs. Execution starts in the first component in row-major order; in all levels this is the red frame. A component executes its slots left to right. If a slot is a literal, it emits that color. If a slot is a pointer, the checker pushes the current component and slot onto a call stack and starts the pointed component at its first slot. When a called component reaches its end, control returns to the caller and continues at the next slot. In recursive levels, the game can win as soon as the final target box has been filled, even if the call stack would otherwise continue.

Target boxes are also scanned in row-major order. Most levels have one target row at the top. Level 8 has two rows; all boxes in the first row are checked before the second row.

The player has 64 energy. Each placement or swap costs 1 energy, and each spacebar check costs 1 energy. Undo restores the previous editing state and does not cost energy. If a check reaches an empty slot, a wrong literal color, a bad pointer sequence, or a program end before all target boxes are filled, the current execution highlight flashes, the board returns to edit mode, and the existing placements remain. If energy reaches zero after an action or failed check, the level is lost. A successful check advances to the next level. Manual reset reloads the current level from its initial layout, restoring the tray, fixed tokens, empty slots, target boxes, energy, selection state, and undo history.

## Reconstruction Model

Use these object conventions:

- Component frame: a hollow colored rectangle, 10 pixels high. Its border color is the component identity.
- Slot positions: for an n-slot frame at top-left `(x, y)`, slot `i` is at `(x + 2 + 6*i, y + 2)` for zero-based `i`.
- Empty slot: a small gray square at a slot position.
- Literal token: a 6x6 solid colored square.
- Pointer token: a 6x6 hollow colored square.
- Target box: a 6x6 hollow colored square whose border is the required emitted color. During a successful check, the inside of each target box fills with the emitted color.
- Tray token: an initially movable token at y=56. A hidden empty slot exists under each tray token so the token can leave the tray.

Only initially movable tray tokens are clickable at level start. Fixed tokens already inside components are part of the puzzle layout and are not initially selectable. Once a tray token has been placed into a component, it remains movable by selecting it again.

Click behavior:

1. Click a movable literal or pointer to select it.
2. Click an empty component slot to move the selected token there.
3. Click another movable token on the board to swap with it.
4. Click another tray token while holding a tray token to change the selection.
5. Click the selected token again to cancel selection.
6. Press undo to restore the last saved placement state.

The layouts below use `_` for an empty slot, `lit color` for a solid literal, and `ptr color` for a hollow pointer to the component with that border color.

## Level 1

This level teaches direct literal placement. There is a single red main component with four empty slots and no pointers.

```text
Targets: blue, green, yellow, purple
Target positions: (18,1), (25,1), (32,1), (39,1)

Components:
red main @ (18,25): [_, _, _, _]

Tray, left to right:
lit green, lit purple, lit blue, lit yellow
```

To win, place the literals in the main component in target order:

```text
red main: [lit blue, lit green, lit yellow, lit purple]
```

Pressing space then scans the red component left to right. Each literal fills the corresponding target box. Any missing slot or wrong order fails the check and returns to edit mode.

## Level 2

This level introduces a fixed pointer and a subcomponent. The red main component has a fixed green pointer in its third slot. A green visual connector runs from that pointer down to the green component; it is only a hint and has no logic of its own.

```text
Targets: orange, purple, red, blue, green, yellow, pink
Target positions: (8,1), (15,1), (22,1), (29,1), (36,1), (43,1), (50,1)

Components:
red main @ (18,18): [_, _, ptr green fixed, _]
green component @ (18,32): [_, _, _, _]

Tray, left to right:
lit red, lit purple, lit green, lit orange, lit pink, lit blue, lit yellow
```

The red program emits its first two slots, calls the green component, then emits its fourth slot. The green component must therefore supply target positions 3 through 6.

Winning layout:

```text
red main:       [lit orange, lit purple, ptr green fixed, lit pink]
green component:[lit red, lit blue, lit green, lit yellow]
```

Execution order is `orange, purple`, then the four green-component literals `red, blue, green, yellow`, then `pink`.

## Level 3

This level removes the connector hint and adds two separate subcomponents. The main component has fixed pointers to the green and blue components.

```text
Targets: red, green, purple, yellow, pink, blue, orange
Target positions: (8,1), (15,1), (22,1), (29,1), (36,1), (43,1), (50,1)

Components:
red main @ (15,19): [_, ptr green fixed, _, ptr blue fixed, _]
green component @ (15,31): [_, _]
blue component @ (33,31): [_, _]

Tray, left to right:
lit orange, lit green, lit purple, lit yellow, lit blue, lit pink, lit red
```

The main component emits one literal, calls green for two outputs, emits one literal, calls blue for two outputs, then emits one final literal.

Winning layout:

```text
red main:        [lit red, ptr green fixed, lit yellow, ptr blue fixed, lit orange]
green component: [lit green, lit purple]
blue component:  [lit pink, lit blue]
```

The key reconstruction detail is that pointer color, not spatial connection, selects the subcomponent.

## Level 4

This level reverses the previous setup. The subcomponent already exists, but the player must place the pointer into the main component. The green component has a fixed green literal in its first slot.

```text
Targets: yellow, red, green, blue, pink, orange, purple
Target positions: (8,1), (15,1), (22,1), (29,1), (36,1), (43,1), (50,1)

Components:
red main @ (15,18): [_, _, _, _, _]
green component @ (21,32): [lit green fixed, _, _]

Tray, left to right:
lit yellow, lit pink, lit orange, lit red, lit purple, lit blue, ptr green
```

The target contains the three-color block `green, blue, pink`, so the green pointer belongs in the third main slot.

Winning layout:

```text
red main:        [lit yellow, lit red, ptr green, lit orange, lit purple]
green component: [lit green fixed, lit blue, lit pink]
```

The green pointer expands into the green component's three literals, producing the middle of the target sequence.

## Level 5

This level requires using the same subcomponent twice. The blue component is empty, and the tray contains two blue pointer tokens.

```text
Targets: pink, green, red, red, green, red, red, yellow, purple
Target positions: (1,1), (8,1), (15,1), (22,1), (29,1), (36,1), (43,1), (50,1), (57,1)

Components:
red main @ (15,18): [_, _, _, _, _]
blue component @ (21,32): [_, _, _]

Tray, left to right:
lit purple, lit pink, lit red, lit red, lit yellow, lit green, ptr blue, ptr blue
```

The repeated target block is `green, red, red`. Build that block once in the blue component and call it twice from the main component.

Winning layout:

```text
red main:       [lit pink, ptr blue, ptr blue, lit yellow, lit purple]
blue component: [lit green, lit red, lit red]
```

The first blue pointer emits target positions 2 through 4, and the second blue pointer emits target positions 5 through 7.

## Level 6

This level uses three subcomponents. Each subcomponent has its first literal fixed, so the player completes the repeated-color tails and places pointers into the red main component.

```text
Targets: blue, yellow, yellow, orange, purple, purple, green, pink, pink
Target positions: (1,1), (8,1), (15,1), (22,1), (29,1), (36,1), (43,1), (50,1), (57,1)

Components:
red main @ (8,18): [_, _, _]
green component @ (34,18): [lit green fixed, _, _]
blue component @ (8,32): [lit blue fixed, _, _]
orange component @ (34,32): [lit orange fixed, _, _]

Tray, left to right:
lit yellow, lit pink, lit purple, lit yellow, lit pink, lit purple, ptr green, ptr blue, ptr orange
```

The target decomposes into three three-color blocks:

```text
blue, yellow, yellow
orange, purple, purple
green, pink, pink
```

Winning layout:

```text
red main:         [ptr blue, ptr orange, ptr green]
blue component:   [lit blue fixed, lit yellow, lit yellow]
orange component: [lit orange fixed, lit purple, lit purple]
green component:  [lit green fixed, lit pink, lit pink]
```

The red component does not emit any literal directly in this solution; it is entirely a dispatcher over the three subcomponents.

## Level 7

This level requires a nested call stack. The red main component can call blue, and blue can call green. The green component has a fixed yellow literal in the middle, making it the center of the target palindrome.

```text
Targets: red, blue, green, yellow, green, blue, red
Target positions: (8,1), (15,1), (22,1), (29,1), (36,1), (43,1), (50,1)

Components:
red main @ (21,12): [_, _, _]
blue component @ (21,25): [_, _, _]
green component @ (21,38): [_, lit yellow fixed, _]

Tray, left to right:
lit red, lit blue, lit red, lit green, lit blue, lit green, ptr green, ptr blue
```

The intended structure is `red, call blue, red`. The blue component similarly becomes `blue, call green, blue`, and green emits `green, yellow, green`.

Winning layout:

```text
red main:        [lit red, ptr blue, lit red]
blue component:  [lit blue, ptr green, lit blue]
green component: [lit green, lit yellow fixed, lit green]
```

Execution produces `red`, enters blue, produces `blue`, enters green, produces `green, yellow, green`, returns to blue for the second `blue`, then returns to red for the final `red`.

## Level 8

This level adds self-recursion over the main component. The target row is repeated twice, and the only way to produce the full sequence is to call the red main component again after producing the first six colors.

```text
Targets, row-major:
red, yellow, orange, blue, green, purple,
red, yellow, orange, blue, green, purple

Target positions:
first row:  (11,1), (18,1), (25,1), (32,1), (39,1), (46,1)
second row: (11,8), (18,8), (25,8), (32,8), (39,8), (46,8)

Components:
red main @ (18,22): [_, _, _, _]
blue component @ (18,36): [_, _, _, _]

Tray, left to right:
lit purple, lit yellow, lit orange, lit red, lit blue, lit green, ptr red, ptr blue
```

Build the red main component to emit the first three colors, then call blue. Build the blue component to emit the next three colors, then call red to restart the main component.

Winning layout:

```text
red main:       [lit red, lit yellow, lit orange, ptr blue]
blue component: [lit blue, lit green, lit purple, ptr red]
```

The first red pass emits `red, yellow, orange`, then the blue call emits `blue, green, purple`. The blue component's red pointer calls the red main component again, restarting it at slot 1. The second pass emits the same six-color sequence. After the second `purple` fills the last target box, the level is complete before the checker needs to process another recursive red call.

For reconstruction, recursion must be bounded by target completion and must reject zero-progress loops. A pointer cycle that repeatedly calls components before emitting any matching literal should fail instead of running forever.
