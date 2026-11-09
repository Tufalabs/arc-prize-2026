# Translate

## Core Game

Translate is a symbolic rule-matching puzzle on a 64x64 grid. The screen is split into a light upper rule area and a darker lower answer area. The upper area shows translation rules as colored symbol sequences joined by thin gray horizontal bars. The lower area shows a source sequence on the upper row and either an editable answer sequence or, in later levels, a fixed target graphic.

The player uses the cursor brackets to select a symbol slot or a whole rule-side sequence. Left/right moves the cursor through selectable items, wrapping at the ends. Up/down cycles the selected symbol variant backward or forward through the seven variants in that symbol's color family. In levels where a whole sequence is selected, every symbol in that selected sequence cycles together by the same amount, preserving the sequence length and relative offsets.

Each player action costs one unit of the action budget. Levels 1-5 start with 128 actions; level 6 starts with 256. The last rendered row is a budget meter: filled cells show remaining budget, and an empty meter loses the attempt. Resetting the level restores the original starting symbols, cursor, rotations, and budget.

The player wins when the current configuration satisfies the translation rules. Validation ignores the visual rotation of a glyph and compares only symbol identity, such as `A4` or `C2`. Rotations are therefore a deliberate visual distraction, especially after level 1.

After a valid configuration is reached, the cursor disappears and the game plays a success animation. The matched source and output segments are recolored through a fixed color cycle, and the rule symbols involved in the proof are backed by black 9x9 highlight squares. When the animation has passed over every matched segment, the next level begins.

## Symbol Model

Every tile has a 7x7 colored background and a 5x5 black glyph placed at offset `(1, 1)` inside it. Adjacent symbols in a sequence are spaced 7 pixels apart, so their colored backgrounds touch edge-to-edge. A rule connector is a 1-pixel-high gray bar running from the center of the final left-side tile to the center of the first right-side tile.

The lower answer panel is a dark 64-pixel-wide band. Its top edge is at `y=34` for levels 1-3, `y=37` for level 4, `y=38` for level 5, and `y=41` for level 6. The selected item is marked by two white bracket caps: one above the selected sequence and one below it. A one-symbol selection uses a 5-pixel-wide cap, a two-symbol selection uses a 12-pixel-wide cap, and a three-symbol selection uses a 19-pixel-wide cap. The top cap sits 4 pixels above the selected glyph top edge; the bottom cap is the same cap rotated 180 degrees and placed just below the selected glyphs.

The three tile families are:

- `A`: cyan/blue background.
- `B`: magenta/pink background.
- `C`: yellow background.

The family letter is fixed by the tile background. Cycling a tile changes only the numeric glyph variant, wrapping `1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 1` or the reverse direction. It never changes the family color.

Glyph masks use `X` for black pixels and `.` for transparent pixels.

```text
A1       A2       A3       A4       A5       A6       A7
..X..    ....X    XXXXX    ..X..    X...X    XXXXX    X...X
..X..    ..X.X    X.X..    XXXXX    XXXXX    ..X..    XXXXX
.XXX.    XXXXX    X....    X.X.X    ..X..    XXX..    X...X
..X..    X.X..    X.X..    X.X.X    XXXXX    X....    X...X
XXXXX    X....    XXXXX    ..X..    X...X    XXXXX    XX.XX

B1       B2       B3       B4       B5       B6       B7
X....    XXXXX    ..XXX    .XXX.    XXXXX    XXXX.    ..X..
XXXX.    X...X    ..X.X    .X.X.    X...X    X..XX    XXXXX
X..X.    XXX.X    XXXXX    XXXXX    X...X    X...X    X.X.X
X..X.    X.X.X    X.X..    X...X    XXXXX    XX..X    XXXXX
XXXXX    XXXXX    XXX..    XXXXX    ..X..    .XXXX    ..X..

C1       C2       C3       C4       C5       C6       C7
X.X.X    XXX.X    X.XXX    ..X..    XX.XX    X.XXX    X...X
X....    X....    ..X..    X.X.X    X...X    X.X.X    .....
X.X.X    .....    ..X..    ..X..    X.X.X    XXX.X    X...X
X....    ....X    ..X..    X.X.X    X...X    .....    XX.XX
XXXXX    X.XXX    XXX.X    ..X..    XX.XX    X.X.X    .X.X.
```

For an exact reproduction of the official starts, assign each glyph a visual rotation of `0`, `90`, `180`, or `270` degrees at level load. Sort all glyph sprites by `(y, x)` and draw rotations from a deterministic per-level random stream with seeds `4, 2, 1, 2, 12, 20` for levels 1-6. These rotations do not affect matching.

## Validation Rules

In ordinary levels, the upper rule table is fixed and the lower target row is editable. The validator scans the lower source row from left to right. At each source position, it finds the first displayed rule whose left sequence exactly matches the source symbols at that position, appends that rule's right sequence to the expected target output, advances by the matched left-sequence length, and continues. The level is solved only if the complete expected target sequence exactly equals the lower editable row. The check runs after a symbol-cycling action; moving the cursor alone only changes selection and spends budget.

Level 4 uses two-hop translation. For each source segment, first apply a rule from the source color to an intermediate sequence. Then find a second rule whose left sequence is the intermediate sequence, and use that second rule's right sequence as the final output. If either hop is missing, the configuration is not valid.

Levels 5 and 6 use rule editing. The lower graphic is fixed, and the cursor selects rule sides in this order: rule 1 left, rule 1 right, rule 2 left, rule 2 right, and so on. Up/down cycles every symbol in the selected side together. In level 5, validation is the ordinary one-hop scan. In level 6, validation is tree-style: apply an `A -> B...` rule to a source symbol, then translate each resulting `B` symbol independently through a matching `B -> C` rule, concatenating the `C` outputs. This is why level 6 can have multiple valid rule configurations.

Coordinates below are glyph top-left positions. The colored background for each listed glyph is at `(x - 1, y - 1)`.

## Level 1

Level 1 teaches direct one-to-one translation. The cursor starts on the first lower magenta target slot. Only the lower target row is editable.

Rule layout:

```text
R1  (13,  5) A3 -> (23,  5) B6
R2  (36,  5) A5 -> (46,  5) B5
R3  (13, 14) A1 -> (23, 14) B1
R4  (36, 14) A4 -> (46, 14) B3
R5  (13, 23) A2 -> (23, 23) B2
R6  (36, 23) A7 -> (46, 23) B7
```

Lower rows at start:

```text
Source y=41: A4 A2 A3 A5 A1
Target y=52: B1 B7 B2 B4 B6
```

To win, translate each cyan source symbol through the matching rule:

```text
A4 -> B3
A2 -> B2
A3 -> B6
A5 -> B5
A1 -> B1
```

The lower target row must therefore be:

```text
B3 B2 B6 B5 B1
```

All target slots are single-symbol slots, so the player can solve the level by moving the cursor across the five magenta cells and cycling each one independently. If any target cell is the wrong `B` variant, the validator rejects the sequence and play continues.

## Level 2

Level 2 keeps the editable lower target row but introduces one-to-many rules. Some single magenta source symbols expand into two or three yellow output symbols. Rotations no longer line up reliably between examples, so the player must recognize glyph identity independent of rotation.

Rule layout:

```text
R1  ( 6,  5) B1 -> (16,  5) C3
R2  (29,  5) B4 -> (39,  5) C4 (46,  5) C3 (53,  5) C6
R3  ( 6, 14) B6 -> (16, 14) C4 (23, 14) C2
R4  (36, 14) B5 -> (46, 14) C2 (53, 14) C2
R5  ( 6, 23) B3 -> (16, 23) C1 (23, 23) C5 (30, 23) C1
R6  (43, 23) B7 -> (53, 23) C7
```

Lower rows at start:

```text
Source y=41: B1 B3 B5 B7
Target y=52: C7 C6 C1 C3 C5 C5 C4
```

To win, expand the source row by rule:

```text
B1 -> C3
B3 -> C1 C5 C1
B5 -> C2 C2
B7 -> C7
```

The lower target row must therefore be:

```text
C3 C1 C5 C1 C2 C2 C7
```

The important reconstruction detail is that target length is not tied to source length. Four source symbols produce seven target symbols, and the cursor moves through all seven yellow target slots.

## Level 3

Level 3 adds many-to-one and many-to-many patterns. A rule's left side can consume a sequence of source symbols, not just one symbol. The lower source row must be parsed left to right into the displayed rule chunks.

Rule layout:

```text
R1  ( 6,  5) C6 -> (16,  5) A4
R2  (29,  5) C3 (36,  5) C3 -> (46,  5) A6 (53,  5) A1
R3  ( 6, 14) C4 -> (16, 14) A7 (23, 14) A7
R4  (36, 14) C7 (43, 14) C7 -> (53, 14) A3
R5  ( 6, 23) C1 (13, 23) C5 (20, 23) C1 -> (30, 23) A6
R6  (43, 23) C2 -> (53, 23) A5
```

Lower rows at start:

```text
Source y=41: C6 C1 C5 C1 C4 C2 C3 C3
Target y=52: A3 A4 A4 A5 A1 A4 A6
```

The source row parses into these chunks:

```text
C6          -> A4
C1 C5 C1    -> A6
C4          -> A7 A7
C2          -> A5
C3 C3       -> A6 A1
```

The lower target row must therefore be:

```text
A4 A6 A7 A7 A5 A6 A1
```

This level should be rebuilt with greedy left-to-right parsing. For example, the `C1 C5 C1` triple is one source chunk that produces one output symbol; those three source cells should not be translated independently.

## Level 4

Level 4 introduces two-hop reasoning. The displayed rules form cyan-to-magenta and magenta-to-yellow links. The lower source row is cyan and the required lower target row is yellow, so each source symbol must be translated through a magenta intermediate symbol before producing the final yellow symbol.

Rule layout:

```text
R1  (13,  4) B7 -> (23,  4) C7
R2  (36,  4) A6 -> (46,  4) B1
R3  (13, 12) A1 -> (23, 12) B3
R4  (36, 12) B6 -> (46, 12) C1
R5  (13, 20) B1 -> (23, 20) C3
R6  (36, 20) A7 -> (46, 20) B6
R7  (13, 28) A4 -> (23, 28) B7
R8  (36, 28) B3 -> (46, 28) C2
```

Lower rows at start:

```text
Source y=41: A6 A1 A4 A7 A1 A6 A4
Target y=52: C6 C5 C3 C1 C7 C6 C6
```

The necessary two-hop chains are:

```text
A6 -> B1 -> C3
A1 -> B3 -> C2
A4 -> B7 -> C7
A7 -> B6 -> C1
```

Applying those chains to the lower source row gives:

```text
A6 A1 A4 A7 A1 A6 A4
 C3 C2 C7 C1 C2 C3 C7
```

The lower target row must therefore be:

```text
C3 C2 C7 C1 C2 C3 C7
```

On success, the proof highlight includes both the intermediate magenta symbol and the second-hop rule that consumes it. This matters visually because the player is meant to see that the answer is not a direct cyan-to-yellow lookup.

## Level 5

Level 5 changes the task from editing the answer to editing the rule table. The lower source and target rows are fixed. The cursor starts on the left side of the first top rule, and each selectable item is a rule side. Selecting a two-symbol side cycles both symbols together.

Initial rule layout:

```text
R1  ( 9, 11) A2 -> (19, 11) B1
R2  (32, 11) A2 -> (42, 11) B6 (49, 11) B6
R3  ( 9, 23) A3 (16, 23) A3 -> (26, 23) B3
R4  (39, 23) A5 -> (49, 23) B2
```

Fixed lower graphic:

```text
Source y=44: A5 A1 A4 A4 A2
Target y=53: B3 B5 B5 B1 B2
```

The target contains a repeated pair `B5 B5`, and the source contains a repeated pair `A4 A4`. The rule table also contains one single-to-double rule and one double-to-single rule. To solve the level, align those structural features with the fixed graphic.

One direct winning rule table is:

```text
R1  A5 -> B3
R2  A1 -> B5 B5
R3  A4 A4 -> B1
R4  A2 -> B2
```

With those rules, the fixed lower rows parse as:

```text
A5       -> B3
A1       -> B5 B5
A4 A4    -> B1
A2       -> B2
```

The concatenated output is exactly:

```text
B3 B5 B5 B1 B2
```

The rule-side cycling constraint is part of the puzzle. For example, `R2` starts as `A2 -> B6 B6`; cycling its left side backward once produces `A1`, and cycling its right side backward once produces `B5 B5`.

## Level 6

Level 6 combines rule editing with tree-style two-stage translation. The fixed lower source row contains three cyan symbols. Each must first translate to a pair of magenta symbols, and each magenta symbol must then translate to one yellow symbol. The fixed lower target row therefore has six yellow symbols.

Initial rule layout:

```text
R1  (10,  6) A3 -> (20,  6) B6 (27,  6) B1
R2  (39,  6) B4 -> (49,  6) C2
R3  (10, 18) A7 -> (20, 18) B5 (27, 18) B3
R4  (39, 18) B5 -> (49, 18) C5
R5  (10, 30) A5 -> (20, 30) B3 (27, 30) B3
R6  (39, 30) B7 -> (49, 30) C6
```

Fixed lower graphic:

```text
Source y=45: A7 A6 A1
Target y=54: C3 C1 C5 C5 C1 C3
```

One winning configuration is:

```text
R1  A1 -> B3 B5
R2  B3 -> C1
R3  A7 -> B5 B3
R4  B5 -> C3
R5  A6 -> B4 B4
R6  B4 -> C5
```

The resulting proof is:

```text
A7 -> B5 B3 -> C3 C1
A6 -> B4 B4 -> C5 C5
A1 -> B3 B5 -> C1 C3
```

Concatenating the three final yellow pairs gives:

```text
C3 C1 C5 C5 C1 C3
```

Multiple solutions are possible because the intermediate magenta names are not fixed by the lower target; only their second-hop yellow outputs matter. The two constraints that must remain true are:

- The three cyan rules must match the fixed source sequence `A7 A6 A1` in left-to-right order.
- The magenta symbols produced by those cyan rules must be covered by magenta-to-yellow rules whose concatenated yellow outputs equal `C3 C1 C5 C5 C1 C3`.

The paired structure is the intended shortcut. The first and last target pairs are reverses of each other, so a clean solution can use reversed magenta pairs for `A7` and `A1`, while the middle source symbol uses a doubled magenta pair to produce `C5 C5`.
