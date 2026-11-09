"""World model for the tile-flipping game.

Twelve 16x16 tiles fill the top 48 rows of the board, three rows of four.
Each tile is a body inside a one-cell dark border, and the body is either
grey or yellow; clicking anywhere on a tile, border included, flips its body
to the other colour. The level is won once every body is yellow. Every click
spends one unit of energy, drawn as the bar along the bottom, and the level
is lost when the energy runs out.

Clicks below the tiles have not been tried yet.
"""

import numpy as np

BACKGROUND = 0
BORDER = 4
GREY = 5
YELLOW = 11

TILE = 16
TILE_ROWS = 3
TILE_COLS = 4

# Each tile's starting body colour, sampled from the middle of its body in the
# level's first frame.
_first = history[0].frame.grid
START_BODIES = np.array(
    [[_first[TILE * i + TILE // 2, TILE * j + TILE // 2] for j in range(TILE_COLS)] for i in range(TILE_ROWS)],
    dtype=np.int8,
)

# The bar is 32 cells wide and loses two cells per click: 16 units of energy.
BAR_ROWS = slice(60, 62)
BAR_LEFT = 16
BAR_WIDTH = 32
BAR_TRACK = 3
BAR_FILL = 11
STARTING_ENERGY = 16


def initial_state():
    # START_BODIES is read-only, as all module-level data is, so the state
    # holds its own copy for do_move to change.
    return {'outcome': 'playing', 'energy': STARTING_ENERGY, 'bodies': START_BODIES.copy()}


def do_move(state, move):
    if move.name != 'MOUSE':
        raise Unknown(f'{move.name} has not been played yet')
    i, j = move.row // TILE, move.col // TILE
    if i >= TILE_ROWS:
        raise Unknown(f'no click below the tiles has been tried (row {move.row})')

    bodies = state['bodies']
    bodies[i, j] = YELLOW if bodies[i, j] == GREY else GREY
    state['energy'] -= 1
    if (bodies == YELLOW).all():
        state['outcome'] = 'won'
    elif state['energy'] <= 0:
        state['outcome'] = 'lost'


def distance_to_win(state):
    return int((state['bodies'] == GREY).sum())


def render(state):
    board = np.full((64, 64), BACKGROUND, dtype=np.int8)
    for i in range(TILE_ROWS):
        for j in range(TILE_COLS):
            top, left = TILE * i, TILE * j
            board[top:top + TILE, left:left + TILE] = BORDER
            board[top + 1:top + TILE - 1, left + 1:left + TILE - 1] = state['bodies'][i, j]
    board[BAR_ROWS, BAR_LEFT:BAR_LEFT + BAR_WIDTH] = BAR_TRACK
    fill = round(BAR_WIDTH * state['energy'] / STARTING_ENERGY)
    board[BAR_ROWS, BAR_LEFT:BAR_LEFT + fill] = BAR_FILL
    return board
