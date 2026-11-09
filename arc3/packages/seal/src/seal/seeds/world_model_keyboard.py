"""World model for the square-recolouring game.

One square sits in the middle of the board. LEFT darkens it by one colour
index and RIGHT lightens it by one; the level is won when the square reaches
colour 15 and lost when it reaches colour 0. Every move spends one unit of
energy, drawn as the bar along the bottom, and the level is also lost when
the energy runs out.

UP has not been played yet, and the small marker in the top-right corner
changes in a way no rule found so far explains.
"""

import numpy as np

BACKGROUND = 0
WIN_COLOUR = 15
LOSE_COLOUR = 0

# Between the marker and the energy bar the board holds only the square, so
# the square's cells and starting colour are read off the level's first frame.
_first = history[0].frame.grid
PLAY_ROWS = slice(8, 56)
SQUARE = np.zeros((64, 64), dtype=bool)
SQUARE[PLAY_ROWS] = _first[PLAY_ROWS] != BACKGROUND
START_COLOUR = int(_first[SQUARE][0])

MARKER = (slice(2, 6), slice(58, 62))

# The bar is 48 cells wide and loses two cells per move: 24 units of energy.
BAR_ROWS = slice(60, 62)
BAR_LEFT = 8
BAR_WIDTH = 48
BAR_TRACK = 3
BAR_FILL = 11
STARTING_ENERGY = 24


def initial_state():
    return {'outcome': 'playing', 'energy': STARTING_ENERGY, 'colour': START_COLOUR}


def do_move(state, move):
    if move.name == 'LEFT':
        state['colour'] -= 1
    elif move.name == 'RIGHT':
        state['colour'] += 1
    else:
        raise Unknown(f'{move.name} has not been played yet')

    state['energy'] -= 1
    if state['colour'] == WIN_COLOUR:
        state['outcome'] = 'won'
    elif state['colour'] == LOSE_COLOUR or state['energy'] <= 0:
        state['outcome'] = 'lost'


def distance_to_win(state):
    return WIN_COLOUR - state['colour']


def render(state):
    board = np.full((64, 64), BACKGROUND, dtype=np.int8)
    board[SQUARE] = state['colour']
    board[MARKER] = -1
    board[BAR_ROWS, BAR_LEFT:BAR_LEFT + BAR_WIDTH] = BAR_TRACK
    fill = round(BAR_WIDTH * state['energy'] / STARTING_ENERGY)
    board[BAR_ROWS, BAR_LEFT:BAR_LEFT + fill] = BAR_FILL
    return board
