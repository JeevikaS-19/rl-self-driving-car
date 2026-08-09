# Shared grid/ego configuration — single source of truth.
# Update here only; all other files import from this.

GRID_SIZE = 64
MAX_RANGE = 50.0
RESOLUTION = MAX_RANGE / GRID_SIZE  # ~0.78125 m/cell

EGO_ROW = 48   # asymmetric: 48 cells forward, 16 cells behind
EGO_COL = 32   # symmetric left/right

