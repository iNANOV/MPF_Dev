from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_FILE = DATA_DIR / "data_df.parquet"
MEMBERSHIP_FILE = DATA_DIR / "dow_membership.csv"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# STRATEGY
# ============================================================

STRATEGY = "mean_reversion"

# Column containing the signal variable.
#
# Example:
# AAPL_Z_excess_ma63
#
# The {} will be replaced by the component ticker.
MA_COLUMN = "{}_Z_excess_ma63"

# Column used to rank components when several signals
# occur on the same day.
RANKING_COLUMN = "{}_mean_sd"


# ============================================================
# WALK-FORWARD SETUP
# ============================================================

# First date from which the walk-forward procedure starts.
#
# Everything before:
#
#     START_DATE + CALIBRATION_WINDOW
#
# is available for the first optimization.
START_DATE = "2010-01-01"


# Historical calibration window in years.
#
# Example:
#
# START_DATE = 2010-01-01
# WINDOW      = 4
#
# First optimization:
# historical data -> 2014-01-01
#
CALIBRATION_WINDOW = 4


# How far the test window moves forward.
#
# 6 means move forward by 6 months.
MOVING_PARAM = 6


# Length of each out-of-sample test period.
#
# 6 means each test period is 6 months.
TEST_SPAN = 6


# ============================================================
# MONTE CARLO OPTIMIZATION
# ============================================================

# Number of random threshold combinations tested
# in every calibration period.
N_SIMULATIONS = 50


# Threshold resolution.
#
# 0.01 means:
#
# -0.30
# -0.29
# -0.28
# ...
#  0.00
# ...
#  0.28
#  0.29
#  0.30
#
THRESHOLD_STEP = 0.01


# Random seed.
#
# Keeping this fixed makes the experiment reproducible.
RANDOM_STATE = 42


# ============================================================
# PORTFOLIO SIZE OPTIMIZATION
# ============================================================

# Test every value:
#
# 2, 3, 4, ..., 20
#
MIN_NUM_COMPONENTS = 2
MAX_NUM_COMPONENTS = 4


# ============================================================
# PORTFOLIO
# ============================================================

INITIAL_CAPITAL = 100_000

ABS_COST_FOR_A_TRADE = 5

PERCENT_COST_FOR_A_TRADE = 0.001

MAX_INVESTMENT_SIZE_IN_PERCENT = 50

MIN_CASH_IN_PERCENT = 10


# ============================================================
# PARALLELIZATION
# ============================================================

# None = automatically use available CPUs.
#
# Example:
#
MAX_WORKERS = 20
#
# can be useful if the machine has many CPUs but you don't
# want the optimization to consume all of them.
# MAX_WORKERS = None


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = (
    RESULTS_DIR /
    "walk_forward_results.csv"
)