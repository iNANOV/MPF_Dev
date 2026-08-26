from pathlib import Path
import json
import uuid
from datetime import datetime, timezone

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
START_DATE = "2009-01-01"


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
CALIBRATION_WINDOW = 6


# How far the test window moves forward.
#
# 6 means move forward by 6 months.
MOVING_PARAM = 12


# Length of each out-of-sample test period.
#
# 6 means each test period is 6 months.
TEST_SPAN = 12


# ============================================================
# MONTE CARLO OPTIMIZATION
# ============================================================

# ============================================================
# WALK-FORWARD MONTE-CARLO SETTINGS
# ============================================================

# Stage 1:
# Broad Monte-Carlo search over the complete threshold space.
N_SIMULATIONS_STAGE1 = 10

# Stage 2:
# Focused Monte-Carlo search around the promising region
# identified by Stage 1.
N_SIMULATIONS_STAGE2 = 10


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
MAX_NUM_COMPONENTS = 3


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
MAX_WORKERS = 2
#
# can be useful if the machine has many CPUs but you don't
# want the optimization to consume all of them.
# MAX_WORKERS = None


# ============================================================
# OUTPUT  (unique run ID + JSON log)
# ============================================================

RUN_ID = uuid.uuid4().hex[:12]                     # e.g. "a3f9c2e1b8d4"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

BASE_NAME = (
    f"wf_{STRATEGY}"
    f"_cal{CALIBRATION_WINDOW}"
    f"_move{MOVING_PARAM}"
    f"_test{TEST_SPAN}"
    f"_ncomp{MIN_NUM_COMPONENTS}-{MAX_NUM_COMPONENTS}"
    f"_{TIMESTAMP}"
    f"_{RUN_ID}"
)

OUTPUT_FILE = RESULTS_DIR / f"{BASE_NAME}.csv"
CONFIG_LOG  = RESULTS_DIR / f"{BASE_NAME}.json"


def save_config_log(path: Path = CONFIG_LOG) -> None:
    """
    Dump every relevant parameter (everything except pure PROJECT PATHS).
    """
    config = {
        # meta
        "run_id": RUN_ID,
        "timestamp_utc": TIMESTAMP,
        "output_file": OUTPUT_FILE.name,

        # strategy
        "STRATEGY": STRATEGY,
        "MA_COLUMN": MA_COLUMN,
        "RANKING_COLUMN": RANKING_COLUMN,

        # walk-forward
        "START_DATE": START_DATE,
        "CALIBRATION_WINDOW": CALIBRATION_WINDOW,
        "MOVING_PARAM": MOVING_PARAM,
        "TEST_SPAN": TEST_SPAN,

        # Monte-Carlo
        "N_SIMULATIONS_STAGE1": N_SIMULATIONS_STAGE1,
        "N_SIMULATIONS_STAGE2": N_SIMULATIONS_STAGE2,
        "THRESHOLD_STEP": THRESHOLD_STEP,
        "RANDOM_STATE": RANDOM_STATE,

        # portfolio size
        "MIN_NUM_COMPONENTS": MIN_NUM_COMPONENTS,
        "MAX_NUM_COMPONENTS": MAX_NUM_COMPONENTS,

        # portfolio
        "INITIAL_CAPITAL": INITIAL_CAPITAL,
        "ABS_COST_FOR_A_TRADE": ABS_COST_FOR_A_TRADE,
        "PERCENT_COST_FOR_A_TRADE": PERCENT_COST_FOR_A_TRADE,
        "MAX_INVESTMENT_SIZE_IN_PERCENT": MAX_INVESTMENT_SIZE_IN_PERCENT,
        "MIN_CASH_IN_PERCENT": MIN_CASH_IN_PERCENT,

        # parallelization
        "MAX_WORKERS": MAX_WORKERS,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Config log written → {path}")