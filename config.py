# config.py
# =============================================================================
# Central configuration for the chess book-exit research pipeline.
#
# ALL methodological parameters live here. Individual scripts import from
# this file — nothing is hardcoded inside a script. This means:
#   - Changing a parameter once changes it everywhere
#   - Your Methods section can point to this file as the authoritative record
#     of every analytic choice used in the analysis
#   - Reviewers can reproduce the analysis by running the pipeline with
#     these exact settings
# =============================================================================

import os

# ── Data acquisition ───────────────────────────────────────────────────────────
#
# ANALYSIS_MONTH: the Lichess data month used for the main analysis.
# REFERENCE_MONTH: a DIFFERENT month used only to build the "book" frequency table.
#
# Why different months? If we defined "book" using the same games we analyze
# for time costs, we'd introduce circular reasoning: a sequence would look
# "book" partly because it appears in our own analysis sample. Using a
# completely separate 6-month-prior dataset eliminates this.
#
ANALYSIS_MONTH  = "2024-12"   # December 2024 — main analysis sample
REFERENCE_MONTH = "2024-06"   # June 2024     — reference corpus (must not overlap)

# ── Time control filter ────────────────────────────────────────────────────────
#
# Only games with these exact time control strings are included.
# Lichess encodes time control as "base_seconds+increment_seconds".
# We restrict to Rapid time controls (roughly 8-30 minutes per player)
# because:
#   - Bullet and Blitz games feature widespread premoving of book moves,
#     corrupting clock-based thinking-time estimates
#   - Rapid gives players enough time that thinking is real and intentional
#   - Rapid is the most common non-blitz format on Lichess, giving large N
#
# Most common Lichess Rapid controls in descending frequency:
ALLOWED_TIME_CONTROLS = [
    "600+0",    # 10 minutes, no increment
    "600+5",    # 10 minutes + 5 sec/move increment
    "600+10",   # 10+10
    "900+0",    # 15 minutes
    "900+10",   # 15+10
    "900+15",   # 15+15
    "1200+0",   # 20 minutes
    "1200+10",  # 20+10
    "1500+0",   # 25 minutes
    "1500+10",  # 25+10
    "1800+0",   # 30 minutes
    "1800+20",  # 30+20
]

# ── Rating bands ───────────────────────────────────────────────────────────────
#
# Players are grouped into five skill tiers by their Lichess Rapid rating
# at the time of the game.
#
# IMPORTANT: Lichess Rapid ratings are NOT equivalent to USCF or FIDE ratings,
# and are NOT comparable to ratings in other Lichess time controls (Blitz,
# Bullet, Classical all have separate rating pools). All results are interpreted
# strictly on the Lichess Rapid scale and are never mapped to OTB ratings.
#
RATING_BANDS = {
    "band_1": (0,    1199),   # Beginner
    "band_2": (1200, 1599),   # Intermediate
    "band_3": (1600, 1999),   # Club player
    "band_4": (2000, 2299),   # Advanced
    "band_5": (2300, 9999),   # Expert / Master level
}

# ── Rating reliability filter ──────────────────────────────────────────────────
#
# Lichess Glicko-2 rating deviation (RD) is not consistently included in PGN
# headers. As a proxy for rating reliability, we require both WhiteRatingDiff
# and BlackRatingDiff fields to be present. These fields appear only in rated
# games where both players have an established rating history. Games missing
# these fields likely involve new accounts with volatile, unreliable ratings.
#
MIN_GAMES_PLAYED_PROXY = True   # If True, exclude games missing RatingDiff headers

# ── Minimum game length ────────────────────────────────────────────────────────
#
# Games must reach at least this many half-moves (plies) to be included.
# 20 half-moves = 10 moves each side. Games shorter than this almost certainly
# never exit the opening book, making them uninformative for our analysis.
#
MIN_PLIES_FOR_INCLUSION = 20

# ── Opening book depth cap ─────────────────────────────────────────────────────
#
# When building the reference corpus, we count move-sequence prefixes only up
# to this depth. Beyond 40 half-moves (20 full moves), even the deepest
# theoretical preparation rarely extends, and counting further adds memory cost
# without analytical value.
#
MAX_PLIES_FOR_BOOK = 40

# ── Book frequency thresholds ──────────────────────────────────────────────────
#
# A move sequence is classified as "in book" if it was played at least N times
# in the reference corpus. This is the key methodological alternative to using
# ECO codes (which stop classifying lines after ~10-20 moves regardless of
# how deeply players actually prepare).
#
# PRIMARY_BOOK_THRESHOLD is used in the main analysis.
# ROBUSTNESS_THRESHOLDS are used in sensitivity checks to confirm the result
# does not depend on the specific cutoff chosen.
#
PRIMARY_BOOK_THRESHOLD  = 10
ROBUSTNESS_THRESHOLDS   = [5, 20]

# The minimum count saved to disk in the reference corpus CSV.
# Must be <= min(ROBUSTNESS_THRESHOLDS) so all thresholds can be applied
# at load time in Script 02 without re-running Script 01.
MIN_COUNT_TO_SAVE = 3

# ── Premove detection ──────────────────────────────────────────────────────────
#
# Lichess allows players to "premove" — queue a move during the opponent's turn
# so it executes instantly when it becomes legal. This is common for well-known
# book moves in Rapid games. A premovedmove registers as near-zero thinking time
# in the clock data, which is a UI artifact, not a cognitive signal.
#
# Moves with thinking time at or below this threshold are flagged as likely
# premoves and excluded from the primary analysis. They are retained in the
# dataset with a boolean flag for transparency.
#
PREMOVE_THRESHOLD_SECONDS = 0.5

# ── Stockfish parameters ───────────────────────────────────────────────────────
#
# Used in Script 03 to compute the "value-of-computation" (VOC) covariate.
# VOC = |eval(depth_deep) − eval(depth_shallow)|
# A large gap means additional thinking would substantially change the evaluation —
# the position objectively rewards deeper calculation.
#
# This is related to, but not the same as, the measure in Russek et al. (2025,
# Cognitive Science), who compared the deep-search value of the best move with
# that of the move a depth-1 search would choose, across 12M+ Lichess games.
# We compute it only at our two boundary positions
# per game (not every move), keeping compute requirements manageable.
#
STOCKFISH_PATH = "./stockfish/stockfish.exe"   # Update if binary is elsewhere
STOCKFISH_DEPTH_SHALLOW = 1
STOCKFISH_DEPTH_DEEP    = 15
STOCKFISH_EVAL_CAP_CP   = 1000    # Cap centipawn scores to avoid mate-distance outliers
STOCKFISH_THREADS = 1   # was max(1, os.cpu_count() - 1); >1 makes Stockfish non-deterministic

# ── Statistical model ──────────────────────────────────────────────────────────
RANDOM_SEED               = 42    # All random operations use this seed (reproducibility)
LOG_TRANSFORM             = True  # Log-transform thinking times before modeling
                                  # (required: move times are right-skewed)
MIN_TRANSITIONS_PER_PLAYER = 3   # Drop players contributing fewer than this many
                                  # usable book-exit transitions (not enough data
                                  # to estimate a reliable random intercept)

# ── Testing / development ──────────────────────────────────────────────────────
#
# Set MAX_GAMES_TO_PARSE to a small integer (e.g., 500_000) to run the pipeline
# on a subset of the data for testing. Set to None to parse the full file.
# ALWAYS set this back to None before running the final analysis.
#
MAX_GAMES_TO_PARSE = None   # None = no limit (full dataset)
