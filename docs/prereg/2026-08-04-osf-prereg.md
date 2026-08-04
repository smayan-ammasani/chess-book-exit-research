**Registered:** https://osf.io/n6tkx/
**Registration date:** 2026-08-04

## METADATA

### Description

An archival study of decision-time costs at the boundary of memorized chess
opening theory, using approximately 89 million Lichess Rapid games from
December 2024. The study tests whether leaving the opening book imposes a
measurable increase in deliberation time that survives statistical control for
engine-derived position complexity, and whether that cost declines with player
skill.

This registers a corrected re-analysis. An earlier version of this pipeline was
built and executed between July 20 and July 29, 2026, and produced results for
three hypotheses. Three defects were subsequently identified in that pipeline,
two of which act to inflate the primary effect. All earlier results are
withdrawn. The corrected analyses specified here have not been run. The full
prior history is disclosed under "Foreknowledge."

### Subjects

Social and Behavioral Sciences → Psychology → Cognitive Psychology

### Tags

chess, expertise, decision making, deliberation, reaction time, opening theory,
archival data, Lichess, Stockfish, value of computation

### License

CC0 1.0 Universal

---

## OVERVIEW

### Research questions or hypotheses

**H1 — Book-exit decision cost, net of position complexity.**
Thinking time on a player's first out-of-book move is greater than thinking
time on that player's last in-book move, and this difference remains positive
after statistically controlling for engine-derived position complexity (value
of computation, VOC).
Directional. Supported only if the is_oob coefficient is positive AND
corresponds to Cohen's d >= 0.20.

**H2 — Skill gradient in the cost.**
The VOC-controlled book-exit time cost decreases monotonically across five
Lichess Rapid rating bands, from band_1 (0-1199) through band_5 (2300+).
Directional. Supported only if band medians are ordered as predicted AND
epsilon-squared >= 0.01.

**H3-revised — Immediate execution as a discrete-transition signal.**
In the unfiltered boundary-pair data, the proportion of moves executed at or
below the premove threshold (0.5 s) is higher for last-in-book moves than for
first-out-of-book moves, in every rating band. Rationale: a retrieved move can
be emitted without deliberation; a move requiring search cannot.
Directional. Supported only if the book proportion exceeds the out-of-book
proportion in all five bands, with a ratio of at least 2.0 in each.

The original H3 (ex-Gaussian decomposition predicting tau-dominated change at
the boundary) is withdrawn as untestable on these data. Justification is given
under "Other planned analysis."


### Explanation of foreknowledge and managing unintended influences

This is a corrected re-analysis, not a first analysis, and the prior history is
disclosed in full.

Between 2026-07-20 and 2026-07-29 a complete pipeline was built and run on this
dataset, producing results for three hypotheses. Those hypotheses were recorded
in analysis code committed 2026-07-27, one day AFTER the analysis dataset was
assembled (2026-07-26). They were not pre-registered. No pre-registration
document exists for that work; the project's version control history confirms
this, as the repository contains no such file at any commit. The earlier
results were: H1 beta = 1.3449, Cohen's d = 1.01; H2 Kruskal-Wallis
H(4) = 3751.8, epsilon-squared = 0.007; H3 reported as tau-dominated in higher
rating bands. All three are withdrawn.

On 2026-08-03 three defects were identified:

1. An asymmetric premove exclusion filter. Pairs were excluded when the
   out-of-book move fell at or below the 0.5 s premove threshold, but not when
   the in-book move did. Since the outcome is (out-of-book time minus in-book
   time), this removed observations arguing against a book-exit cost while
   retaining those arguing for it. Diagnostic evidence: 0.00% of out-of-book
   moves fell below 0.5 s in all five bands across 932,933 pairs, versus
   19.96% to 10.82% of in-book moves.

2. Stockfish transposition tables were never cleared between positions
   (ucinewgame was emitted only once), while positions were processed in
   sorted FEN order, which groups near-identical positions adjacently. Each
   depth-1 evaluation therefore ran against a hash populated by the depth-15
   evaluation of a similar position. Measured on 180 sorted opening positions:
   93.3% of depth-1 evaluations and 97.8% of VOC values changed when the hash
   was cleared per position; correlation between contaminated and clean VOC
   r = 0.878, implying roughly 23% of the covariate's control power lost to
   measurement error. Additionally, the engine was configured with more than
   one thread, making evaluations non-deterministic. Both are corrected here.

3. The ex-Gaussian distribution has a structural skewness ceiling of 2.0
   (skew = 2*tau^3 / (sigma^2 + tau^2)^1.5). All ten band-by-condition sample
   skewness values fall between 4.35 and 6.95, so no parameterization of that
   family can describe these data. Method-of-moments estimation clamps at
   parameter floors; maximum likelihood degenerates toward a shifted
   exponential. Window sensitivity was checked at 15 s, 30 s, 60 s and 120 s
   upper cutoffs; the ceiling is exceeded at every admissible cutoff.

Defects 1 and 2 both act in the same direction: to inflate the apparent
book-exit cost, which is the study's primary finding.

Steps taken to limit unintended influence on this plan:

- All inference criteria are stated as effect-size thresholds rather than
  p-values, fixed before the corrected results exist. At n on the order of
  900,000, essentially any non-zero effect attains p < .001, so significance
  carries no information here. The earlier H2 result illustrates the point:
  H(4) = 3751.8 alongside epsilon-squared = 0.007.
- Falsification conditions are stated explicitly and in advance.
- It is stated in advance that the corrected H1 may fall below the threshold,
  and that H2's gradient may be an artifact: the in-book premove rate declined
  monotonically across bands (19.96% to 10.82%), so the defective filter
  removed a different share of each band, in the same direction as H2 predicts.
- H3-revised was formulated after observing in-book premove proportions in the
  filtered sample. The corresponding unfiltered out-of-book proportions have
  not been examined. This is disclosed as a partially informed prediction and
  should be weighted accordingly against H1 and H2.

---

## RESEARCH DESIGN

### Study type

**Non-randomized study**, and **Adjustment or control-based design** if
multiple selections are permitted. Archival observational data with no
assignment; the book-exit association is adjusted for position complexity in
order to isolate a memory-specific component.

### Intention for causal interpretation

**Indirect inference on causal relationship(s).**

The design tests whether the book boundary carries a decision cost independent
of position complexity, but archival play cannot rule out unmeasured
confounders correlated with book exit. Findings inform a causal account without
identifying a causal effect.

### Blinding of experimental treatments

**No blinding is involved.** There are no assigned treatments. Players were
engaged in ordinary online games and were unaware of any study.

### Study design

Within-player paired archival design. For each player in each qualifying game,
two observations are extracted: the last move played while still inside the
opening book, and the first move played after leaving it. These form a matched
pair from the same player in the same game, minutes apart, so the comparison is
within-subject and within-game.

The book boundary is defined empirically by move-sequence frequency in a
separate reference corpus, not by ECO codes. Rating band (five levels) is a
between-subject grouping factor. Position complexity (VOC) enters as a
continuous covariate. Player identity enters as a random intercept.

No randomization, no manipulation, no blinding. All data pre-exist.

### Randomization

None. No randomization is performed at any level. Where subsampling is used to
cap band sizes, it is seeded (RANDOM_SEED = 42) and reproducible.

---

## SAMPLING

### Data collection procedures

Source: the Lichess open game database, monthly PGN dumps, publicly available
at database.lichess.org.

Two non-overlapping months are used, to prevent circularity:

- Reference corpus: June 2024. Used ONLY to build the opening-book frequency
  table. 128,048 games from higher-rated players.
- Analysis corpus: December 2024. Used ONLY for the time-cost analysis.
  Approximately 89 million games parsed.

Defining "book" from the same games used to measure time costs would make a
sequence count as book partly because it appears in the analysis sample. The
six-month separation removes this.

Inclusion criteria (all fixed in config.py, committed 2026-07-11, before any
analysis):

- Rated games only.
- Time control must exactly match one of twelve permitted Rapid controls:
  600+0, 600+5, 600+10, 900+0, 900+10, 900+15, 1200+0, 1200+10, 1500+0,
  1500+10, 1800+0, 1800+20. Rapid is used because Bullet and Blitz feature
  pervasive premoving of book moves, which corrupts clock-derived thinking
  times.
- Game must reach at least 20 plies.
- Both WhiteRatingDiff and BlackRatingDiff headers must be present, used as a
  proxy for established rating reliability, since Glicko-2 rating deviation is
  not consistently exposed in PGN headers.
- Clock annotations (%clk) must be present for thinking-time derivation.
- The player must actually exit the book within the first 40 plies.

Opening book construction: move-sequence prefixes are counted in the reference
corpus up to 40 plies. A sequence is in book if it appears at least 10 times
(PRIMARY_BOOK_THRESHOLD). Counts down to 3 are retained on disk so thresholds
of 5 and 20 can be applied as robustness checks without rebuilding.

No new data will be collected. The corpus is fixed, public, and complete.

### Sample size

Unit of analysis: one boundary pair (one player, one game).

- Games parsed: approximately 89 million.
- Boundary pairs extracted: approximately 22.8 million.
- Unique board positions requiring engine evaluation: 114,083.
- Analysis sample: stratified by rating band and capped at 200,000 pairs per
  band. The prior (defective) build yielded 932,933 pairs — 200,000 each in
  bands 1 through 4 and 132,933 in band 5, band 5 being capacity-limited
  rather than cap-limited.

The corrected symmetric filter will change these counts, in an unknown
direction and magnitude. Final counts at every stage will be reported.

Nesting: pairs are nested within players; players are nested within rating
bands.

### Sample size rationale

Not determined by power analysis. Sample size is set by the size of the public
corpus after fixed inclusion criteria are applied. Power is not a limiting
factor at this scale — the binding constraint is the reverse, since at
n ≈ 900,000 statistical significance is uninformative. This is precisely why
inference criteria below are stated as effect-size thresholds.

The 200,000-per-band cap exists to prevent the larger bands from dominating
estimation and to keep model fitting tractable, not to reach a target power.

### Starting and stopping rules

No ongoing collection; the archive is static. Processing stops when the
December 2024 dump has been fully parsed (MAX_GAMES_TO_PARSE = None). There is
no interim analysis, no optional stopping, and no decision point at which
additional data could be added.

Analysis begins once all three defect corrections are implemented and Scripts
03 through 05 have been re-run end to end. Results will not be inspected until
the full corrected pipeline has completed.

---

## VARIABLES

### Manipulated variables

None. This is an observational archival study with no random assignment and no
experimental manipulation.

### Measured variables

**Primary outcome.** Thinking time per move, in seconds, derived from
consecutive %clk annotations for the same player, accounting for time
increment. Integer-second resolution as recorded by Lichess.

**Primary predictor.** is_oob — binary. 0 = last move inside the opening book;
1 = first move after leaving it.

**Covariate.** Value of computation (VOC), continuous, in centipawns.
Absolute difference between Stockfish evaluations at depth 15 and depth 1,
from White's perspective, capped at ±1000 centipawns to prevent mate-distance
scores from distorting the distribution. Computed with ucinewgame forced before
each position so the transposition table is cleared, and with Threads = 1 for
determinism. Note that Stockfish at depth 1 still performs quiescence search,
so depth 1 approximates minimal deliberation rather than static evaluation.

**Grouping variable.** Rating band, five levels, by Lichess Rapid rating at
game time: band_1 (0-1199), band_2 (1200-1599), band_3 (1600-1999),
band_4 (2000-2299), band_5 (2300+). Lichess Rapid ratings are not comparable
to FIDE or USCF ratings, nor to Lichess ratings in other time controls, and
will not be mapped onto them.

**Random effect.** Player identity (Lichess username).

**Exclusion variables.** Per-move premove flags for both the in-book and
out-of-book move (thinking time <= 0.5 s); Stockfish analysis-failure flag;
per-player count of usable transitions.

### Indices

**Thinking time.** For a player's move at ply p:
thinking_time = clock[p-2] - clock[p] + increment
where clock[] is that player's remaining time. Values below 0 are treated as
missing.

**Book classification.** A move sequence prefix is in book if its frequency in
the June 2024 reference corpus is >= 10. The boundary is the first ply at
which the player's own move sequence falls below that threshold.

**VOC.** voc = |eval_depth15 - eval_depth1|, both from White's perspective,
each clamped to [-1000, +1000] centipawns before differencing. The absolute
difference is invariant to perspective.

**Outcome for modeling.** log(thinking_time). Log transform is required because
raw move times are severely right-skewed (sample skewness 4.35 to 6.95).

**H1 effect size.** Cohen's d computed from the is_oob coefficient of the
mixed-effects model and the residual standard deviation.

**H2 effect size.** Epsilon-squared from the Kruskal-Wallis statistic:
epsilon_squared = (H - k + 1) / (n - k), with k = 5 bands.

**H3-revised.** Per band, the proportion of moves at or below 0.5 s, computed
separately for in-book and out-of-book moves on the UNFILTERED boundary-pair
file, plus their ratio.

---

## ANALYSIS PLAN

### Statistical models

**H1.** Linear mixed-effects model on log thinking time:

    log_thinking_time ~ is_oob + voc + (1 | player)

Fitted with statsmodels MixedLM. Optimizer fallback chain: lbfgs, then powell,
then Nelder-Mead, taking the first that converges. Convergence status will be
reported. If no optimizer converges, the random intercept will be dropped and
an OLS model with cluster-robust standard errors clustered on player will be
reported instead, and this substitution will be stated.

A Wilcoxon signed-rank test on the paired within-player differences is reported
alongside as a distribution-free check. It does not control for VOC and is
secondary.

Model check: the is_oob coefficient will be reported both with and without the
VOC term, so the degree to which complexity control attenuates the effect is
visible rather than implicit.

**H2.** Kruskal-Wallis test across the five rating bands on VOC-residualized
log time differences, with Dunn's post-hoc pairwise comparisons and Holm
correction for the ten pairwise contrasts. Band medians reported with
bootstrap confidence intervals.

**H3-revised.** Two-proportion z-tests per rating band comparing in-book and
out-of-book premove rates, Holm-corrected across the five bands. Ratios
reported per band with confidence intervals.

No three-way interactions are specified. A prior simulation of a fully
interacted specification (relative move position × rating band × VOC) produced
161 fixed-effect parameters, which is an interpretability problem regardless of
convergence, and it is not part of this plan.

### Transformations

- Thinking time is log-transformed for all modeling (LOG_TRANSFORM = True).
  Given a 0.5 s exclusion floor, no zero-handling constant is required.
- is_oob is coded 0 (in book) / 1 (out of book).
- Rating band is treated as an ordered categorical variable. Where a reference
  level is required, band_3 (1600-1999) is used as the middle band.
- VOC enters untransformed and uncentered in the primary model. A
  specification with VOC mean-centered will be reported as a robustness check.
- No trimming or winsorizing of the outcome beyond the stated inclusion window.

### Inference criteria

**All primary criteria are effect sizes, not p-values.** At n ≈ 900,000,
p-values are uninformative; the earlier H2 result (H(4) = 3751.8 alongside
epsilon-squared = 0.007) demonstrates this directly. P-values will be reported
for completeness but are not decision rules.

- **H1 supported** if the is_oob coefficient is positive AND corresponds to
  Cohen's d >= 0.20, in the model that includes VOC.
- **H2 supported** if band medians are monotonically ordered in the predicted
  direction AND epsilon-squared >= 0.01.
- **H3-revised supported** if the in-book premove proportion exceeds the
  out-of-book proportion in all five bands AND the ratio is at least 2.0 in
  every band.

All tests are one-tailed, as all three hypotheses are directional. Multiple
comparisons: Holm correction within H2's pairwise contrasts and within
H3-revised's five band comparisons. The three hypotheses are treated as
separate confirmatory tests and are not corrected against one another; this is
stated so readers can apply a stricter correction if they prefer.

**Falsification.** If H1's coefficient falls below d = 0.20 after VOC control
with the corrected filter and corrected VOC, the study's central claim is not
supported. That outcome will be reported as the result and will not be
reframed as a smaller positive finding, nor replaced by a post-hoc alternative
specification.

### Data inclusion and exclusion

Applied in this order:

1. Games failing any inclusion criterion under "Data collection procedures"
   are excluded.
2. Players contributing fewer than 3 usable book-exit transitions are excluded
   (MIN_TRANSITIONS_PER_PLAYER = 3), as a single observation cannot support a
   random intercept.
3. **A boundary pair is excluded if EITHER the last-in-book move OR the
   first-out-of-book move is at or below the 0.5 s premove threshold.** This
   is the corrected symmetric criterion. The prior filter tested the
   out-of-book move only and is the reason for this registration.
4. Positions where Stockfish analysis fails are excluded and counted. They are
   NOT recoded to VOC = 0, which would misrepresent an analysis failure as a
   trivially simple position. Failure counts will be reported.
5. Terminal positions (checkmate, stalemate) are excluded.
6. For distributional analyses, a fitting window of [0.5 s, 120 s] applies.
   Window sensitivity at 15 s, 30 s and 60 s will be reported.

No outlier removal beyond the above. No observation is excluded on the basis of
its effect on results.

Exclusion counts at every stage will be reported, along with the resulting N.

### Missing data

All models are fitted on complete cases; no imputation is performed.

Missingness sources, each counted and reported: games lacking %clk
annotations; games lacking RatingDiff headers; games not reaching 40 plies
without a book exit; negative or non-computable thinking times; Stockfish
analysis failures.

Missingness is not expected to be uniform across rating bands — the in-book
premove rate itself varies by band — so per-band missingness will be reported
separately rather than pooled, and any band-differential pattern will be
discussed as a limitation.

### Other planned analysis

**Robustness checks, specified in advance:**

1. Book frequency threshold at 5 and 20, alongside the primary value of 10.
2. Direct comparison of contaminated versus cleared-hash VOC across all
   114,083 unique positions, quantifying on this dataset what a 180-position
   test estimated at r = 0.878.
3. H1 re-estimated excluding all premove-adjacent pairs entirely, not merely
   the flagged moves.
4. Sensitivity of H1 to the log-transform choice.
5. H1 and H2 re-estimated with VOC mean-centered.

**Withdrawn analysis, reported as a negative methodological result.**
The original H3 predicted that book exit would produce a tau-dominated change
in an ex-Gaussian decomposition of the response distribution, indicating a
discrete transition rather than a continuous shift. This cannot be tested. The
ex-Gaussian's skewness is bounded above by 2.0 for all parameter values, while
sample skewness in these data ranges from 4.35 to 6.95 across all ten
band-by-condition cells. Method-of-moments estimation clamps at parameter
floors; maximum likelihood degenerates toward a shifted exponential. The
finding that a widely used response-time model is inadmissible for chess clock
data at this timescale will be reported as a methodological result in its own
right.

**Exploratory, clearly labeled as such and not confirmatory.**
A transition-curve extension using relative move indexing around the book
boundary (window of -5 to +10 plies), with categorical fixed effects for
boundary position, recovery-slope and area-under-curve metrics, and opening
family as an additional random effect. This is exploratory, was formulated
after observing earlier results, requires re-running the parsing stage, and
does not substitute for H1 under any outcome.

---

## OTHER

### Context and additional information

This project is conducted independently by a secondary school student, without
institutional supervision or laboratory affiliation. Publication and
dissemination venues have not been determined.

The full pipeline is version-controlled at the project repository. Code state
at registration: commit 8dcaa04. Each defect correction will be a separate
commit, referenced by hash in the manuscript, and the diagnostic that found
the defects was committed before any of the fixes so the sequence is visible
in the history.

The closest prior work is Russek et al. (2025, Cognitive Science), which
established that players spend more time in positions where additional
computation is valuable, across more than 12 million Lichess games, using a
continuous value-of-computation signal. The present study differs by
introducing a discrete, empirically defined book boundary as the independent
variable and testing whether a decision cost survives control for the
continuous complexity signal that work established. Chassy and Gobet (2011,
PLOS ONE) estimated the depth of memorized preparation by skill level but did
not measure time costs at the point of departure.

All earlier results from this project are withdrawn and are retained in version
control rather than deleted.
