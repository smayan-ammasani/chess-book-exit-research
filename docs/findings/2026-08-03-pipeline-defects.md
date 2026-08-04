# Findings: three defects found in the analysis pipeline

**Date:** 2026-08-03
**Status:** Found. Not yet fixed.
**Found by:** diagnostic review of Scripts 03, 05, and 06, prompted by implausible
ex-Gaussian output (all fitted parameters landing on their clamp floors).

This note records the defects **before** any of them were corrected, so the
repository history shows what was found, when, and on what evidence.

---

## Defect 1 — Ex-Gaussian method-of-moments formula (Script 06)

**Severity:** Corrected, but see Defect 1b.

Line 177 of `06_exgaussian_fitting.py` read:

```python
tau = max(0.05, (v * max(skew, 0.01) / 2) ** (1/3))
```

The standard Heathcote (1996) / Hohle method-of-moments estimator is

    tau = ( skewness * variance^1.5 / 2 ) ^ (1/3)

i.e. variance raised to the 1.5 power, not the first power. The error is
visible from units alone: tau is measured in seconds and skewness is
dimensionless, so cube-rooting `variance^1 * skewness` yields seconds^(2/3),
which is not a time.

Verified on synthetic data generated from a known ex-Gaussian: with a true
pure-tau effect (tau 2.11 -> 4.26, mu held fixed, true ratio infinite), the
original formula recovered a delta-tau/delta-mu ratio of 0.76 -- which the
script's own thresholds classify as "mixed / continuous gradient", the
opposite of the correct conclusion. The corrected formula recovered 12.6.

The error systematically understated tau-dominance, and grew worse as tau
increased -- that is, worst precisely in the regime H3 was designed to detect.

## Defect 1b — The ex-Gaussian is inadmissible for this data

**Severity:** H3 as pre-registered cannot be tested.

Correcting Defect 1 did not produce usable estimates. Every one of the ten
band x condition fits returned `mu = 0.05` and `sigma = 0.10` -- exactly the
clamp floors in `max(0.05, ...)` and `sqrt(max(0.01, ...))`. The reported
`delta_mu = 0.000` in all five bands was floor minus floor, and the resulting
"infinite" delta-tau/delta-mu ratios were artifacts of dividing by that zero,
not evidence for a discrete switch.

Cause: the ex-Gaussian distribution has a structural skewness ceiling.

    Var  = sigma^2 + tau^2
    Skew = 2*tau^3 / (sigma^2 + tau^2)^1.5

Letting tau/sigma grow without bound drives skewness to 2.0 and no further.
No ex-Gaussian, at any parameter values, can represent skewness >= 2.

Measured sample skewness in `analysis_sample.csv` (see
`outputs/diagnostics/h3_diagnostic_output.txt`):

| band   | book | oob  |
|--------|------|------|
| band_1 | 5.49 | 4.35 |
| band_2 | 6.26 | 4.94 |
| band_3 | 6.87 | 5.43 |
| band_4 | 6.95 | 5.23 |
| band_5 | 6.42 | 4.60 |

All ten conditions exceed the ceiling, by a factor of roughly two to three.
Window sensitivity was checked at 15s, 30s, 60s and 120s upper cutoffs; even
at 15s the book condition remains above the ceiling in bands 3, 4 and 5
(2.11, 2.22, 2.07), so no admissible cutoff rescues the fit.

No estimator recovers this. Method-of-moments clamps; maximum likelihood was
tested on synthetic heavy-tailed data and degenerates differently, pushing
sigma toward 0 and mu toward the truncation point, collapsing the ex-Gaussian
to a shifted exponential.

**Consequence:** H3 cannot be evaluated with the ex-Gaussian decomposition as
pre-registered. This is itself a reportable result: a pre-registered
distributional prediction was tested for admissibility against the data and
the model family was found inadmissible.

---

## Defect 2 — Asymmetric premove filter (Scripts 02 and 04)

**Severity:** Affects H1, the primary confirmatory finding.

The diagnostic reported the proportion of each condition falling below the
0.5-second floor:

| band   | book < 0.5s | oob < 0.5s |
|--------|-------------|------------|
| band_1 | 19.96%      | 0.00%      |
| band_2 | 15.80%      | 0.00%      |
| band_3 | 13.98%      | 0.00%      |
| band_4 | 11.84%      | 0.00%      |
| band_5 | 10.82%      | 0.00%      |

Exactly zero out-of-book moves below 0.5 seconds, in all five bands, across
932,933 pairs. This is not a property of the behaviour -- players do premove
out of book, on forced recaptures and single-legal-reply positions. A true
zero indicates a filter.

Source: `02_parse_and_tag_games.py`, line 489:

```python
premove_suspect = oob_is_premove
```

The flag is raised when the **out-of-book** move is fast. It is not raised
when the book move is fast. Script 04 filters on this flag.

The filter therefore deletes pairs with fast out-of-book moves while retaining
pairs with fast book moves. H1 measures (out-of-book time - book time). The
observations removed are exactly those that would count against a book-exit
cost; the observations retained include those that count for it. The measured
effect is inflated by construction, by an amount not yet estimated.

The pre-fix H1 values, recorded here for comparison: beta = 1.3449,
Cohen's d = 1.01.

**Remedy:** `last_book_is_premove` and `first_oob_is_premove` are already
written as separate columns of `boundary_pairs.csv` (Script 02, lines
102-111). The filter can therefore be made symmetric -- dropping pairs where
*either* move is a premove -- entirely within Script 04, without re-running
the Script 02 stream over the 89-million-game corpus.

---

## Defect 3 — Stockfish transposition table never cleared (Script 03)

**Severity:** Degrades VOC, the control variable on which the paper's central
methodological claim rests.

`engine.analyse()` is called at lines 246 and 251 without the `game=`
parameter. In python-chess, `ucinewgame` is emitted only when the game
identifier changes:

```python
if self.engine.first_game or self.engine.game != game:
    self.engine.game = game
    self.engine._ucinewgame()
```

With `game` defaulting to `None` on every call, the comparison is
`None != None`, which is false. After the first position, `ucinewgame` is
never sent again, and Stockfish's transposition table is never cleared across
all 114,083 positions.

Line 156 compounds this: positions are processed in `sorted()` FEN order,
which places lexicographically similar positions adjacently. Each depth-1
search therefore runs against a hash freshly populated by the depth-15 search
of a near-identical position -- exactly the condition under which a shallow
search returns a deep-search-quality score.

Measured on 180 sorted opening positions with Stockfish 17, at the configured
depths and evaluation cap, comparing the script as written against
`game=fen` per position:

| quantity                                    | value      |
|---------------------------------------------|------------|
| positions where the depth-1 eval differs    | 93.3%      |
| positions where VOC differs                 | 97.8%      |
| correlation between the two VOC measures    | r = 0.878  |
| RMS difference                              | 33.7 cp    |
| SD of the VOC variable itself               | 69.7 cp    |
| noise-to-signal (RMSE / SD)                 | 0.484      |
| implied reliability (r^2)                   | 0.771      |

The contamination is close to unbiased in the mean (87 positions
under-measured, 89 over-measured, mean shift -3.9 cp). It is therefore
measurement *noise* rather than systematic bias -- but noise in a covariate
attenuates that covariate's coefficient toward zero. Roughly 23% of VOC's
control power is lost. Complexity is under-controlled, and the residual
complexity variance is absorbed by whichever predictor correlates with it,
which here is `is_oob`.

**Direction of effect:** Defects 2 and 3 push the same way. Both inflate the
apparent book-exit cost, which is the paper's headline finding.

**Remedy:** pass `game=board.fen()` to both `analyse()` calls. Requires
re-running Script 03 (3-8 hours).

### Secondary issues in the same file

- Lines 240-242 and 261-263 return `voc: 0.0` for terminal positions and for
  engine exceptions. This recodes analysis failures as "trivially simple
  position", the worst available default for a complexity covariate. Failures
  are logged at `debug` level and are therefore invisible during a normal run;
  their frequency is currently unknown.
- `STOCKFISH_THREADS` must be verified. Any value above 1 makes Stockfish
  non-deterministic, so VOC would not be reproducible from identical input.
- Stockfish at depth 1 still performs quiescence search, so "depth 1 = no
  thinking" is an approximation and should be stated as such in Methods. This
  is a matter for disclosure, not a defect.
- The evaluation perspective handling is correct as written: `.white()` is
  applied to both scores and the difference is taken in absolute value, which
  is invariant to perspective.

---

## Effect on the pre-registered hypotheses

- **H1** — must be re-run after Defects 2 and 3 are corrected. The direction
  of both defects is to inflate the effect. Whether it survives correction is
  an open question and must be treated as such.
- **H2** — the rating-band gradient is estimated from the same contaminated
  sample and inherits the same problem. Note that the book premove rate
  declines monotonically with rating (19.96% -> 10.82%), so the asymmetric
  filter does not act uniformly across bands and may itself generate or
  distort a rating gradient.
- **H3** — cannot be tested as pre-registered. See Defect 1b.

## Order of remediation

1. Fix Script 03 (`game=` parameter); re-run.
2. Fix Script 04 (symmetric premove filter); rebuild `analysis_sample.csv`.
3. Re-run Script 05 for corrected H1 and H2.
4. Report pre-correction and post-correction values side by side.

Retaining the old VOC table permits a direct empirical estimate, on this
dataset rather than on a 180-position sample, of how much the transposition
contamination actually mattered. That comparison belongs in the paper as a
robustness check.
