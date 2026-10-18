# The Thinking-Time Cost of Leaving Opening Theory in Online Chess

Code, configuration and output tables for a pre-registered archival study of chess thinking times. The registration is at https://osf.io/n6tkx/ (DOI [10.17605/OSF.IO/N6TKX](https://doi.org/10.17605/OSF.IO/N6TKX)).

The study compares each player's last move inside memorized opening theory with that player's first move outside it, using rated Lichess games from December 2024 (96,587,411 games read). Opening theory comes from a separate set of June 2024 Rapid games between players rated 2200 or higher: a move sequence counts as "in book" if it was played at least 10 times there. Position complexity is measured as value of computation (VOC), the change in Stockfish's evaluation from depth 1 to depth 15.

## Requirements

- Python 3.13 and the packages in `requirements.txt` (`pip install -r requirements.txt`)
- Stockfish 18, placed at the path given by `STOCKFISH_PATH` in `config.py` (not included here)
- Disk space for two monthly Lichess files of roughly 30 GB each, plus several GB of processed data

## Data

The games come from the Lichess open database (https://database.lichess.org/), released under CC0. Scripts 01 and 02 download the June 2024 and December 2024 files. Large data files are not stored in this repository (`data/` is ignored by git); the scripts rebuild them.

## Running the pipeline

Run each script from the project root, in this order.

| Step | Script | What it does | Main output |
| --- | --- | --- | --- |
| 1 | `scripts/01_build_reference_corpus.py` | Builds the opening book from June 2024 games of 2200+ players | `data/reference_corpus/book_sequences.csv` |
| 2 | `scripts/02_parse_and_tag_games.py` | Parses the December 2024 Rapid games, finds each player's book exit and computes thinking times (takes 2 to 4 days) | `data/processed/boundary_pairs.csv`, `transition_window.csv` |
| 3 | `scripts/03_stockfish_covariate.py` | Evaluates every boundary position at depths 1 and 15 (VOC) | `data/processed/stockfish_covariate.csv` |
| 4 | `scripts/04_assemble_dataset.py` | Applies the premove filter, joins VOC and draws the analysis sample | `filtered_pairs.csv`, `analysis_sample.csv`, `player_aggregated.csv` |
| 5 | `scripts/10_registered_analyses.py` | Runs the registered analyses: H1, H2, H3-revised and robustness checks 3 to 5 | `outputs/tables/registered_analyses.txt` |
| 6 | `scripts/12_threshold_robustness.py parse`, then `voc`, then `analyze` | Robustness check 1 (book thresholds 5 and 20) | `outputs/tables/threshold_robustness.txt` |
| 7 | `scripts/13_first_move_check.py` | Checks the untimed first moves and reruns the analyses without them | `outputs/tables/first_move_check.txt` |
| 8 | `scripts/08_transition_curve.py`, then `scripts/09_curve_figure.py --journal` | Exploratory transition curve and the two manuscript figures | `outputs/tables/transition_curve.txt`, `outputs/figures/journal/` |

Other scripts:

- `voc_robustness_check.py`: registered robustness check 2 (VOC before and after the Stockfish cache fix)
- `05_statistical_analysis.py`, `05b_mixed_model_refit.py` and `06b_h3_revised.py`: first-stage analyses, superseded for the registered decisions by Script 10
- `06_exgaussian_fitting.py`: the original H3, which was withdrawn; `diagnose_h3.py` shows why
- `07_figures.py`: figures from the first version of the analysis, which are not used in the paper
- `ply_trend_probe.py`, `validate_window.py`, `health_check.py` and `print_versions.py`: diagnostics and checks

All shared settings (data months, time controls, rating bands, book thresholds, premove threshold, Stockfish depths and the random seed of 42) are in `config.py`.

## Untimed first moves

Lichess does not time either player's first move, so the recorded "thinking time" of a first move is just the increment. Script 13 removes pairs whose last in-book move was a player's first move. The paper reports every registered test both with and without those pairs.

## Repository contents

- `outputs/tables/` and `outputs/diagnostics/`: result and diagnostic logs cited in the paper
- `outputs/figures/journal/`: Figures 1 and 2 as submitted, and their captions
- `outputs/figures/fig6` and `fig7`: titled versions of the transition-curve figures
- `outputs/figures/fig1` to `fig5`: figures from the first version of the analysis, whose results were withdrawn
- `docs/prereg/2026-08-04-osf-prereg.md`: the registration text
- `docs/findings/2026-08-03-pipeline-defects.md`: the record of pipeline defects found on August 3, 2026, committed before most of them were fixed

## Citation

Ammasani, S. (2026). *chess-book-exit-research* [Computer software]. GitHub. https://github.com/smayan-ammasani/chess-book-exit-research

## Acknowledgements

An AI assistant (Claude, by Anthropic) was used for code review, debugging, and writing diagnostic and analysis scripts.
