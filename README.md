# How much does model choice move the loan-loss allowance?

Paper 10. Under CECL (ASC 326) a lender reserves today for the credit losses it
expects over a loan's whole life, and it may estimate them with "various
methods" (ASC 326-20-30-3). Many credit models fit the same history about
equally well. This study measures how far the allowance moves across models
that are equally accurate on held-out loans, with the data, the economic
forecast and the reversion policy held fixed, and compares that spread with
the discretion the standard grants explicitly (the reversion method) and with
the losses that actually followed.

**Status:** design recorded on 2026-10-29, before any loan-level file was
downloaded or opened. Nothing below has been changed by looking at data;
later changes go in "Deviations" with the date and the reason.

## Data (public; registration required)

- **Primary: Freddie Mac Single-Family Loan-Level Dataset**, standard
  (fixed-rate, fully amortizing, full documentation) dataset, all origination
  quarters available from 1999 on, monthly performance to the latest release.
  The terms (dated 2025-11-03) allow publishing noncommercial research results
  and derived products that cannot recreate the data. The repository therefore
  releases code and aggregate results only, never loan-level files.
- **Replication: Fannie Mae Single-Family Loan Performance Data**, only if
  Fannie Mae gives written consent to publish results (its licence is for
  internal use). If it does not, no Fannie Mae number appears in the paper.
- **Macro:** FRED, FHFA all-transactions house price index by state
  (`{ST}STHPI`, quarterly) and BLS unemployment rate by state (`{ST}UR`,
  monthly).

Raw files live outside the repository under `$GSE_ROOT` (default `data/gse`).

## Unit of analysis

A **reporting date** t is a calendar year-end, 2005 to the last year-end at
which the portfolio exists (Dec 2005 is the first date with at least six
full years of history). At each t:

- **Portfolio:** every loan active at t (not yet zero-balance, not 180+ days
  delinquent). A fixed random sample of 200,000 portfolio loans is scored and
  the allowance scaled up by the inverse sampling fraction; the sampling error
  this adds is reported.
- **Training data:** loan-years whose outcome year ends on or before t, so a
  model at t uses only what a lender knew at t. Features are measured at the
  start of the loan-year; the outcome is what happened in it.

## Outcome and loss definitions

Per loan-year, three outcomes: **default** (first reaching 180+ days
delinquent, or a credit-event zero-balance code 02, 03, 09 or 15, whichever
comes first), **prepayment** (zero-balance code 01), or **survival**. Loss
severity uses Freddie Mac's documented actual-loss calculation, divided by
the unpaid balance at default.

## Features

At the start of each loan-year: credit score, original LTV and CLTV, DTI,
original rate, rate spread to the 30-year mortgage rate at origination
(Freddie Mac PMMS via FRED, `MORTGAGE30US`), loan purpose, occupancy,
property type, units, first-time buyer, number of borrowers, channel, MI
percentage, original term, loan age, **current LTV** (current balance over
the origination value marked to the state house price index), state
house-price growth over the prior year, and the state unemployment rate and
its change over the prior year. Seller and servicer names, zip3 and MSA are
not used (they identify institutions or are near-identifiers).

## Allowance

For each portfolio loan, two models give annual hazards: a default model,
h_d = P(default | active), and a prepayment model, h_p = P(prepay | active,
no default). Then P(default) = h_d, P(prepay) = (1 − h_d) h_p and P(survive) =
(1 − h_d)(1 − h_p). Any three-outcome distribution factors this way, so
nothing is lost relative to a multinomial model, and the multinomial log loss
is exactly the sum of the two binary log losses. (A single multiclass LightGBM
model was rejected because its monotone constraints apply in the same
direction to every class's score, which would force prepayment to rise with
LTV as default does.) Lifetime default probability is the cumulative default
probability over the remaining contractual term (competing risks), with the
scheduled balance from the contractual rate and term as exposure at default.
Allowance = Σ EAD × marginal default probability × LGD, undiscounted (the
PD/LGD method; ASC 326 does not require discounting outside a DCF method).

- **Macro path (primary):** the reasonable and supportable period is two
  years, during which state HPI growth and unemployment stay at their last
  observed values, followed by straight-line reversion over one year to their
  long-run means computed on data before t. Secondary: the realized path
  (perfect foresight), to separate model error from forecast error.
- **LGD (primary):** one severity model per date, held fixed across the PD
  models: mean severity by current-LTV band and judicial-foreclosure state
  (a published classification, cited in the paper), from defaults resolved
  before t. Secondary: LGD varied with the PD model.

## Models

The prepayment model is fixed at each date (the best of the grid below on
held-out prepayment log loss), so the set varies the default model only; a
secondary analysis varies both. At each date, the default model is one of,
on the same training data:

- 32 LightGBM binary hazard models: 15 or 63 leaves × four row/column
  subsampling settings (80/100% × 80/100%), five seeds for subsampled
  settings and one for the deterministic setting (as in Paper 9);
- the same 32 with monotone constraints (default hazard not falling as
  current LTV or DTI rise, or as credit score falls);
- one logistic regression on splines of the numeric features.

Training rows are capped at 5,000,000 loan-years per date, sampled by loan.

## The set of equally accurate models

Accuracy is the default model's log loss on a held-out 20% of loans (split
by loan, never by row), the proper scoring rule that rewards calibration,
which an allowance needs. Its sampling error is a loan-clustered bootstrap (1,000
replicates). A model is in the set if its held-out log loss is within one
bootstrap standard error of the best. This is a tolerance, not a test of
equality. Sensitivity: 0.5 and 2 standard errors, and a set defined by
default-class AUC.

## Predictions and how each could fail

1. **H1, model choice exceeds training noise.** At a majority of dates, the
   range (max − min) of the allowance across the set exceeds the 95% interval
   width of the best configuration's allowance across 20 refits on loan-level
   bootstrap resamples of its training data. Fails otherwise.
2. **H2, model choice rivals permitted policy discretion.** At a majority of
   dates, the allowance range across the set is at least the range across the
   three reversion policies ASC 326-20-30-9 names (immediate, straight-line
   over one year, straight-line over three years), holding the best model
   fixed. Fails otherwise.
3. **H3, the spread widens into stress.** The allowance range as a share of
   the best model's allowance is larger at year-ends 2006 and 2007 than at
   the median of all other dates. Fails if either date is at or below it.
4. **H4, constraints narrow the spread.** Within each family's own set (same
   tolerance, relative to that family's best), the relative range is smaller
   for monotone-constrained models than for unconstrained ones at a majority
   of dates. Fails otherwise.
5. **H5, the range is not forecast uncertainty.** For dates whose portfolio
   has at least eight years of realized outcomes, realized lifetime credit
   losses lie inside the range of allowances across the set at fewer than
   half of those dates. Fails if the range covers realized losses at half or
   more.
6. **H6, staging flips.** Under the IFRS 9 analogue (lifetime PD at t more
   than twice the lifetime PD at origination as the significant-increase
   test), the share of portfolio loans whose stage differs between at least
   two models in the set exceeds the share whose stage differs across the 20
   bootstrap refits of H1, at a majority of dates. Fails otherwise.

Each prediction is reported whether it holds or fails.

## Data log

- 2026-10-29 10:36 ET: this design committed (07c8ffd), no loan-level file on
  the machine.
- 2026-10-29 13:25 to 14:30 ET: the 28 Standard Dataset year files
  (`historical_data_1999.zip` to `historical_data_2026.zip`, Release 47,
  performance through June 2026) downloaded by the author from Freddie Mac's
  Clarity portal; every archive passed `unzip -t`.
- 2026-10-29 14:33 ET: first loan-level file opened (2026Q2, to read the
  Release 47 layout).

## Deviations from the recorded design

1. **Prepayment grid reduced** (2026-10-29, before any model output was
   seen). The design picks the prepayment model as the best of the full
   65-model grid. A probe fit at December 2008 showed a prepayment LightGBM
   model needs about 2,700 boosting rounds (~100 s), so the full grid would
   cost about two hours per date. The prepayment model is now the best of the
   eight unconstrained LightGBM settings at seed 1 plus the spline logit. The
   monotone constraints are stated for default risk (they would force
   prepayment to fall as credit scores rise), and extra seeds only choose
   the luckiest draw of a model the design holds fixed. The default-model
   grid, the object of the study, is unchanged. The secondary analysis that
   varies the prepayment model uses this nine-model grid.
2. **Secondary "LGD varied with the PD model" not run** (recorded
   2026-10-30). The design listed it as a secondary analysis; it was not
   implemented. The severity table is held fixed across models throughout.
3. **Allowance scaling** (implementation, recorded 2026-10-30). The design
   scales the scored sample's allowance by the inverse sampling fraction;
   the code uses the ratio estimator (expected loss over balance on the
   sample) times the balance of the whole active book. The two agree in
   expectation; the ratio estimator has lower variance.
4. **Long-run means** (implementation, recorded 2026-10-30). The design says
   the reversion means are computed "on data before t"; the code uses every
   year up to and including T, whose December values are observed at T.

## Analyses added after seeing results (labelled as such in the paper)

1. **Paired tolerance** (2026-10-29, after the December 2007 results). The
   recorded set admits models within one bootstrap SE of the best model's own
   held-out log loss. That SE is much larger than the SE of a paired
   difference between two models on the same held-out loans: at December
   2007 the recorded set has 35 models, while only 2 are within one paired SE
   of the best. The paper reports a stricter set alongside the recorded one:
   models whose held-out log loss exceeds the best's by at most one (or two)
   loan-clustered bootstrap SEs of the paired difference. It does not replace
   the recorded set in any prediction.
2. **Like-for-like versions of H1 and H6** (2026-10-30, after all dates). As
   recorded, H1 compares the full range of the ~50-model set with the
   central 95% width of 20 refits, and H6 compares disagreement among ~50
   models with disagreement among 20 refits; both comparisons favour the set
   mechanically. The paper also reports 20 randomly drawn set models against
   the full range and disagreement of the 20 refits.
3. **Lower bound for realized losses** (2026-10-30, after all dates). Unresolved defaults
   (mostly still-active, largely reperforming pandemic-era loans) are valued
   at the severity of resolved defaults in the recorded estimate; the paper
   also reports realized losses with them at zero.
4. **Independent-sample refits** (2026-10-30, after an independent referee
   review). The recorded H1 refits fit on bootstrap resamples, which keep
   about 63% of distinct loans; their held-out log loss is worse than the
   best model's at every date and their mean allowance is biased. The paper
   also reports 20 refits of the best configuration on independent random
   samples of the same size (`code/04b_resample.py`), with their accuracy
   and bias, and repeats the like-for-like comparisons against them.
5. **H5 decomposition** (2026-10-30, after the referee review). For the most
   accurate model, the allowance on the realized macro path and with the
   severity table of all defaults resolved by December 2025, to separate the
   forecast and severity components of the miss.
6. **Delinquency disclosure** (2026-10-30, after the referee review). The
   design has no current-delinquency input; the paper reports the share of
   each portfolio 30-179 days past due and their next-year default rates.

7. **Second review checks** (2026-10-30, after a second independent
   referee review; `code/07b_posthoc.py`). Where the most accurate model sits
   in the set and the rank correlation of held-out log loss with the
   allowance (across the set and across the refits of one configuration); H4
   on random equal-size subsets of the two families' sets; whether the refit
   ranges cover realized losses; the H5 decomposition with the December 2019
   severity table (resolutions before pandemic forbearance) and the model's
   lifetime PD on the realized path against the realized default share.

## Bugs found and fixed

1. 2026-10-29: the realized macro path looked up each state with a tuple
   key and silently fell back to the long-run means; a few dozen loans a year
   with no remaining term were cast to invalid integers. Both fixed in
   `05_allowance.py` (with a regression test) after the first three dates'
   allowances had been computed; the allowance step was rerun for those dates.
2. 2026-10-30: the severity table took the unweighted mean of per-loan
   ratios (loss / balance at default). A handful of loans with a near-zero
   balance at default dominate that mean (one with $0.19 at default and a
   $27,168 loss has a ratio of 143,000), which put the December 2023
   low-LTV, power-of-sale cell at 124%. Severity is now balance-weighted
   (total loss / total balance at default), the rate a dollar allowance
   multiplies by; the same change applies to the imputation of unresolved
   realized losses. Found after all per-date allowances existed, from an
   implausible allowance level; the allowance step was rerun for every date;
   the fitted models are unchanged.
3. 2026-10-30 (found by the referee review): "immediate" reversion set the
   macro value at year-end T+2 to the long-run mean. Because that value is
   the HPI growth over year T+2, this shortened the two-year forecast period
   for house prices. Immediate reversion now starts at T+3, which at annual
   resolution is the same path as one-year straight-line reversion; the
   policy range for H2 is therefore the range between that path and
   three-year straight-line reversion. The allowance step was rerun for every
   date.

## Trade-secret firewall

Public data and published methods only. Nothing from any private trading work.

## Results summary (v0.1; every number is generated by `code/09_paper.py`)

| | Result |
|---|---|
| Set of equally accurate models | median 50 of 65 (27–61); best is unconstrained LightGBM at every date |
| Allowance range across the set | median 55% of the best model's allowance (22%–133%); up to $18.0bn (Dec. 2011) |
| Stricter paired 2-SE set (post hoc) | median 7 models, median relative range 17% |
| H1 model choice > bootstrap refit noise | holds, 13 of 21 dates |
| H2 model choice ≥ reversion-policy range | holds, 21 of 21 dates (median ratio 11.7) |
| H3 spread widens into stress | **fails** (37% and 46% vs 56%) |
| H4 monotone family narrower | holds narrowly, 12 of 21 dates; medians 42% vs 38% |
| H5 realized loss inside the range at < half | holds, 1 of 13 dates |
| H6 more stage flips across set than refits | holds, 17 of 21 dates (27.1% vs 27.5%) |
| Independent-sample refits (post hoc) | span 49% vs 49% for 20 set models (61% vs 32% at the 8 disjoint dates); flip 27.1% vs 18.4% of loans' stage |

## Reproduce

1. Register with Freddie Mac and download the Standard Dataset year archives
   (Release 47) and the FRED series; check them against `data/SHA256SUMS_*`.
   Point `data/gse` at the folder holding them.
2. `python code/01_ingest.py Q` for each quarter, then `code/02_macro.py`,
   `code/03_panel.py`.
3. `code/run_dates.sh` (models and allowances, 21 dates), then
   `code/run_resample.sh` and `code/run_alloc.sh` (independent-sample refits
   and the allowance step with them), `code/06_realized.py`,
   `code/07_analysis.py`, `code/07b_posthoc.py`.
4. `bash paper/build.sh` (figures, numbers, tables, PDF), then
   `python paper/audit.py` (independent recomputation; exits non-zero on any
   failure) and `python -m pytest tests`.

Python 3.12 or later; see `pyproject.toml`.
