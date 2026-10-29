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

## Deviations from the recorded design

None yet.

## Trade-secret firewall

Public data and published methods only. Nothing from any private trading work.
