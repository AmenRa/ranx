# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.19] - 2023-11-28
### Added
- `Run` now has an additional property to store metrics standard deviation.
- `evaluate` now has `return_std` flag to compute metrics standard deviation.

## [0.3.18] - 2023-09-29
### Changed
- `Qrels.from_df` now checks that scores are `numpy.int64` to avoid errors on Windows.
- `Run.from_df` now checks that scores are `numpy.float64` to avoid errors on Windows.

## [0.3.17] - 2023-09-27
### Changed
- All `Run` import methods allow for specifying the `name` of the run.

### Fixed
- Fixed misleading error messages when importing `Qrels` and `Run` from `pandas.DataFrame` with wrong `dtypes`.

## [0.3.16] - 2023-08-03
### Added
- Added support for importing qrels from `parquet` files in `qrels.py`.
- Added support for importing runs from `parquet` files in `run.py`.
- Added support for exporting qrels as `pandas.DataFrame` in `qrels.py`.
- Added support for exporting runs as `pandas.DataFrame` in `run.py`.
- Added support for saving qrels as `parquet` files in `qrels.py`.
- Added support for saving runs as `parquet` files in `run.py`.
  
### Fixed
- Fixed `f1` when there are no relevants.
  
### Changed
- Moved `numba` threading layer settings to `ranx/__init__.py`.

### Removed
- Removed dependency from `pytrec_eval`.

## [0.3.15] - 2023-07-18
### Added
- Added support for gzipped TREC files to `from_file` in `qrels.py`.
- Added support for gzipped TREC files to `from_file` in `run.py`.
- Added `name` parameter to `from_file` in `run.py`.
  
### Fixed
- Fixed `rank_biased_precision` considering relevance as binary instead of graded.
- Fixed high memory consumption for `qrels` and `run`.

## [0.3.14] - 2023-06-24
### Fixed
- Fixed missing metric labels for `dcg` and `dcg_burges` in `report.py`.

## [0.3.13] - 2023-06-16
### Added
- Added `dcg` and `dcg_burges` among the available metrics.

## [0.3.12] - 2023-06-07
### Fixed
- Fixed missing dependency `seaborn`.

## [0.3.10] - 2023-05-26
### Fixed
- Fixed a bug affecting the download of ranxhub runs with special symbols in their ids, such as `+`.

## [0.3.9] - 2023-05-26
### Changed
- Changed `save` in `ranxhub.py` to automatically save average metric scores.

### Fixed
- Fixed a bug affecting `make_comparable` in `run.py`: runs were not sorted after this operation, resulting in wrong metrics computation afterwards.

## [0.3.8] - 2023-05-01
### Added
- It is now possible to plot Interpolated Precision-Recall Curve. Click [here](https://colab.research.google.com/github/AmenRa/ranx/blob/master/notebooks/7_plot.ipynb) for further details.

## [0.3.7] - 2023-04-17
### Added
- Added `make_comparable` to `run.py`. It makes a run comparable to a given qrels whether the run misses results for queries appearing in the qrels or have results for additional queries, which are removed.
- Added  `make_comparable` parameter to `evaluate.py`.
- Added  `make_comparable` parameter to `compare.py`.
  
## [0.3.5] - 2023-02-13
### Fixed
- Fixed a bug affecting `Tukey's HSD Test`: results from the test were not converted to proper dtypes from strings, causing the superscript reporting statistical significance differences in `report.py` to be wrong.

### Changed
- Changed `tukey_hsd_test.py` to use `tukey_hsd` provided by `scipy`.
- `ranx` now requires `python>=3.8`.
- `ranx` now requires `scipy>=1.8`.

### Removed
- Removed dependency from `statsmodels`.
 
## [0.3.4] - 2022-11-22
### Fixed
- Fixed a bug affecting `precision.py`, `recall.py`, and `f1.py`: `numba` does not raise ZeroDivisionError, added a control to make sure zero is returned when no retrieved results are provided for a specific query.
- Fixed a bug in `f1.py`: missing argument in function call.

## [0.3.x] - 2022

Sorry, I have been lazy.

## [0.2.13] - 2022-09-30
### Fixed
- Fixed a bug in `posfuse.py`: `numba` does not raise out of bounds error in some specific cases, added a control to make sure ranking positions with no associated probability get 0 probability.
- Fixed a bug in `baysfuse.py`: as it uses log odds, which can be negative, `comb_sum` cannot be used. Added a `odds_sum` function to combine the log odds.

## [0.2.12] - 2022-09-22
### Fixed
- Fixed a bug in `data_structures/common.py:sort_dict_by_value` that was preventing result list sorting to be consistent for documents with the same score. 
- Fixed a bug causing original runs to be modified by fusion methods.

## [0.2.11] - 2022-09-21
### Fixed
- Fixed a bug in `max_norm.py`, `min_max_norm.py`, and `sum_norm.py`: `min` and `max` functions called on empty lists do not raise error in `Numba` causing downstream miscalculations.
  
## [0.2.10] - 2022-09-12
### Fixed
- Fixed a bug in `bordafuse.py`: `get_candidates` raised error if no run had retrieved docs for a given query.
- Fixed a bug in `borda_norm.py`: `get_candidates` raised error if no run had retrieved docs for a given query.
- Fixed a bug in `condorcet.py`: `get_candidates` raised error if no run had retrieved docs for a given query.

## [0.2.9] - 2022-08-29
### Fixed
- Fixed a bug in `report.py:Report`: some metric labels were missing.
- `SciPy` version explicitly stated in `setup.py` to avoid errors.

### Changed
- `Qrels`'s `save` and `from_file` functions now automatically infer file extension. `kind` parameter can be used to override default behavior.
- `Qrels`'s `save` and `from_file` functions are now much faster with `json` files thanks to [`orjson`](https://github.com/ijl/orjson).
- `Run`'s `save` and `from_file` functions now automatically infer file extension. `kind` parameter can be used to override default behavior.
- `Run`'s `save` and `from_file` functions are now much faster with `json` files thanks to [`orjson`](https://github.com/ijl/orjson).
- `Two-sided Paired Student's t-Test` is now the default statistical test used when calling `compare`. It is much faster than `Fisher's` and usually agrees with it.

## [0.2.x] - 2022

Sorry, I have been lazy.

## [0.1.14] - 2022-04-19
### Fixed
- Fixed a bug in `report.py:Report.to_dict`.
  
## [0.1.13] - 2022-04-18
### Added
- Added `from_ir_datasets` to `qrels.py`. It allows loading qrels from [`ir_metadata`](https://ir-datasets.com).

## [0.1.12] - 2022-04-04
### Added
- Added `paired_student_t_test` to `statistical_testing.py`.
- Added `stat_test` parameter to `compare`. Defaults to `fisher`.
- Added `stat_test` parameter to `report`. Defaults to `fisher`.

### Changed
- `Report`'s `to_latex` function now takes into account the newly introduced `stat_test` parameter to correctly generating LaTeX tables' captions.
- `Report`'s `to_dict` function now takes into account the newly introduced `stat_test` parameter and adds it to the output dictionary.
- `Report`'s `save` function now takes into account the newly introduced `stat_test` parameter and adds it to the output JSON file.

## [0.1.11] - 2022-02-15
### Added
- Added `show_percentages` parameter to `Report`. Defaults to `False`.
- Added `show_percentages` parameter to `compare`. Defaults to `False`.
- Added `rounding_digits` parameter to `compare`. Defaults to `3`.
- Added usage example notebooks for Google Colab.

### Changed
- <span style="color:blue">\[IMPORTANT\]</span> `Qrels` and `Run` now accept a Python Dictionary as initialization parameter and this is the preferred way of creating new instances for those classes. They also accept a `name` parameter. None of those is mandatory, so it should not break code based on previous `ranx` version although this could be changed in the future.
- <span style="color:red">\[BREAKING CHANGE\]</span> `Qrels` and `Run` `save` function `type` parameter renamed to `kind` to prevent it to be interpreted as the `type` Python utility function.
- <span style="color:red">\[BREAKING CHANGE\]</span> `Qrels` and `Run` `save` function now defaults to `json` instead of `trec` for the `kind` parameter (previously called `type`).
- <span style="color:red">\[BREAKING CHANGE\]</span> `Qrels` and `Run` `from_file` function `type` parameter renamed to `kind` to prevent it to be interpreted as the `type` Python utility function.
- <span style="color:red">\[BREAKING CHANGE\]</span> `Qrels` and `Run` `from_file` function now defaults to `json` instead of `trec` for the `kind` parameter (previously called `type`).
- `rounding_digits` parameter of `Report` now defaults to `3`.
- `Report`'s `to_latex` function now produces a simplified LaTeX table.
- Various improvements to `Report` source code.