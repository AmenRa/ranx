# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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