# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Experimental `Fusion` functionalities (_undocumented_).
- Experimental `Normalization` functionalities (_undocumented_).

## [0.1.12] - 2022-04-04
### Added
- Added `paired_student_t_test` to `statistical_testing.py`.
- Added `stat_test` parameter to `compare`. Defaults to `fisher`.
- Added `stat_test` parameter to `report`. Defaults to `fisher`.

### Changed
- `Report`'s `to_latex` function now takes into account the newly introduce `stat_test` parameter to correctly generating LaTeX tables' captions.
- `Report`'s `to_dict` function now takes into account the newly introduce `stat_test` parameter and adds it to the output dictionary.
- `Report`'s `save` function now takes into account the newly introduce `stat_test` parameter and adds it to the output JSON file.

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