# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**Aeolus: A Multi-structural Flight Delay Dataset** — an ML pipeline for predicting US domestic flight delays using three representations of the same flights: **tabular**, **chain-based** (temporal, aircraft/crew continuity), and **graph-based** (airport/airspace network).

**Primary task:** Binary classification — predict whether a flight is delayed (`|ARR_DELAY| > 15 min`) using deep tabular baselines (MLP, AutoInt, ResNet, FT-Transformer, SAINT, TabulaRNN, etc. from `mambular`). The repo's novel contribution is a feature-importance method combining **SHAP** with **High-Utility Itemset Mining (HUIM)**.

**Data:** US domestic flights from BTS, available on Kaggle ([mfdd-multi-modal-flight-delay-dataset](https://www.kaggle.com/datasets/flnny123/mfddmulti-modal-flight-delay-dataset)). Both **2020** and **2024** tabular snapshots are used.

> The dataset itself is **not** in git. `Datasets/Aeolus/*`, all `lightning_logs/`, and `model_checkpoints/` are gitignored (see `.gitignore`). Download the data from Kaggle into `Datasets/Aeolus/Flight_Tab/Tab/` before running anything.

## Project rules

Detailed guidance lives in `.claude/rules/` and is imported below:

- @.claude/rules/setup-and-commands.md
- @.claude/rules/architecture.md
- @.claude/rules/shap-huim-method.md
- @.claude/rules/data-and-training-conventions.md
- @.claude/rules/results-interpretation.md

## Git remotes

- **Origin (fork):** https://github.com/brkeudunman/Delay-data.git
- **Upstream (original):** https://github.com/Flnny/Delay-data.git
