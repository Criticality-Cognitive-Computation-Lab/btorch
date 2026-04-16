# Changelog

All notable changes to btorch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Expanded API reference coverage: visualisation, io, utils, models (functional, history, environ, init, scale, regularizer, ode, connection_conversion), analysis (dynamic_tools), and connectome (connection, augment).
- New Core Concepts guides: "Stateful Modules", "The `dt` Environment", and "Surrogate Gradients".
- New Tutorials: "Tutorial 1: Building an RSNN" and "Tutorial 2: Training an SNN".
- New Guides: "OmegaConf Configuration Guide".
- New pages: FAQ, Skills Reference, Examples Gallery, Contributing.
- MkDocs Material UX upgrades: edit/view links, breadcrumb navigation, back-to-top button, shareable search URLs, and version badge.

### Changed
- Updated `api/neurons.md` and `api/analysis.md` to expose full public APIs.
- Improved Google-style docstrings across `btorch.models.functional`, `environ`, `init`, `scale`, `regularizer`, and `btorch.utils.conf`.

### Removed
- Redundant `zh/docs/api/` English mkdocstrings blocks (cleanup pending full translation workflow update).
