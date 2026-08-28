<!-- badges: start -->
[![Render and Publish](https://github.com/UBC-MDS/introduction-machine-learning/actions/workflows/publish.yaml/badge.svg)](https://github.com/UBC-MDS/introduction-machine-learning/actions/workflows/publish.yaml)
[![Website](https://img.shields.io/badge/website-live-brightgreen.svg)](https://ubc-mds.github.io/introduction-machine-learning/)
[![Built with Quarto](https://img.shields.io/badge/built%20with-Quarto-75AADB.svg)](https://quarto.org)
[![scikit-learn](https://img.shields.io/badge/built%20on-scikit--learn-F7931E.svg)](https://scikit-learn.org)
[![Instructional material: CC BY-NC-SA 4.0](https://img.shields.io/badge/instructional%20material-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE.md)
[![Software: MIT](https://img.shields.io/badge/software-MIT-blue.svg)](LICENSE.md)
<!-- badges: end -->

# Introduction to Machine Learning

This repository holds the [Quarto](https://quarto.org) source for **Introduction to Machine Learning**,
a course in UBC Extended Learning's [Key Capabilities in Data Science
program](https://extendedlearning.ubc.ca/programs-credentials/key-capabilities-data-science-certificate).

The rendered course is published at
**<https://ubc-mds.github.io/introduction-machine-learning/>**.

The course takes a data science perspective on introductory machine learning, with a focus on making
predictions with [scikit-learn](https://scikit-learn.org):

- Supervised learning terminology, baseline models, and the `.fit()` / `.predict()` / `.score()` paradigm.
- Decision trees, k-nearest neighbours, support vector machines with RBF kernels, and linear models.
- Splitting data, cross-validation, the fundamental trade-off, and the golden rule.
- Preprocessing numeric, categorical, and text features with pipelines and column transformers, and
  tuning hyperparameters.
- Evaluating classification and regression models, including on imbalanced data.

The prerequisite is [Programming in Python for Data
Science](https://ubc-mds.github.io/programming-in-python-for-data-science/).

Every coding exercise runs in the browser through [Quarto Live](https://github.com/r-wasm/quarto-live)
and Pyodide, so learners need nothing installed to take the course — the setup below is only for
people editing the material.

## Course Instructors

- [Socorro Dominguez Vidana](https://ht-data.com/about)

## Contributors

This course was built and is maintained by many people from the UBC Master of Data Science program.

- [Hayley Boyce](https://www.hayleyfboyce.com)
- [Mike Gelbart](https://www.mikegelbart.com/)
- [Varada Kolhatkar](https://kvarada.github.io/)
- [Joel Ostblom](https://joelostblom.com/)
- [Tiffany Timbers](https://www.tiffanytimbers.com/)
- [Elijah Willie](https://www.linkedin.com/in/elijah-willie-203845b9/)
- [Mengxin (Betty) Zhao](https://www.linkedin.com/in/mengxinzhao/)
- Sandra Gross
- Ariel Rokem
- [Socorro Dominguez Vidana](https://ht-data.com/about)

### Attribution

- Material adapted from UBC's [DSCI 571: Supervised Learning I](https://github.com/UBC-MDS/DSCI_571_sup-learn-1)
  by [Varada Kolhatkar](https://kvarada.github.io/) and UBC's [CPSC 330](https://github.com/UBC-CS/cpsc330)
  by [Mike Gelbart](https://www.mikegelbart.com/).
- A previous version of the course was built on [Ines Montani](https://ines.io/)'s course framework.
- Datasets and figures credit their sources on the slide where they are used.

## Tips for Contributing

Issues and bug reports are always welcome — please
[open an issue](https://github.com/UBC-MDS/introduction-machine-learning/issues).

For changes, branch off `develop` and open a pull request. Every pull request automatically gets a
preview deployment of the full site, and merges to `develop` publish a staging copy (see
[Deployment](#deployment)), so you can check how a change renders before it reaches `main` and students.
Because the material is course content, prose edits matter as much as code edits:

- Keep the conversational voice of the surrounding text.
- Slide speaker notes (`::: {.notes}` blocks) are the scripts of the recorded videos.
- Prose uses Canadian spelling (*neighbours*, *colour*). Library identifiers keep their own spelling
  (`n_neighbors`), and dataset column names are never "corrected".
- Run the spell check in [Checking spelling](#checking-spelling) before opening a pull request.

## What this repository contains

The course is nine modules plus a closing section:

| Module | Topic |
| --- | --- |
| M0 | Welcome to Introduction to Machine Learning |
| M1 | Machine Learning Terminology |
| M2 | Decision Trees |
| M3 | Splitting, Cross-Validation and the Fundamental Tradeoff |
| M4 | Similarity-Based Approaches to Supervised Learning |
| M5 | Preprocessing Numerical Features, Pipelines and Hyperparameter Optimization |
| M6 | Preprocessing Categorical Variables |
| M7 | Assessment and Measurements |
| M8 | Linear Models |
| — | Module closing remarks (`modules/module9/`) |

Each module directory holds the module's pages, and a `slides/` subdirectory holding its
[reveal.js](https://quarto.org/docs/presentations/revealjs/) decks. In total there are 112 pages and 64
slide decks.

### Repository layout

| Path | What it is |
| --- | --- |
| `_quarto.yml` | Project config: render list, resources, and the hand-maintained sidebar |
| `modules/` | All course content — the only directory that is rendered |
| `data/` | CSV datasets used by slides and exercises |
| `src/utils.py` | Shared Python helpers: display settings, Altair theme, exercise checkers |
| `src/quiz.js` | The `generateQuiz()` multiple-choice widget |
| `src/display_tree.py` | Draws fitted decision trees with Graphviz (Modules 2 and 3 slides) |
| `src/model_plotting_mg.py` | Decision-boundary plots (`plot_classifier`) |
| `static/` | Images, logos, and other media |
| `styles.scss` | Site and slide theming |
| `_extensions/` | Vendored [Quarto Live](https://github.com/r-wasm/quarto-live) extension |
| `doc/` | Syllabus |
| `environment.yaml` | The `kcds-ml` conda environment used locally and in CI |
| `cspell.json` | Spell-check configuration and course vocabulary |

Up to commit `1edf0f3`, the repository also held `chapters/` and `exercises/`: content from the course's
pre-Quarto platform. They were never built by Quarto and have been removed, but that commit remains a
reference for the original exercise numbering.

## How it works

`_quarto.yml` drives everything:

- **`execute-dir: project`** — code cells resolve paths from the repository root, so a lesson reads
  `data/candybars.csv`, never a path relative to its own `.qmd`.
- **`render:`** includes only `modules/**/*.qmd`. Anything outside `modules/` is not built.
- **`resources:`** copies `styles.scss`, the `src/` helpers, and `data/` into the site unexecuted. A
  separate top-level **`pyodide:`** block copies `src/utils.py` and `data/` into the browser's virtual
  filesystem so in-page code can read them.
- **The sidebar `contents:` list is written by hand.** A new page will not appear on the site until it is
  added there.

Pages come in three shapes:

1. **Video and slides** — `format: html`, a tabset holding a YouTube embed and an `<iframe>` pointing at
   the page's rendered reveal.js deck.
2. **Quiz** — `format: html`, loading `src/quiz.js` and calling `generateQuiz()` once per question. The
   correct answer is matched against the option keys by exact string equality, markup included, so an
   edited answer must be changed in both places.
3. **Live coding exercise** — `format: live-html`, using `{pyodide}` cells that execute in the learner's
   browser. A cell tagged `#| exercise: <slug>` holds the fill-in-the-blank starter; a second cell with
   the same slug and `#| check: true` asserts against the learner's `result` and reports back through the
   helpers in `src/utils.py`. Matching `.hint` and `.solution` blocks are keyed by the same slug.

A few pages were renamed to remove typos, colons, and non-ASCII characters from their URLs. Each keeps
its old address working through an `aliases:` entry in its front matter.

## Running the site locally

### System requirements

- [conda](https://docs.conda.io/) or [mamba](https://mamba.readthedocs.io/)
- The packages in `environment.yaml` — Python 3.11 or 3.12, `scikit-learn`, `pandas`, `altair`,
  `matplotlib`, `graphviz`, `xgboost`, `jupyter`, and Quarto itself

### Setup

```bash
git clone git@github.com:UBC-MDS/introduction-machine-learning.git
cd introduction-machine-learning
conda env create -f environment.yaml
conda activate kcds-ml
```

The environment is named `kcds-ml` and installs Quarto, which is what CI uses.

On Apple Silicon Macs the conda-forge Quarto in this environment currently fails to start
(`tools/aarch64/deno: No such file or directory`). Install [Quarto](https://quarto.org/docs/get-started/)
separately and point it at the environment's Python instead:

```bash
QUARTO_PYTHON="$CONDA_PREFIX/bin/python" /Applications/quarto/bin/quarto preview
```

### Build

```bash
quarto preview   # local server with live reload — the usual way to work
quarto render    # full static build into _site/
```

To rebuild a single page instead of the whole site:

```bash
quarto render modules/module2/module2-01-introducing_decision_trees.qmd
```

A full `quarto render` executes all 64 reveal.js decks through Jupyter and takes a while, so prefer
`quarto preview` or a single-file render while editing.

### Troubleshooting

Rendering **any** page starts a local Jupyter kernel — including `live-html` pages, whose `{pyodide}`
cells run in the browser rather than locally. No `.qmd` pins a kernel with a `jupyter:` key, so Quarto
picks the first Python kernelspec it finds. If a stale kernelspec points at an interpreter that no longer
exists, the render fails with a `FileNotFoundError` naming a path unrelated to this repository. Check with:

```bash
jupyter kernelspec list
```

and remove or repair any entry pointing at a deleted environment.

`_site/` and `.quarto/` are build output. Neither belongs in a commit.

### Checking spelling

Prose is checked with [CSpell](https://cspell.org) using Canadian English. Code blocks, inline code,
URLs, and HTML attributes are skipped; course vocabulary lives in the `words` list in `cspell.json`.

```bash
npm install -g cspell @cspell/dict-en-ca
cspell link add @cspell/dict-en-ca
cspell --no-progress "modules/**/*.qmd" _quarto.yml README.md
```

## Deployment

| Workflow | Trigger | Result |
| --- | --- | --- |
| `.github/workflows/publish.yaml` | Push to `main` | Renders the site and deploys `_site/` to the root of the `gh-pages` branch, removing files the build no longer produces |
| `.github/workflows/staging.yaml` | Push to `develop` | Deploys a staging copy to [`/staging/`](https://ubc-mds.github.io/introduction-machine-learning/staging/) |
| `.github/workflows/pr-preview.yaml` | Pull request against `main` or `develop` | Builds and deploys a preview, removed when the PR closes |

All three build the environment from `environment.yaml` with micromamba and run `quarto render`, so a
change that renders cleanly for you locally will render in CI. A new Python dependency must be added to
`environment.yaml` or the build will fail.

## License

This repository carries two licenses, split between the instructional material and the software that
frames it:

- **Instructional material** — the course content — is licensed
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Software** — the framework used to build and host the material — is licensed under the MIT License.

See [`LICENSE.md`](LICENSE.md) for the full text of both.
