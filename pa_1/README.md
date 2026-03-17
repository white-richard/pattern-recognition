# Pattern Recognition Programming Assignment 1

## Prerequisites

- Python 3.14 (possibly 3.10+)
- Some pip package manager. I recommened [`uv`](https://github.com/astral-sh/uv)

If you're on Linux/Mac, you can install `uv` with:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

Create and use a virtual environment, then install dependencies.
Here is an example with `uv`:

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Compile and Run

To run all experiments through the provided script:

```bash
chmod +x run.sh
./run.sh
```

`run.sh` executes the experiment scripts in order:

1. `src/experiment_1.py`
2. `src/experiment_2.py`
3. `src/experiment_3.py`
4. `src/experiment_4.py`

You can also run them individually:

```bash
python src/experiment_1.py
python src/experiment_2.py
python src/experiment_3.py
python src/experiment_4.py
```

## Repo Structure

```bash
.
|-- attachments
|   |-- experiment1_decision.png
|   |-- experiment2_decision.png
|   |-- experiment3_decision.png
|   |-- experiment4_decision.png
|   |-- plotted_dataset_A.png
|   `-- plotted_dataset_B.png
|-- docs
|   `-- results.md
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- run.sh
|-- src
|   |-- bayes.py
|   |-- experiment_1.py
|   |-- experiment_2.py
|   |-- experiment_3.py
|   |-- experiment_4.py
|   |-- generate_normals.py
|   `-- seed.py
```

# Notes

My exact results can be found in `docs/results.md`, though I did heavily seed so it should be highly reproducible.
