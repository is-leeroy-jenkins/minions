# Development

## Set up the repository

```powershell
git clone https://github.com/is-leeroy-jenkins/minions.git
Set-Location minions
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest pytest-asyncio build
python -m pip install -r requirements-docs.txt
```

## Run verification

```powershell
python -m pytest
python -m build
python -m mkdocs build --strict
```

Tests mock provider network boundaries. They cover provider inheritance, optional tools, concrete
class exports, synchronous and asynchronous execution, streaming, native tools, and local tool
loops.

## Preview documentation

```powershell
python -m mkdocs serve
```

Open `http://127.0.0.1:8000` and verify navigation, code examples, API signatures, and images.

## Documentation conventions

- Lead with the action or contract readers need.
- Use provider-native names and types.
- Keep examples complete enough to adapt directly.
- State whether a tool is provider-hosted or executed locally.
- Keep every page in `mkdocs.yml` navigation so strict builds detect omissions.
- Update the README, user guide, API docstrings, and tests with behavior changes.

## Project structure

```text
minions/
├── gpt.py
├── gemini.py
├── grok.py
├── claude.py
├── mistral.py
├── config.py
├── docs/
├── tests/
├── mkdocs.yml
├── README.md
└── user-guide.md
```
