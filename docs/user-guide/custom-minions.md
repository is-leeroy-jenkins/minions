# Custom Minions

Create a domain-specific reusable agent by subclassing the closest workflow role. The provider
implementation, execution methods, and tool behavior are inherited unchanged.

## Create a specialization

```python
from minions.gpt import ComplianceMinion


class BudgetMinion( ComplianceMinion ):
    """OpenAI Minion specialized for federal budget compliance."""

    minion_name: str = 'Budget Minion'
```

Instantiate it like any other Minion:

```python
budget = BudgetMinion(
    model='gpt-5.6-sol',
    instructions=(
        'Evaluate the supplied budget evidence against the stated requirements. '
        'Identify the controlling requirement for every finding.'
    ),
)

result = budget.run( 'Review the proposed obligation.' )
```

## Override one instance name

Use `name` when a reusable subclass is unnecessary:

```python
from minions.gemini import ResearchMinion


minion = ResearchMinion(
    model='gemini-2.5-flash',
    instructions='Research the supplied scientific question.',
    name='Climate Evidence Minion',
)
```

Gemini normalizes the human-readable display name into an ADK-compatible identifier while
retaining the display name.

## Select configured models

`minions.config` publishes provider model lists for discovery and application validation:

```python
from minions.config import OPENAI_MODELS


model = 'gpt-5.6-sol'
if model not in OPENAI_MODELS:
    raise ValueError( f'Unsupported OpenAI model: {model}' )
```

Provider availability can change faster than a package release. Treat these lists as supported
defaults, not as live provider catalogs.

## Keep specializations small

A specialization should normally supply a role name and, when useful, application-level default
configuration. Avoid wrapping return types, translating tools, or duplicating provider execution
logic; those changes weaken the provider-native contract.
