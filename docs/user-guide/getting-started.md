# Getting Started

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install git+https://github.com/is-leeroy-jenkins/minions.git
```

Install the optional companion libraries when a workflow uses them:

```powershell
python -m pip install git+https://github.com/is-leeroy-jenkins/guro.git
python -m pip install git+https://github.com/is-leeroy-jenkins/fonky.git
```

## Authenticate

Set only the credential used by the selected provider.

| Provider | Environment variable | Constructor override |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | Native SDK configuration |
| Gemini | `GOOGLE_API_KEY` | Native SDK configuration |
| Grok | `XAI_API_KEY` | `api_key` |
| Claude | `ANTHROPIC_API_KEY` | `api_key` |
| Mistral | `MISTRAL_API_KEY` | `api_key` |

```powershell
$env:OPENAI_API_KEY = '...'
```

Avoid putting credentials directly in committed source files.

## Select a workflow class

| Workflow category | Class |
| --- | --- |
| Research / Academic | `ResearchMinion` |
| Writing / Administrative | `WritingMinion` |
| Compliance / Legal / Budget | `ComplianceMinion` |
| Business / Finance / Marketing | `BusinessMinion` |
| Software Engineering | `CodingMinion` |
| Data Analytics | `DataMinion` |
| Data Governance | `GovernanceMinion` |
| Instruction / Training / Planning | `PlanningMinion` |
| Image Generation | `ImageGenerationMinion` |
| Image Analysis | `ImageAnalysisMinion` |
| Image Editing | `ImageEditingMinion` |
| Translation | `TranslationMinion` |
| Transcription | `TranscriptionMinion` |
| Speech | `SpeechMinion` |

## Construct a Minion

All providers share these arguments:

| Argument | Required | Default | Purpose |
| --- | ---: | ---: | --- |
| `model` | Yes | — | Provider model identifier |
| `instructions` | Yes | — | System instructions for the workflow |
| `tools` | No | `None` | Provider-compatible tools |
| `max_turns` | No | `10` | Maximum model calls or turns |
| `name` | No | Class default | Human-readable display name |

Grok and Mistral additionally accept `functions`. Claude accepts `max_tokens`, defaulting to
`4096`. Grok, Claude, and Mistral accept `api_key` overrides.

## First tool-free workflow

```python
from minions.gpt import WritingMinion


minion = WritingMinion(
    model='gpt-5.6-sol',
    instructions=(
        'Write concise technical documentation. '
        'Use direct language and complete examples.'
    ),
)

result = minion.run(
    'Draft release notes for a library that adds provider-native search tools.'
)
print( result.final_output )
```

The return type is provider-specific. See [Execution and Streaming](execution.md) for the complete
return-type matrix.

## Use Guro and Fonky

```python
from fonky.gpt import tools
from guro import instructions
from minions.gpt import DataMinion


minion = DataMinion(
    model='gpt-5.6-sol',
    instructions=instructions.get( 'DATA_SCIENTIST' ),
    tools=[ tools.fetch_wikipedia, tools.load_csv ],
)

result = minion.run( 'Analyze the supplied data and relevant background evidence.' )
print( result.final_output )
```

Match all three imports to the same provider pathway.
