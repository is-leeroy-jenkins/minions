# Models and Validation API

Model lists support application defaults and validation. `throw_if` is the shared required-value
guard used by provider modules.

::: config

## Required-value guard

Provider modules use `throw_if` from `minions.__init__` to reject `None`, empty strings, and empty
collections before an SDK call. It is an internal consistency helper rather than part of the
workflow-facing API.
