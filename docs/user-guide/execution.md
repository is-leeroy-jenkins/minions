# Execution and Streaming

Every provider exposes `run`, `run_async`, and `stream`. The methods preserve provider-native
return types and streaming conventions.

## Return types

| Provider | `run` | `run_async` | `stream` |
| --- | --- | --- | --- |
| OpenAI | `RunResult` | `RunResult` | `RunResultStreaming` |
| Gemini | Final `Event` | Final `Event` | `AsyncIterator[Event]` |
| Grok | `Response` | `Response` | `AsyncIterator[tuple[Response, Chunk]]` |
| Claude | `BetaMessage` | `BetaMessage` | `BetaAsyncStreamingToolRunner[object]` |
| Mistral | `ChatCompletionResponse` | `ChatCompletionResponse` | `Iterator[CompletionEvent]` |

## Asynchronous execution

```python
import asyncio

from minions.gpt import ResearchMinion


async def main( ) -> None:
    minion = ResearchMinion(
        model='gpt-5.6-sol',
        instructions='Research the subject and return a concise summary.',
    )
    result = await minion.run_async( 'Research the requested subject.' )
    print( result.final_output )


if __name__ == '__main__':
    asyncio.run( main( ) )
```

Grok and Mistral accept synchronous or asynchronous local functions during async execution.
Synchronous functions run in worker threads.

## OpenAI streaming

```python
stream = minion.stream( 'Explain the provider boundary.' )

async for event in stream.stream_events( ):
    print( event )
```

## Gemini streaming

```python
async for event in minion.stream( 'Research the requested subject.' ):
    print( event )

print( minion.result )
```

## Grok streaming

```python
async for response, chunk in minion.stream( 'Research the requested subject.' ):
    print( chunk )

print( minion.result )
```

## Claude streaming

```python
runner = minion.stream( 'Research the requested subject.' )
```

`runner` is Anthropic's asynchronous streaming tool runner. Consume it through the streaming
interfaces provided by the installed Anthropic SDK; it continues through hosted and local tool
calls automatically.

## Mistral streaming

```python
for event in minion.stream( 'Research the requested subject.' ):
    print( event.data )
```

## Result retention

After execution, `minion.result` refers to the most recent provider-native result or streaming
object. Gemini also retains invocation events in `minion.events`.

## Limits and failures

`max_turns` must be at least `1`. A Grok or Mistral workflow raises `RuntimeError` if local tool
calls exhaust that limit without producing a final answer. Empty required strings, mismatched
local tool mappings, and missing applicable credentials fail before useful provider work begins.
