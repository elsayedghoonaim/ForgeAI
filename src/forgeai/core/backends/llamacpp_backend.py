"""llama.cpp implementation of the BaseBackend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from rich.console import Console

from forgeai.core.backends.base import BaseBackend, GenerationResult
from forgeai.core.config import BackendType, DevToolSettings

console = Console()


class LlamaCppBackend(BaseBackend):
    """llama.cpp implementation of the inference backend."""

    def __init__(
        self,
        settings: DevToolSettings,
        *,
        streaming: bool = False,
        quiet_startup: bool = False,
    ) -> None:
        super().__init__(settings)
        self._streaming_enabled = streaming
        self._quiet_startup = quiet_startup
        self._engine: Any = None

    @property
    def supports_streaming(self) -> bool:
        return True  # llama-cpp-python always supports streaming

    def initialize(self) -> None:
        """Initialize the llama.cpp engine."""
        if not (self.settings.model_name or self.settings.model_path):
            raise ValueError("A model_name or model_path is required to start the engine.")

        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with: pip install 'forgeai[llamacpp]'"
            ) from err

        kwargs = self.settings.to_llamacpp_kwargs()
        if not self._quiet_startup:
            console.print(f"[dim]Initializing llama.cpp engine with: {kwargs}[/dim]")

        self._engine = Llama(**kwargs)
        self._is_running = True

        if not self._quiet_startup:
            console.print(
                f"[green]OK[/green] Engine initialized: "
                f"[bold]{self.settings.model_name or self.settings.model_path}[/bold] "
                f"(backend={BackendType.LLAMA_CPP.value})"
            )

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages into a model prompt."""
        # 1. Attempt to render using the GGUF model's native Jinja template
        if self._engine is not None and hasattr(self._engine, "metadata"):
            chat_template = self._engine.metadata.get("tokenizer.chat_template")
            if chat_template:
                if isinstance(chat_template, bytes):
                    chat_template = chat_template.decode("utf-8", errors="ignore")

                try:
                    import jinja2

                    # Extract native BOS and EOS tokens from the engine
                    bos_token = "<s>"
                    if hasattr(self._engine, "token_bos"):
                        try:
                            bos_bytes = self._engine.detokenize([self._engine.token_bos()])
                            if bos_bytes:
                                bos_token = bos_bytes.decode("utf-8", errors="ignore")
                        except Exception:
                            pass

                    eos_token = "</s>"
                    if hasattr(self._engine, "token_eos"):
                        try:
                            eos_bytes = self._engine.detokenize([self._engine.token_eos()])
                            if eos_bytes:
                                eos_token = eos_bytes.decode("utf-8", errors="ignore")
                        except Exception:
                            pass

                    # Compile and render using Jinja2
                    env = jinja2.Environment(loader=jinja2.BaseLoader())
                    # HF chat templates sometimes use custom filters or raise_exception function.
                    env.globals["raise_exception"] = lambda msg: msg
                    template = env.from_string(chat_template)

                    rendered = template.render(
                        messages=messages,
                        bos_token=bos_token,
                        eos_token=eos_token,
                        add_generation_prompt=True,
                    )
                    if rendered and isinstance(rendered, str):
                        return rendered
                except Exception:
                    pass

        # 2. Dynamic high-fidelity fallbacks based on model name or model path
        model_identifier = ""
        if self.settings.model_name:
            model_identifier += self.settings.model_name.lower()
        if self.settings.model_path:
            model_identifier += self.settings.model_path.lower()

        # A. Llama-3 Template
        if "llama-3" in model_identifier or "llama3" in model_identifier:
            parts = ["<|begin_of_text|>"]
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        # B. ChatML Template (Qwen, Yi, DeepSeek-Coder, etc.)
        if any(keyword in model_identifier for keyword in ["chatml", "qwen", "yi", "deepseek"]):
            parts = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            return "".join(parts)

        # C. Llama-2 / Mistral / Mixtral Template (INST format)
        if any(keyword in model_identifier for keyword in ["llama-2", "llama2", "mistral", "mixtral"]):
            # Find system message
            system_msg = ""
            for message in messages:
                if message.get("role") == "system":
                    system_msg = message.get("content", "")
                    break

            prompt = ""
            first_user = True
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if role == "system":
                    continue
                elif role == "user":
                    if first_user and system_msg:
                        prompt += f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{content} [/INST]"
                        first_user = False
                    else:
                        prompt += f"<s>[INST] {content} [/INST]"
                elif role == "assistant":
                    prompt += f" {content} </s>"
            return prompt

        # 3. Standard fallback if none of the patterns matched
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        if not self._is_running or not self._engine:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        # If it's a raw string prompt, we use __call__ in a thread pool to avoid blocking the event loop
        output = await asyncio.to_thread(
            self._engine,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        return self._request_output_to_result(output)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream output deltas from the runtime."""
        if not self._is_running or not self._engine:
            raise RuntimeError("Engine is not initialized. Call initialize() first.")

        import queue
        import threading

        q = queue.Queue(maxsize=16)
        stop_event = threading.Event()

        def _producer():
            try:
                generator = self._engine(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    stream=True,
                )
                for chunk in generator:
                    if stop_event.is_set():
                        break
                    while not stop_event.is_set():
                        try:
                            q.put(chunk, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception as e:
                if not stop_event.is_set():
                    q.put(e)
            finally:
                if not stop_event.is_set():
                    q.put(None)

        threading.Thread(target=_producer, daemon=True).start()

        try:
            while True:
                chunk = await asyncio.to_thread(q.get)
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                text = chunk["choices"][0]["text"]
                if text:
                    yield text
        finally:
            stop_event.set()

    @staticmethod
    def _request_output_to_result(output: Any) -> GenerationResult:
        """Convert a llama-cpp-python output dict into GenerationResult."""
        choices = output.get("choices", [])
        if not choices:
            return GenerationResult(text="")

        choice = choices[0]
        text = choice.get("text", "")
        finish_reason = choice.get("finish_reason", "stop")

        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

    def shutdown(self) -> None:
        """Gracefully shut down the engine."""
        if self._engine is not None:
            if hasattr(self._engine, "close"):
                import contextlib
                with contextlib.suppress(Exception):
                    self._engine.close()
            self._engine = None
        self._is_running = False
        console.print("[yellow]llama.cpp Engine shut down.[/yellow]")
