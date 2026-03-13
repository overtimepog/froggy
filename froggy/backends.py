"""Backend interface and implementations for model inference."""

import json
import shutil
import subprocess
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from rich.console import Console

from .discovery import ModelInfo

console = Console()


class Backend(ABC):
    @abstractmethod
    def load(self, model_info: ModelInfo, device: str) -> None: ...

    @abstractmethod
    def generate_stream(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class TransformersBackend(Backend):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @property
    def name(self) -> str:
        return "transformers"

    def load(self, model_info: ModelInfo, device: str) -> None:
        import warnings

        import torch
        # Suppress "fast path not available" warning from Qwen3.5 hybrid attention
        # (causal-conv1d / flash-linear-attention don't build on Windows/Python 3.14)
        warnings.filterwarnings("ignore", message=".*fast path is not available.*")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        def _auto_load(path, **kwargs):
            """Load model with the right auto class based on architecture."""
            cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
            archs = getattr(cfg, "architectures", []) or []
            # Multimodal / conditional generation models need a different auto class
            if any("ConditionalGeneration" in a for a in archs):
                from transformers import AutoModelForImageTextToText
                console.print(f"  [dim]Architecture:[/] multimodal ({', '.join(archs)})")
                return AutoModelForImageTextToText.from_pretrained(path, **kwargs)
            console.print(f"  [dim]Architecture:[/] {', '.join(archs) or 'unknown'}")
            return AutoModelForCausalLM.from_pretrained(path, **kwargs)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._dev_label = "CUDA" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"  [dim]Device:[/] {gpu_name} ({vram:.1f} GB VRAM)")
        console.print(f"  [dim]Precision:[/] {dtype}")

        load_kwargs = dict(
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

        if model_info.has_lora:
            self._load_lora(model_info, load_kwargs)
        else:
            console.print(f"  [cyan]\u27f3[/] Loading model: [bold]{model_info.name}[/]")
            t0 = time.time()
            self.model = _auto_load(str(model_info.path), **load_kwargs)
            console.print(f"  [green]\u2714[/] Model loaded [dim]({time.time() - t0:.1f}s)[/]")

        console.print("  [cyan]\u27f3[/] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_info.path), trust_remote_code=True
        )
        console.print("  [green]\u2714[/] Tokenizer loaded")

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
        console.print(f"  [dim]Parameters:[/] {param_count:.1f}B  [dim]Backend:[/] {self._dev_label}")

    def _load_lora(self, model_info: ModelInfo, load_kwargs: dict):
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM

        base_id = model_info.lora_base_model or str(model_info.path)
        base_dir = model_info.path / "base_model"

        # Clean up partial/failed downloads
        if base_dir.exists():
            has_config = (base_dir / "config.json").exists()
            has_weights = (
                any(base_dir.glob("model*.safetensors"))
                or any(base_dir.glob("*.bin"))
            )
            if has_config and has_weights:
                console.print(f"  [green]\u2714[/] Base model found locally: [dim]{base_dir}[/]")
            else:
                console.print("  [yellow]\u26a0[/] Incomplete base model detected, cleaning up...")
                shutil.rmtree(base_dir)
                console.print(f"  [green]\u2714[/] Cleaned [dim]{base_dir}[/]")

        if not base_dir.exists():
            console.print(f"  [yellow]\u2193[/] Downloading base model: [bold]{base_id}[/]")
            console.print(f"    [dim]Saving to: {base_dir}[/]")
            console.print("    [dim]This is a one-time download. Progress bars below:[/]")
            t0 = time.time()
            snapshot_download(
                repo_id=base_id,
                local_dir=str(base_dir),
                ignore_patterns=["*.md", ".gitattributes"],
            )
            console.print(f"  [green]\u2714[/] Base model downloaded [dim]({time.time() - t0:.1f}s)[/]")

        console.print("  [cyan]\u27f3[/] Loading base model into memory...")
        t0 = time.time()

        # Auto-detect architecture (multimodal vs causal)
        cfg = AutoConfig.from_pretrained(str(base_dir), trust_remote_code=True)
        archs = getattr(cfg, "architectures", []) or []
        if any("ConditionalGeneration" in a for a in archs):
            from transformers import AutoModelForImageTextToText
            console.print(f"  [dim]Architecture:[/] multimodal ({', '.join(archs)})")
            base_model = AutoModelForImageTextToText.from_pretrained(str(base_dir), **load_kwargs)
        else:
            console.print(f"  [dim]Architecture:[/] {', '.join(archs) or 'unknown'}")
            base_model = AutoModelForCausalLM.from_pretrained(str(base_dir), **load_kwargs)
        console.print(f"  [green]\u2714[/] Base model loaded [dim]({time.time() - t0:.1f}s)[/]")

        console.print(f"  [cyan]\u27f3[/] Applying LoRA adapter: [dim]{model_info.name}[/]")
        t0 = time.time()
        self.model = PeftModel.from_pretrained(base_model, str(model_info.path))
        console.print(f"  [green]\u2714[/] LoRA adapter applied [dim]({time.time() - t0:.1f}s)[/]")

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(self.model.device)
        else:
            input_ids = inputs["input_ids"].to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Build a comprehensive set of EOS token IDs so the model stops
        # at end-of-turn instead of looping into simulated user/assistant turns
        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        # Add common end-of-turn tokens used by chat models
        for tok_name in ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
                         "<|eot_id|>", "<end_of_turn>", "</s>"]:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok_name)
            # convert_tokens_to_ids returns unk_token_id for unknown tokens
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                eos_ids.add(tok_id)
        # Also check the model's generation_config if available
        if hasattr(self.model, "generation_config"):
            gc = self.model.generation_config
            if gc.eos_token_id is not None:
                if isinstance(gc.eos_token_id, int):
                    eos_ids.add(gc.eos_token_id)
                elif isinstance(gc.eos_token_id, list):
                    eos_ids.update(gc.eos_token_id)

        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            eos_token_id=list(eos_ids) if eos_ids else None,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        thread = threading.Thread(
            target=self.model.generate, kwargs=gen_kwargs, daemon=True
        )
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def _find_llama_cli() -> str | None:
    """Search for a llama.cpp CLI executable on the system PATH."""
    candidates = ["llama-cli", "llama-cli.exe", "main", "main.exe"]
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    # Check common install locations on Windows
    for extra_dir in [
        Path.home() / "llama.cpp" / "build" / "bin" / "Release",
        Path.home() / "llama.cpp" / "build" / "bin",
        Path.home() / "llama.cpp",
    ]:
        for name in candidates:
            exe = extra_dir / name
            if exe.is_file():
                return str(exe)
    return None


def _format_chat_prompt(messages: list[dict]) -> str:
    """Convert chat messages to a ChatML-style prompt string for llama-cli."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class LlamaCppBackend(Backend):
    """Runs GGUF models via llama-cli subprocess."""

    def __init__(self):
        self._exe: str | None = None
        self._gguf_path: Path | None = None
        self._process: subprocess.Popen | None = None
        self._gpu: bool = False

    @property
    def name(self) -> str:
        return "llama.cpp"

    def load(self, model_info: ModelInfo, device: str) -> None:
        exe = _find_llama_cli()
        if exe is None:
            raise FileNotFoundError(
                "Could not find llama-cli on PATH. "
                "Install llama.cpp and ensure llama-cli is on your PATH.\n"
                "  → https://github.com/ggerganov/llama.cpp"
            )
        self._exe = exe
        console.print(f"  [dim]Executable:[/] {exe}")

        # Find the GGUF file in the model directory
        gguf_files = sorted(model_info.path.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf files found in {model_info.path}")
        self._gguf_path = gguf_files[0]
        if len(gguf_files) > 1:
            console.print(f"  [dim]Multiple GGUF files found, using:[/] {self._gguf_path.name}")
        else:
            console.print(f"  [dim]Model file:[/] {self._gguf_path.name}")

        size_gb = self._gguf_path.stat().st_size / 1024**3
        console.print(f"  [dim]Size:[/] {size_gb:.1f} GB")

        self._gpu = device != "cpu"
        if self._gpu:
            console.print("  [dim]GPU offload:[/] enabled (all layers)")

        # Validate the executable works
        try:
            result = subprocess.run(
                [self._exe, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            version_info = (result.stdout or result.stderr).strip().split("\n")[0]
            if version_info:
                console.print(f"  [dim]Version:[/] {version_info}")
        except (subprocess.TimeoutExpired, OSError):
            pass  # --version may not be supported on older builds

        console.print("  [bold green]\u2714 Ready![/]")

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        if self._exe is None or self._gguf_path is None:
            raise RuntimeError("Backend not loaded — call load() first")

        prompt = _format_chat_prompt(messages)

        cmd = [
            self._exe,
            "-m", str(self._gguf_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--no-display-prompt",
            "-s", str(int(time.time())),  # random seed from timestamp
        ]
        if self._gpu:
            cmd.extend(["-ngl", "999"])

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            assert self._process.stdout is not None
            while True:
                char = self._process.stdout.read(1)
                if not char:
                    break
                yield char
        finally:
            self._process.wait()
            self._process = None

    def unload(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        self._exe = None
        self._gguf_path = None


class OllamaBackend(Backend):
    """Communicates with a running Ollama server via its REST API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._model_name: str | None = None

    @property
    def name(self) -> str:
        return "ollama"

    def load(self, model_info: ModelInfo, device: str) -> None:
        console.print(f"  [dim]Server:[/] {self.base_url}")

        # Verify server is reachable and model exists
        with urllib.request.urlopen(f"{self.base_url}/api/tags") as resp:
            data = json.loads(resp.read())

        available = [m["name"] for m in data.get("models", [])]
        if model_info.name not in available:
            raise ValueError(
                f"Model '{model_info.name}' not found on Ollama server. "
                f"Available: {', '.join(available)}"
            )

        self._model_name = model_info.name
        console.print(f"  [dim]Model:[/] {self._model_name}")
        console.print("  [bold green]\u2714 Ready![/]")

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        if self._model_name is None:
            raise RuntimeError("Backend not loaded — call load() first")

        body = json.dumps({
            "model": self._model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("done"):
                    break
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

    def unload(self) -> None:
        self._model_name = None


BACKENDS: dict[str, type[Backend]] = {
    "transformers": TransformersBackend,
    "llama.cpp": LlamaCppBackend,
    "ollama": OllamaBackend,
}


def pick_backend(model_info: ModelInfo) -> Backend:
    """Auto-select the best backend for a given model."""
    if model_info.is_ollama:
        return OllamaBackend()
    if model_info.has_gguf:
        return LlamaCppBackend()
    return TransformersBackend()
