"""Backend interface and implementations for model inference."""

import shutil
import threading
import time
from abc import ABC, abstractmethod
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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

        console.print(f"  [cyan]\u27f3[/] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_info.path), trust_remote_code=True
        )
        console.print(f"  [green]\u2714[/] Tokenizer loaded")

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
        console.print(f"  [dim]Parameters:[/] {param_count:.1f}B  [dim]Backend:[/] {self._dev_label}")

    def _load_lora(self, model_info: ModelInfo, load_kwargs: dict):
        from peft import PeftModel
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoConfig

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
                console.print(f"  [yellow]\u26a0[/] Incomplete base model detected, cleaning up...")
                shutil.rmtree(base_dir)
                console.print(f"  [green]\u2714[/] Cleaned [dim]{base_dir}[/]")

        if not base_dir.exists():
            console.print(f"  [yellow]\u2193[/] Downloading base model: [bold]{base_id}[/]")
            console.print(f"    [dim]Saving to: {base_dir}[/]")
            console.print(f"    [dim]This is a one-time download. Progress bars below:[/]")
            t0 = time.time()
            snapshot_download(
                repo_id=base_id,
                local_dir=str(base_dir),
                ignore_patterns=["*.md", ".gitattributes"],
            )
            console.print(f"  [green]\u2714[/] Base model downloaded [dim]({time.time() - t0:.1f}s)[/]")

        console.print(f"  [cyan]\u27f3[/] Loading base model into memory...")
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

        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
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


class LlamaCppBackend(Backend):
    """Future: wraps llama-cli.exe via subprocess."""

    @property
    def name(self) -> str:
        return "llama.cpp"

    def load(self, model_info: ModelInfo, device: str) -> None:
        raise NotImplementedError("llama.cpp backend not yet implemented — needs GGUF files")

    def generate_stream(self, messages, temperature, max_tokens) -> Iterator[str]:
        raise NotImplementedError

    def unload(self) -> None:
        pass


BACKENDS: dict[str, type[Backend]] = {
    "transformers": TransformersBackend,
    "llama.cpp": LlamaCppBackend,
}


def pick_backend(model_info: ModelInfo) -> Backend:
    """Auto-select the best backend for a given model."""
    if model_info.has_gguf:
        return LlamaCppBackend()
    return TransformersBackend()
