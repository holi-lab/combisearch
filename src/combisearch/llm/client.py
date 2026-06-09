"""LLM client classes: OpenAIChatClient for OpenAI chat models, LlamaClient for
local Llama models (vLLM or Transformers)."""

import logging
import os
from typing import List, TypedDict, Optional, Dict, Any
import torch

from transformers import AutoTokenizer, LlamaForCausalLM

import openai
from openai import BadRequestError
from openai._exceptions import RateLimitError, APIError, APIConnectionError, OpenAIError

from transformers import BitsAndBytesConfig

from combisearch.llm.base import AbstractLMClient
from combisearch.llm.rate_limit import SpeedLimitTimer

TOO_MANY_TOKENS_FOR_ENGINE: str = "This model's maximum context length is"

DEFAULT_STOP_SEQUENCES: List[str] = ['--', '\n', ';', '#']


def check_argument(assertion: Any, message: Optional[str]) -> None:
    if not assertion:
        raise ValueError(message)


def _is_prompt_overlength_error(error: BadRequestError) -> bool:
    message = str(error).lower()
    return (
        TOO_MANY_TOKENS_FOR_ENGINE.lower() in message
        or "maximum context length" in message
        or "too many tokens" in message
    )


class PromptOverlengthError(ValueError):
    """Raised when a prompt exceeds the model's token limit, so callers can retry
    with fewer few-shot examples instead of catching a generic ValueError."""
    pass


class OpenAIAPIConfig(TypedDict):
    """OpenAI API credentials and rate limit; seconds_per_step is the minimum time
    between API calls."""
    api_key: str
    organization: Optional[str]
    seconds_per_step: float


def _load_openai_config_from_env() -> OpenAIAPIConfig:
    """Builds OpenAIAPIConfig from OPENAI_API_KEY (required) and OPENAI_ORGANIZATION
    (optional); raises if no API key is set."""
    api_key: str = os.environ.get("OPENAI_API_KEY")
    organization: str = os.environ.get("OPENAI_ORGANIZATION")

    check_argument(api_key, "must set an API key. Use environment variable OPENAI_API_KEY or otherwise provide "
                            "an OpenAIAPIConfig")

    return {
        "api_key": api_key.strip(),
        "organization": organization,
        "seconds_per_step": 0.2
    }


class OpenAIChatClient(AbstractLMClient):
    """OpenAI chat-completion client with rate limiting and configurable stop
    sequences; greedy, top-p, or beam-search (n) decoding."""

    config: OpenAIAPIConfig
    engine: str
    stop_sequences: List[str]
    timer: SpeedLimitTimer
    beam_search_config: Optional[Dict]

    def __init__(self, config: OpenAIAPIConfig = None, engine: str = "gpt-3.5-turbo-0125",
                 stop_sequences: List[str] = None, beam_search_config=None) -> None:
        super().__init__()

        self.config = config or _load_openai_config_from_env()

        self.engine = engine

        self.stop_sequences = stop_sequences or DEFAULT_STOP_SEQUENCES

        self.timer = SpeedLimitTimer(second_per_step=self.config['seconds_per_step'])

        self.beam_search_config = beam_search_config

    def greedy_lm_completion(self, prompt_text: str) -> Dict[str, float]:
        """Greedy (or beam search if beam_search_config is set) completion for one
        chat prompt. Returns {completion_text: logprob}; raises PromptOverlengthError
        on token-limit overflow."""
        stop_sequences = self.stop_sequences or DEFAULT_STOP_SEQUENCES

        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']

        try:
            args: Dict[str, Any] = {
                "model": self.engine,
                "messages": prompt_text,
                "max_tokens": 120,
                "logprobs": True,
                "temperature": 0.0,
                "stop": stop_sequences,
            }

            if self.beam_search_config:
                args["n"] = self.beam_search_config["beam_size"]

            self.timer.step()

            result = openai.chat.completions.create(**args)

            completions = dict(zip(
                [x.message.content for x in result.choices],
                [sum(token.logprob for token in x.logprobs.content) for x in result.choices]
            ))

            return completions

        except BadRequestError as e:
            if _is_prompt_overlength_error(e):
                raise PromptOverlengthError(e) from e
            raise e

        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def top_p_lm_completion(self, prompt_text: str, top_p: float = 0.9, n: int = 5, best_of: int = 10,
                            max_tokens: int = 120, **kwargs) -> Dict[str, float]:
        """Top-p (nucleus) sampling; returns {completion_text: logprob} for n completions."""
        stop_sequences = self.stop_sequences or DEFAULT_STOP_SEQUENCES

        openai.api_key = self.config['api_key']
        if "organization" in self.config:
            openai.organization = self.config['organization']

        try:
            args = {
                "model": self.engine,
                "messages": prompt_text,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stop": stop_sequences,
                "n": n,
                "logprobs": True,
            }

            self.timer.step()

            result = openai.chat.completions.create(**args)

            completions = dict(zip(
                [x.message.content for x in result.choices],
                [sum(token.logprob for token in x.logprobs.content) for x in result.choices]
            ))
            return completions

        except BadRequestError as e:
            if _is_prompt_overlength_error(e):
                raise PromptOverlengthError(e) from e
            raise e

        except (RateLimitError, APIError, APIConnectionError, OpenAIError) as e:
            logging.warning(e)
            self.timer.sleep(10)
            raise e

    def get_completion_log_probabilities(self, prompt_text: str, completion: str, token_log_probs_telemetry_hook=None):
        raise NotImplementedError("CombiSearch chat completion runs do not use mutual-information rescoring.")


class LlamaClient(AbstractLMClient):
    """Llama client running via vLLM (use_vllm=True) or HF Transformers with 4-bit
    quantization; greedy or beam-search decoding."""

    engine: str
    stop_sequences: List[str]
    timer: SpeedLimitTimer
    model: Any
    tokenizer: Any
    use_vllm: bool
    terminators: List[int]

    def __init__(
        self, 
        config = None, 
        engine: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        stop_sequences: List[str] = None, 
        use_vllm: bool = True,
        quantization: str = None, 
        beam_search_config=None
    ) -> None:
        super().__init__()

        self.config = config
        self.engine = engine
        self.stop_sequences = stop_sequences or DEFAULT_STOP_SEQUENCES
        self.use_vllm = use_vllm
        self.timer = SpeedLimitTimer(second_per_step=0.2)
        self.beam_search_config = beam_search_config

        if use_vllm:
            from vllm import LLM

            self.model = LLM(model=self.engine, quantization=quantization, enforce_eager=False)
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.engine)
            # Use EOS token as pad token (Llama doesn't have a separate pad token)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = LlamaForCausalLM.from_pretrained(
                self.engine,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=False
            )

        self.terminators =  [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def _tokenized_prompts_to_vllm_text_prompts(self, prompt_ids: str) -> List[str]:
        if not isinstance(prompt_ids[0], torch.Tensor):
            prompt_ids = [self.tokenizer.apply_chat_template(
                prompt_ids, add_generation_prompt=True, return_tensors="pt"
            )]
        return [self.tokenizer.batch_decode(prompt, skip_special_tokens=False)[0] for prompt in prompt_ids]

    def _tokenized_prompts_to_vllm_token_prompts(self, prompt_ids: str) -> List[List[int]]:
        if not isinstance(prompt_ids[0], torch.Tensor):
            prompt_ids = [self.tokenizer.apply_chat_template(
                prompt_ids, add_generation_prompt=True, return_tensors="pt"
            )]

        token_prompts = []
        for prompt in prompt_ids:
            if isinstance(prompt, torch.Tensor):
                if prompt.ndim > 1:
                    prompt = prompt[0]
                token_prompts.append(prompt.detach().cpu().tolist())
            else:
                token_prompts.append(list(prompt))
        return token_prompts

    def _trim_vllm_stop_markers(self, completion_text: str) -> str:
        stop_sequences = self.stop_sequences or []
        stop_markers = [stop_sequences] if isinstance(stop_sequences, str) else list(stop_sequences)
        for token_id in self.terminators:
            if token_id is None:
                continue
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            if token_text:
                stop_markers.append(token_text)

        stop_positions = [
            completion_text.find(stop_marker)
            for stop_marker in stop_markers
            if stop_marker and completion_text.find(stop_marker) >= 0
        ]
        if stop_positions:
            completion_text = completion_text[:min(stop_positions)]
        return completion_text

    def _vllm_generate_lm_completion(self, prompt_ids: str) -> Dict[str, float]:
        """vLLM greedy completion for one prompt (chat message list or pre-tokenized);
        returns {completion_text: 1}."""
        stop_sequences = self.stop_sequences or DEFAULT_STOP_SEQUENCES
        self.timer.step()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=1,
            best_of=1,
            max_tokens=120,
            temperature=0,
            stop=stop_sequences,
            stop_token_ids=self.terminators,
        )

        prompts = self._tokenized_prompts_to_vllm_text_prompts(prompt_ids)
        result = self.model.generate(prompts, sampling_params=sampling_params)
        return {result[0].outputs[0].text: 1}

    def _vllm_beam_search_lm_completion(self, prompt_ids: str) -> Dict[str, float]:
        self.timer.step()

        from vllm.sampling_params import BeamSearchParams

        beam_params = BeamSearchParams(
            beam_width=self.beam_search_config["beam_size"],
            max_tokens=120,
            ignore_eos=False,
            temperature=0.0,
            length_penalty=1.0,
        )
        prompt_token_ids = self._tokenized_prompts_to_vllm_token_prompts(prompt_ids)
        prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]
        result = self.model.beam_search(prompts, beam_params)

        completions = []
        for output, input_token_ids in zip(result, prompt_token_ids):
            # vLLM beam_search decodes the full prompt+generation into
            # sequence.text. Slice tokens first to match generate().outputs[0].text.
            sequence = output.sequences[0]
            sequence_token_ids = sequence.tokens
            if sequence_token_ids[:len(input_token_ids)] == input_token_ids:
                completion_token_ids = sequence_token_ids[len(input_token_ids):]
                completion_text = self.tokenizer.decode(completion_token_ids, skip_special_tokens=True)
            else:
                completion_text = sequence.text or ""
            completions.append({self._trim_vllm_stop_markers(completion_text): 1})
        return completions

    def batch_greedy_lm_completion(self, prompt_ids_list: List[Any]) -> List[Dict[str, float]]:
        """Greedy completions for a batch of prompts, each as {completion_text: 1}.
        Separate from greedy_lm_completion, whose single-prompt contract returns one dict."""
        if not self.use_vllm:
            return [self.greedy_lm_completion(prompt_ids) for prompt_ids in prompt_ids_list]

        if self.beam_search_config:
            return self._vllm_beam_search_lm_completion(prompt_ids_list)

        stop_sequences = self.stop_sequences or DEFAULT_STOP_SEQUENCES
        self.timer.step()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=1,
            best_of=1,
            max_tokens=120,
            temperature=0,
            stop=stop_sequences,
            stop_token_ids=self.terminators,
        )

        prompts = self._tokenized_prompts_to_vllm_text_prompts(prompt_ids_list)
        result = self.model.generate(prompts, sampling_params=sampling_params)
        return [{output.outputs[0].text: 1} for output in result]

    def _trim_stop_sequences(self, completion_text: str) -> str:
        for stop_sequence in self.stop_sequences or []:
            if stop_sequence and stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence, 1)[0]
        return completion_text.replace('<|eot_id|>', '').strip()

    def _transformers_greedy_lm_completion(self, prompt_ids: str) -> Dict[str, float]:
        if isinstance(prompt_ids[0], torch.Tensor):
            input_ids = prompt_ids[0].to(self.model.device)
        else:
            input_ids = self.tokenizer.apply_chat_template(
                prompt_ids,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": 120,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": [token_id for token_id in self.terminators if token_id is not None],
        }
        if self.beam_search_config:
            generation_kwargs["num_beams"] = self.beam_search_config["beam_size"]

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generation_kwargs)

        completion_ids = output_ids[0][input_ids.shape[-1]:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return {self._trim_stop_sequences(completion_text): 1}

    def top_p_lm_completion(self, prompt_text: str, top_p: float = 0.9, n: int = 5, best_of: int = 10,
                            max_tokens: int = 120, **kwargs) -> Dict[str, float]:
        raise NotImplementedError("LlamaClient currently supports greedy/beam-search decoding only.")

    def get_completion_log_probabilities(self, prompt_text: str, completion: str, token_log_probs_telemetry_hook=None):
        raise NotImplementedError("LlamaClient currently supports greedy/beam-search decoding only.")
        
    def greedy_lm_completion(self, prompt_ids: str) -> Dict[str, float]:
        """Greedy (or beam search) completion for one prompt via vLLM or Transformers;
        returns {completion_text: 1}."""
        if not self.use_vllm:
            self.timer.step()
            return self._transformers_greedy_lm_completion(prompt_ids)

        if self.beam_search_config:
            return self._vllm_beam_search_lm_completion(prompt_ids)[0]

        return self._vllm_generate_lm_completion(prompt_ids)
