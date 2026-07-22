import json
import os
import re
import time
from urllib.parse import urlparse
from openai import OpenAI
from together import Together
from logbatcher.cluster import Cluster
from logbatcher.postprocess import normalize_template_text, post_process
from logbatcher.matching import extract_variables, prune_from_cluster
from logbatcher.parse_trace import CorrectionSignal, RoutedParseResult
from logbatcher.postprocess import correct_single_template
from logbatcher.util import verify_template, count_message_tokens

class Parser:

    LLM_REQUEST_TIMEOUT_SEC = 300
    LLM_MAX_ATTEMPTS = 3

    def __init__(self, model, theme, config, base_url=None):

        self.model = model
        self.model_lower = self.model.lower()
        self.theme = theme
        self.base_url_override = (
            base_url
            or os.environ.get("LOGBATCHER_BASE_URL")
            or config.get("base_url")
        )
        self.is_local_endpoint = self._is_local_base_url(self.base_url_override)
        self.request_timeout_sec = self._read_float_env(
            "LOGBATCHER_LLM_TIMEOUT_SEC",
            self.LLM_REQUEST_TIMEOUT_SEC,
        )
        self.max_attempts = self._read_int_env(
            "LOGBATCHER_LLM_MAX_ATTEMPTS",
            self.LLM_MAX_ATTEMPTS,
        )
        self.client_max_retries = self._read_int_env(
            "LOGBATCHER_CLIENT_MAX_RETRIES",
            0,
        )
        self.max_output_tokens = self._read_int_env(
            "LOGBATCHER_MAX_TOKENS",
            128,
        )
        self.r2r_max_sample_logs = self._read_int_env(
            "LOGBATCHER_R2R_MAX_SAMPLE_LOGS",
            5,
        )
        self.r2r_fewshot_k = self._read_int_env(
            "LOGBATCHER_R2R_FEWSHOT_K",
            5,
            min_value=0,
        )
        default_fewshot_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "CSLParser",
                "results_sample2k_dataset_1",
                "samples",
            )
        )
        self.fewshot_sample_dir = os.path.abspath(
            os.environ.get("LOGBATCHER_FEWSHOT_SAMPLE_DIR")
            or config.get("fewshot_sample_dir", "")
            or default_fewshot_dir
        )
        self.r2r_template_retry_attempts = self._read_int_env(
            "LOGBATCHER_R2R_TEMPLATE_RETRY_ATTEMPTS",
            2,
        )
        self.r2r_min_template_coverage = self._read_float_env(
            "LOGBATCHER_R2R_MIN_TEMPLATE_COVERAGE",
            0.8,
        )
        self.r2r_validation_log_limit = self._read_int_env(
            "LOGBATCHER_R2R_VALIDATE_LOGS",
            20,
        )
        self.ascii_only_templates = self._read_bool_env(
            "LOGBATCHER_ASCII_ONLY_TEMPLATES",
            False,
        )
        self.r2r_fallback_on_validation_fail = self._read_bool_env(
            "LOGBATCHER_R2R_FALLBACK_ON_VALIDATION_FAIL",
            False,
        )
        self.request_model = (
            os.environ.get("LOGBATCHER_REQUEST_MODEL")
            or ("default" if "r2r" in self.model_lower else self.model)
        )
        self.verbose_llm_io = os.environ.get("LOGBATCHER_VERBOSE_LLM", "0") in (
            "1",
            "true",
            "True",
            "yes",
        )
        self.dataset = 'null'
        self.token_list = [0,0]
        self.time_consumption_llm = 0
        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0
        self.llm_total_tokens = 0
        self.r2r_router_trigger_count = 0
        self.r2r_routed_token_count = 0
        self.r2r_token_trace_count = 0
        self.r2r_endpoint_invocations = 0
        self.r2r_endpoint_latency_sec = 0.0
        self.r2r_reference_invocations = 0
        self.r2r_reference_latency_sec = 0.0
        self.pruning_cache_lookups = 0
        self.pruning_cache_matches = 0
        self.pruning_trusted_cache_hits = 0
        self.pruning_pruned_logs = 0
        self.last_correction_signal = None
        self._fewshot_cache = {}
        if (
            not self.is_local_endpoint
            and config['api_key_from_openai'] == '<OpenAI_API_KEY>'
            and config['api_key_from_together'] == '<Together_API_KEY>'
        ):
            raise ValueError("Please provide your OpenAI API key and Together API key in the config.json file.")
        if self.is_local_endpoint:
            self.api_key = "EMPTY"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url_override,
                timeout=self.request_timeout_sec,
                max_retries=self.client_max_retries,
            )
        elif self.base_url_override:
            if self._is_together_base_url(self.base_url_override):
                self.api_key = config['api_key_from_together']
                self.client = Together(
                    api_key=self.api_key,
                    base_url=self.base_url_override,
                    timeout=self.request_timeout_sec,
                    max_retries=self.client_max_retries,
                )
            else:
                self.api_key = config['api_key_from_openai']
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url_override,
                    timeout=self.request_timeout_sec,
                    max_retries=self.client_max_retries,
                )
        elif 'gpt' in self.model_lower:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=self.request_timeout_sec,
                max_retries=self.client_max_retries,
            )
        elif 'r2r' in self.model_lower:
            self.api_key = "EMPTY"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url_override or "http://localhost:30000/v1",
                timeout=self.request_timeout_sec,
                max_retries=self.client_max_retries,
            )
        elif 'qwen-local' in self.model_lower:
            self.api_key = "EMPTY"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url_override or "http://localhost:30001/v1",
                timeout=self.request_timeout_sec,
                max_retries=self.client_max_retries,
            )
        elif 'qwen' in self.model_lower:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=self.request_timeout_sec,
                max_retries=self.client_max_retries,
            )
        else:
            self.api_key = config['api_key_from_together']
            self.client = Together(
                api_key=self.api_key
            )
        print(
            f"model: {self.model}, base_url: {self.client.base_url}, "
            f"local_endpoint: {self.is_local_endpoint}, "
            f"request_model: {self.request_model}, "
            f"max_tokens: {self.max_output_tokens}, "
            f"r2r_max_sample_logs: {self.r2r_max_sample_logs}, "
            f"r2r_fewshot_k: {self.r2r_fewshot_k}, "
            f"fewshot_sample_dir: {self.fewshot_sample_dir}, "
            f"r2r_template_retry_attempts: {self.r2r_template_retry_attempts}, "
            f"r2r_min_template_coverage: {self.r2r_min_template_coverage}, "
            f"r2r_fallback_on_validation_fail: {self.r2r_fallback_on_validation_fail}, "
            f"ascii_only_templates: {self.ascii_only_templates}, "
            f"timeout: {self.request_timeout_sec}s, "
            f"attempts: {self.max_attempts}, "
            f"client_retries: {self.client_max_retries}"
        )

    @staticmethod
    def _read_int_env(name, default, min_value=1):
        try:
            value = os.environ.get(name)
            if value is None or value.strip() == "":
                return default
            parsed = int(value)
            return parsed if parsed >= min_value else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _read_float_env(name, default):
        try:
            value = os.environ.get(name)
            if value is None or value.strip() == "":
                return default
            parsed = float(value)
            return parsed if parsed > 0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _read_bool_env(name, default):
        value = os.environ.get(name)
        if value is None or value.strip() == "":
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _is_local_base_url(base_url):
        if not base_url:
            return False
        url = base_url if "://" in base_url else f"//{base_url}"
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    @staticmethod
    def _is_together_base_url(base_url):
        if not base_url:
            return False
        url = base_url if "://" in base_url else f"//{base_url}"
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return "together" in hostname

    def reset_metrics(self):
        self.token_list = [0, 0]
        self.time_consumption_llm = 0
        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0
        self.llm_total_tokens = 0
        self.r2r_router_trigger_count = 0
        self.r2r_routed_token_count = 0
        self.r2r_token_trace_count = 0
        self.r2r_endpoint_invocations = 0
        self.r2r_endpoint_latency_sec = 0.0
        self.r2r_reference_invocations = 0
        self.r2r_reference_latency_sec = 0.0
        self.pruning_cache_lookups = 0
        self.pruning_cache_matches = 0
        self.pruning_trusted_cache_hits = 0
        self.pruning_pruned_logs = 0
        self.last_correction_signal = None

    def _chat_once(self, messages):
        response = self.client.chat.completions.create(
            model=self.request_model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_output_tokens,
        )
        return response.choices[0].message.content.strip('\n')

    def _chat_full_response(self, messages):
        kwargs = {}
        if "r2r" in self.model_lower:
            kwargs["extra_body"] = {
                "trace_in_content": True,
                "return_trace": True,
            }
        return self.client.chat.completions.create(
            model=self.request_model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_output_tokens,
            **kwargs,
        )

    @staticmethod
    def _response_to_dict(response_obj):
        if response_obj is None:
            return {}
        if isinstance(response_obj, dict):
            return response_obj
        if hasattr(response_obj, "model_dump"):
            return response_obj.model_dump()
        if hasattr(response_obj, "dict"):
            return response_obj.dict()
        return {}

    @staticmethod
    def _get_response_attr(response_obj, name):
        if response_obj is None:
            return None
        if isinstance(response_obj, dict):
            return response_obj.get(name)
        return getattr(response_obj, name, None)

    @staticmethod
    def _coerce_int(value, default=0):
        try:
            if value is None:
                return default
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                stripped = value.strip().replace(",", "")
                if stripped == "":
                    return default
                return int(float(stripped))
        except (TypeError, ValueError):
            return default
        return default

    def _extract_r2r_reference_usage(self, response_obj):
        response_dict = self._response_to_dict(response_obj)
        candidates = [
            self._get_response_attr(response_obj, "reference_usage"),
            self._get_response_attr(response_obj, "dashscope_usage"),
            response_dict.get("reference_usage"),
            response_dict.get("dashscope_usage"),
        ]

        model_extra = self._get_response_attr(response_obj, "model_extra")
        if isinstance(model_extra, dict):
            candidates.extend([
                model_extra.get("reference_usage"),
                model_extra.get("dashscope_usage"),
            ])

        response_extra = response_dict.get("model_extra")
        if isinstance(response_extra, dict):
            candidates.extend([
                response_extra.get("reference_usage"),
                response_extra.get("dashscope_usage"),
            ])

        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    def _extract_usage(self, response_obj, messages):
        response_dict = self._response_to_dict(response_obj)
        usage = None
        if "r2r" in self.model:
            usage = self._extract_r2r_reference_usage(response_obj)
        if usage is None:
            usage = self._get_response_attr(response_obj, "usage")
            if usage is None:
                usage = response_dict.get("usage")
        prompt_tokens = self._get_response_attr(usage, "prompt_tokens")
        completion_tokens = self._get_response_attr(usage, "completion_tokens")
        total_tokens = self._get_response_attr(usage, "total_tokens")

        if prompt_tokens is None:
            prompt_tokens = count_message_tokens(messages, 'gpt-4o-mini')
        if completion_tokens is None:
            completion_tokens = 0
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _record_llm_usage(self, usage, latency):
        prompt_tokens = self._coerce_int(self._get_response_attr(usage, "prompt_tokens"))
        completion_tokens = self._coerce_int(self._get_response_attr(usage, "completion_tokens"))
        total_tokens = self._coerce_int(self._get_response_attr(usage, "total_tokens"))
        latency = latency or 0.0

        if "r2r" in self.model:
            self.r2r_endpoint_invocations += 1
            self.r2r_endpoint_latency_sec += latency

            if total_tokens <= 0:
                return

            self.r2r_reference_invocations += 1
            self.r2r_reference_latency_sec += latency

        self.token_list[0] += 1
        self.token_list[1] += total_tokens
        self.llm_prompt_tokens += prompt_tokens
        self.llm_completion_tokens += completion_tokens
        self.llm_total_tokens += total_tokens
        self.time_consumption_llm += latency

    def get_r2r_trace_metrics(self):
        return {
            "router_trigger_count": self.r2r_router_trigger_count,
            "routed_token_count": self.r2r_routed_token_count,
            "token_trace_count": self.r2r_token_trace_count,
        }

    def get_pruning_cache_metrics(self):
        return {
            "cache_lookups": self.pruning_cache_lookups,
            "cache_matches": self.pruning_cache_matches,
            "trusted_cache_hits": self.pruning_trusted_cache_hits,
            "pruned_logs": self.pruning_pruned_logs,
        }

    def get_llm_usage_metrics(self):
        return {
            "invocations": self.token_list[0],
            "prompt_tokens": self.llm_prompt_tokens,
            "completion_tokens": self.llm_completion_tokens,
            "total_tokens": self.llm_total_tokens,
            "latency_sec": round(self.time_consumption_llm, 3),
            "avg_latency_sec": (
                round(self.time_consumption_llm / self.token_list[0], 6)
                if self.token_list[0] else 0
            ),
            "r2r_endpoint_invocations": self.r2r_endpoint_invocations,
            "r2r_endpoint_latency_sec": round(self.r2r_endpoint_latency_sec, 3),
            "r2r_endpoint_avg_latency_sec": (
                round(self.r2r_endpoint_latency_sec / self.r2r_endpoint_invocations, 6)
                if self.r2r_endpoint_invocations else 0
            ),
            "r2r_reference_invocations": self.r2r_reference_invocations,
            "r2r_reference_latency_sec": round(self.r2r_reference_latency_sec, 3),
            "r2r_reference_avg_latency_sec": (
                round(self.r2r_reference_latency_sec / self.r2r_reference_invocations, 6)
                if self.r2r_reference_invocations else 0
            ),
        }

    def _parse_r2r_answer(self, answer: str, response_obj=None):
        try:
            payload = json.loads(answer)
            if isinstance(payload, dict) and "final_template" in payload:
                nested_payload = self._parse_nested_r2r_payload(payload)
                if nested_payload is not None:
                    return nested_payload
                return RoutedParseResult.from_payload(payload)
        except (TypeError, json.JSONDecodeError, ValueError):
            pass

        response_dict = self._response_to_dict(response_obj)
        candidates = [
            self._get_response_attr(response_obj, "extra_body"),
            self._get_response_attr(response_obj, "model_extra"),
            response_dict.get("extra_body"),
            response_dict.get("model_extra"),
            response_dict,
        ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            payload = candidate.get("r2r_trace")
            if payload is None and "final_template" in candidate:
                payload = candidate
            if isinstance(payload, dict) and "final_template" in payload:
                try:
                    nested_payload = self._parse_nested_r2r_payload(payload)
                    if nested_payload is not None:
                        return nested_payload
                    return RoutedParseResult.from_payload(payload)
                except ValueError:
                    continue
        return None

    def _parse_nested_r2r_payload(self, payload):
        final_template = payload.get("final_template")
        if not isinstance(final_template, str):
            return None
        try:
            nested = json.loads(final_template)
        except json.JSONDecodeError:
            return None
        if isinstance(nested, dict) and "final_template" in nested:
            merged = dict(nested)
            for field in ("source", "router_trigger_count", "routed_token_count", "token_trace"):
                if field in payload:
                    merged[field] = payload[field]
            return RoutedParseResult.from_payload(merged)
        return None

    @staticmethod
    def _strip_thinking_text(text):
        if not isinstance(text, str):
            return ""
        stripped = text.strip()
        if "<think>" not in stripped:
            if "</think>" in stripped:
                return stripped.split("</think>", 1)[1].strip()
            return stripped
        if "</think>" in stripped:
            return stripped.split("</think>", 1)[1].strip()
        return ""

    @staticmethod
    def _contains_non_ascii(text):
        return any(ord(ch) > 127 for ch in str(text or ""))

    @staticmethod
    def _truncate_at_first_non_ascii(text):
        for idx, char in enumerate(text):
            if ord(char) > 127:
                return text[:idx], True
        return text, False

    @staticmethod
    def _is_low_information_template(template):
        text = str(template or "").replace("<*>", "")
        informative_chars = [ch for ch in text if ch.isalnum()]
        return len(informative_chars) < 3

    @staticmethod
    def _similarity_tokens(text):
        return set(re.findall(r"[A-Za-z_]+|\d+|<\*>", str(text or "").lower()))

    @staticmethod
    def _dedupe_preserve_order(items, limit=None):
        result = []
        seen = set()
        for item in items or []:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
            if limit is not None and len(result) >= limit:
                break
        return result

    def _load_fewshot_samples(self):
        dataset = self.dataset
        if dataset in self._fewshot_cache:
            return self._fewshot_cache[dataset]

        samples = []
        sample_file = os.path.join(
            self.fewshot_sample_dir,
            f"{dataset}_sampled_sim_32.json",
        )
        if not os.path.exists(sample_file):
            self._fewshot_cache[dataset] = samples
            return samples

        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    log = item.get("log")
                    template = item.get("template")
                    if isinstance(log, str) and isinstance(template, str):
                        samples.append({"log": log, "template": template})
        except OSError as e:
            print(f"few-shot sample load skipped for {dataset}: {e}")
            samples = []

        self._fewshot_cache[dataset] = samples
        return samples

    def _select_fewshot_examples(self, request_logs):
        samples = self._load_fewshot_samples()
        if not samples or self.r2r_fewshot_k <= 0:
            return []

        query_logs = self._dedupe_preserve_order(
            request_logs,
            limit=max(1, self.r2r_max_sample_logs),
        )
        query_tokens = self._similarity_tokens(" ".join(query_logs))
        if not query_tokens:
            return samples[:self.r2r_fewshot_k]

        scored = []
        query_text = " ".join(query_logs).lower()
        for idx, sample in enumerate(samples):
            sample_log = sample["log"]
            sample_tokens = self._similarity_tokens(sample_log)
            if sample_tokens:
                overlap = len(query_tokens & sample_tokens)
                union = len(query_tokens | sample_tokens)
                token_score = overlap / union if union else 0.0
            else:
                token_score = 0.0
            substring_bonus = 0.05 if sample_log.lower() in query_text else 0.0
            scored.append((token_score + substring_bonus, -idx, sample))

        scored.sort(reverse=True)
        return [sample for score, _, sample in scored[:self.r2r_fewshot_k] if score > 0]

    def _build_r2r_messages(self, request_logs, variable_prompt, feedback=None):
        examples = self._select_fewshot_examples(request_logs)
        instruction = (
            "/no_think\n"
            "You are a log template extraction engine. Do not reason, do not "
            "explain, and do not output markdown.\n"
            "The input log messages are from the same cluster. Infer one common "
            "template that matches all of them. Replace dynamic values and fields "
            "that vary across the examples with <*>. Keep stable constants, "
            "keywords, punctuation, field names, and units unchanged."
            + variable_prompt
            + "\nRules: preserve units and semantic constants such as MB, GB, KB, "
            "RAM, bytes, decomp:, len:, status:, exception names, and field names. "
            "If a field is present as key= with an empty value, output key=<*>. "
            "For byte-size pairs like '1121 bytes (1.09 KB) sent', output "
            "'<*> bytes (<*>) sent'. Output exactly one final template string. "
            "Do not output JSON, backticks, analysis, <think> blocks, or extra text."
        )

        sections = []
        if examples:
            lines = ["Reference examples:"]
            for idx, sample in enumerate(examples, 1):
                lines.append(f"Example[{idx}] Log: {sample['log']}")
                lines.append(f"Example[{idx}] Template: {sample['template']}")
            sections.append("\n".join(lines))

        log_lines = ["Cluster logs:"]
        for idx, log in enumerate(request_logs, 1):
            log_lines.append(f"Log[{idx}]: {log}")
        sections.append("\n".join(log_lines))

        if feedback:
            sections.append(
                "Previous template failed validation. Fix only the template.\n"
                f"Validation feedback: {feedback}"
            )

        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    def _validation_logs(self, cluster_logs, request_logs):
        logs = list(request_logs or []) + list(cluster_logs or [])
        return self._dedupe_preserve_order(logs, limit=self.r2r_validation_log_limit)

    @staticmethod
    def _missing_required_terms(template, logs):
        template = str(template or "")
        if not logs:
            return []

        missing = []
        terms = [
            "MB",
            "GB",
            "KB",
            "RAM",
            "bytes",
            "decomp:",
            "len:",
            "status:",
            "lifetime",
            "sent",
            "received",
        ]
        threshold = max(1, int(len(logs) * 0.6))
        for term in terms:
            frequency = sum(1 for log in logs if term in log)
            if (
                term in ("KB", "MB", "GB")
                and "bytes (<*>)" in template
                and frequency > 0
            ):
                byte_unit_frequency = 0
                for log in logs:
                    byte_unit_frequency += sum(
                        1
                        for match in re.findall(r"\bbytes \(([^)]*)\)", log)
                        if term in match
                    )
                if byte_unit_frequency >= frequency:
                    continue
            if frequency >= threshold and term not in template:
                missing.append(term)

        has_sent_size = any(re.search(r"\bbytes \([^)]+\) sent\b", log) for log in logs)
        if has_sent_size and "bytes (<*>) sent" not in template:
            missing.append("bytes (<*>) sent")

        has_received_size = any(
            re.search(r"\bbytes \([^)]+\) received\b", log) for log in logs
        )
        if has_received_size and "bytes (<*>) received" not in template:
            missing.append("bytes (<*>) received")

        return missing

    def _validate_r2r_template(self, template, cluster_logs, request_logs):
        validation_logs = self._validation_logs(cluster_logs, request_logs)
        reasons = []

        if not template or not verify_template(template):
            reasons.append("template is empty or has no stable constant text")

        if self._is_low_information_template(template):
            reasons.append("template has too little constant information")

        if validation_logs:
            matched = 0
            for log in validation_logs:
                try:
                    if extract_variables(log, template) is not None:
                        matched += 1
                except Exception:
                    pass
            coverage = matched / len(validation_logs)
            if coverage < self.r2r_min_template_coverage:
                reasons.append(
                    f"template coverage {coverage:.2f} is below "
                    f"{self.r2r_min_template_coverage:.2f}"
                )

            missing_terms = self._missing_required_terms(template, validation_logs)
            if missing_terms:
                reasons.append(
                    "template dropped stable constants or units: "
                    + ", ".join(sorted(set(missing_terms)))
                )

        return {
            "valid": not reasons,
            "feedback": "; ".join(reasons),
        }

    @staticmethod
    def _clean_r2r_template_text(text, ascii_only=False):
        cleaned = Parser._strip_thinking_text(text)
        for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "�"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0]
        truncated_non_ascii = False
        if ascii_only:
            cleaned, truncated_non_ascii = Parser._truncate_at_first_non_ascii(
                cleaned
            )
        cleaned = cleaned.replace("<*)>", "<*>)").replace("<*)", "<*>)")
        cleaned = cleaned.replace("\\n", "\n").strip().strip("`").strip()
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            cleaned = lines[0]
        cleaned = re.sub(
            r"^(?:final_template|template|output template)\s*[:=]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^Log\[\d+\]:\s*", "", cleaned).strip().strip("`").strip()
        return cleaned, truncated_non_ascii

    def _answer_preview(self, answer, template, router_trigger_count, routed_token_count):
        if "r2r" in self.model:
            return (
                "R2R answer preview: "
                f"template={template}, "
                f"router_trigger_count={router_trigger_count}, "
                f"routed_token_count={routed_token_count}"
            )

        answer_preview = (answer or "")[:500].replace("\n", "\\n")
        if answer and len(answer) > 500:
            answer_preview += "...[truncated]"
        return f"LLM answer preview: {answer_preview}"

    def chat(self, messages):
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                answer = self._chat_once(messages)
                latency = time.time() - t0
                if latency > self.request_timeout_sec:
                    print(
                        f"Invalid LLM response: latency {latency:.3f}s exceeds "
                        f"{self.request_timeout_sec}s, retry "
                        f"{attempt}/{self.max_attempts}."
                    )
                    continue
                return answer, latency
            except Exception as e:
                latency = time.time() - t0
                last_error = e
                print(
                    f"LLM request failed: attempt {attempt}/{self.max_attempts}, "
                    f"latency {latency:.3f}s, error: {e}"
                )

        print(
            f"LLM request abandoned after {self.max_attempts} attempts. "
            f"Last error: {last_error}"
        )
        return None, None

    def chat_full_response(self, messages):
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                response = self._chat_full_response(messages)
                latency = time.time() - t0
                if latency > self.request_timeout_sec:
                    print(
                        f"Invalid LLM response: latency {latency:.3f}s exceeds "
                        f"{self.request_timeout_sec}s, retry "
                        f"{attempt}/{self.max_attempts}."
                    )
                    continue
                return response, latency
            except Exception as e:
                latency = time.time() - t0
                last_error = e
                print(
                    f"LLM request failed: attempt {attempt}/{self.max_attempts}, "
                    f"latency {latency:.3f}s, error: {e}"
                )

        print(
            f"LLM request abandoned after {self.max_attempts} attempts. "
            f"Last error: {last_error}"
        )
        return None, None

    @staticmethod
    def _coerce_match_result(match_result):
        if hasattr(match_result, "template"):
            return {
                "template": match_result.template,
                "template_id": getattr(match_result, "template_id", None),
                "relevant_templates": getattr(match_result, "relevant_templates", []),
                "trusted": getattr(match_result, "trusted", True),
                "match_type": getattr(match_result, "match_type", "cache"),
                "best_similarity": getattr(match_result, "best_similarity", 0.0),
                "matched_template": getattr(match_result, "matched_template", match_result.template),
                "legacy": False,
            }

        template, template_id, relevant_templates = match_result
        return {
            "template": template,
            "template_id": template_id if template_id != "NoMatch" else None,
            "relevant_templates": relevant_templates,
            "trusted": True,
            "match_type": "legacy_cache" if template != "NoMatch" else "nomatch",
            "best_similarity": 1.0 if template != "NoMatch" else 0.0,
            "matched_template": template if template != "NoMatch" else None,
            "legacy": True,
        }

    def _emit_correction_signal(
            self,
            cache_base,
            last_match,
            llm_called,
            template,
            router_trigger_count=0,
            routed_token_count=0,
            token_trace=None,
            runtime_metrics=None):
        router_trigger_count = self._coerce_int(router_trigger_count)
        routed_token_count = self._coerce_int(routed_token_count)
        token_trace = token_trace or []
        matched_template = last_match.get("matched_template") if last_match else None
        template_changed = bool(matched_template and matched_template != template)
        signal = CorrectionSignal(
            cache_match_type=last_match.get("match_type", "nomatch") if last_match else "nomatch",
            matched_template_id=last_match.get("template_id") if last_match else None,
            best_similarity=last_match.get("best_similarity", 0.0) if last_match else 0.0,
            llm_used=("r2r" in self.model and router_trigger_count > 0) or (
                "r2r" not in self.model and llm_called
            ),
            final_template=template,
            slm_template=matched_template,
            router_trigger_count=router_trigger_count,
            routed_token_count=routed_token_count,
            token_trace=token_trace,
            template_changed=template_changed,
            conflict=template_changed,
        )
        self.last_correction_signal = signal
        if runtime_metrics is not None:
            runtime_metrics.record_cache_template_check(
                matched_template=matched_template,
                final_template=template,
                checked_by_model=llm_called,
                error=template_changed or signal.conflict,
            )
        if hasattr(cache_base, "update_by_signal"):
            cache_base.update_by_signal(signal)

    def bind_last_signal_to_template(self, cache_base, template_id):
        signal = getattr(self, "last_correction_signal", None)
        if signal is None or signal.matched_template_id is not None:
            return
        if template_id is None or template_id == "NoMatch":
            return
        try:
            if template_id < 0:
                return
        except TypeError:
            return
        signal.matched_template_id = template_id
        if hasattr(cache_base, "update_by_signal"):
            cache_base.update_by_signal(signal)
        self.last_correction_signal = None

    def get_responce(self, cluster, cache_base, runtime_metrics=None):

        # initialize
        logs = cluster.batch_logs
        sample_log = cluster.sample_log
        last_match = None
        llm_called = False
        router_trigger_count = 0
        routed_token_count = 0
        token_trace = []
        self.last_correction_signal = None
        
        # Matching and Pruning
        new_cluster = Cluster()
        for log in cluster.logs:
            self.pruning_cache_lookups += 1
            match = self._coerce_match_result(cache_base.match_event(log))
            template = match["template"]
            if runtime_metrics is not None:
                runtime_metrics.record_cache_lookup(
                    stage="pruning",
                    matched=template != "NoMatch",
                    trusted=match["trusted"],
                )
            if template != "NoMatch":
                self.pruning_cache_matches += 1
                last_match = match
            if template != "NoMatch" and match["trusted"]:
                self.pruning_trusted_cache_hits += 1
                cluster, new_cluster = prune_from_cluster(
                    template, cluster)
                pruned_logs = len(cluster.indexs)
                self.pruning_pruned_logs += pruned_logs
                if runtime_metrics is not None:
                    runtime_metrics.record_cache_hit_logs("pruning", pruned_logs)
                if new_cluster.size >= 0 and new_cluster.size < cluster.size:
                    self._emit_correction_signal(
                        cache_base,
                        last_match,
                        llm_called,
                        template,
                        router_trigger_count,
                        routed_token_count,
                        token_trace,
                        runtime_metrics,
                    )
                    return template, cluster, new_cluster
                elif new_cluster.size == cluster.size:
                    cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
                    new_cluster = Cluster()

        # historical variables
        variable_cluster = Cluster()
        variable_cluster.logs = cache_base.variable_candidates
        if variable_cluster.logs != []:
            try:
                variable_cluster.varaible_sampling(5)
            except ValueError as e:
                print(f"historical variable sampling skipped: {e}")
                variable_cluster.batch_logs = []
        variables = variable_cluster.batch_logs

        variable_prompt = f' Historical variables: {variables}.' if variables != [] else ''
        instruction = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. The variable type in log messages can be any of the following: ['url', 'IPv4_port', 'host_port', 'package_host', 'IPv6', 'Mac_address', 'time', 'path', 'id', 'date', 'duration', 'size', 'numerical', 'weekday_months', 'user_name']." + variable_prompt + " Constant text and strings should not be recognized as variables.\nPrint the input log's template delimited by backticks."

        # invoke LLM
        if "r2r" in self.model:
            request_logs = self._dedupe_preserve_order(
                logs,
                limit=self.r2r_max_sample_logs,
            )
            if not request_logs and sample_log:
                request_logs = [sample_log]
            messages = self._build_r2r_messages(request_logs, variable_prompt)
        else:
            user_content = '\n'.join(f'Log[{i+1}]: `{log}`' for i, log in enumerate(logs))
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_content}
            ]

        answer = None
        template = None
        usage = None
        response_for_metrics = None
        recorded_usage = False
        try:
            if "r2r" in self.model:
                feedback = None
                validation_succeeded = False
                last_validation_feedback = None
                for semantic_attempt in range(1, self.r2r_template_retry_attempts + 1):
                    messages = self._build_r2r_messages(
                        request_logs,
                        variable_prompt,
                        feedback=feedback,
                    )
                    response, latency = self.chat_full_response(messages)
                    response_for_metrics = response
                    if response is None:
                        continue

                    answer = response.choices[0].message.content.strip('\n')
                    usage = self._extract_usage(response, messages)
                    current_router_trigger_count = 0
                    current_routed_token_count = 0
                    current_token_trace = []
                    routed_result = self._parse_r2r_answer(answer, response)
                    if routed_result is not None:
                        ascii_only_for_request = (
                            self.ascii_only_templates
                            and not self._contains_non_ascii(sample_log)
                        )
                        clean_template, truncated_non_ascii = self._clean_r2r_template_text(
                            routed_result.final_template,
                            ascii_only=ascii_only_for_request,
                        )
                        template = normalize_template_text(clean_template)
                        if (
                            not template
                            or (
                                ascii_only_for_request
                                and truncated_non_ascii
                                and self._is_low_information_template(template)
                            )
                        ):
                            template = correct_single_template(sample_log)
                        current_router_trigger_count = routed_result.router_trigger_count
                        current_routed_token_count = routed_result.routed_token_count
                        current_token_trace = routed_result.token_trace
                    else:
                        template = post_process(answer)

                    validation = self._validate_r2r_template(
                        template,
                        cluster.logs,
                        request_logs,
                    )
                    self._record_llm_usage(usage, latency)
                    recorded_usage = True
                    if runtime_metrics is not None:
                        runtime_metrics.record_model_call(
                            response_obj=response_for_metrics,
                            messages=messages,
                            answer=answer,
                            latency_sec=latency,
                            is_r2r=True,
                            router_trigger_count=current_router_trigger_count,
                            routed_token_count=current_routed_token_count,
                            token_trace_count=len(current_token_trace),
                        )

                    router_trigger_count += current_router_trigger_count
                    routed_token_count += current_routed_token_count
                    token_trace.extend(current_token_trace)
                    self.r2r_router_trigger_count += current_router_trigger_count
                    self.r2r_routed_token_count += current_routed_token_count
                    self.r2r_token_trace_count += len(current_token_trace)

                    if validation["valid"]:
                        validation_succeeded = True
                        break

                    feedback = validation["feedback"]
                    last_validation_feedback = feedback
                    if semantic_attempt < self.r2r_template_retry_attempts:
                        print(
                            "R2R template validation failed, retry "
                            f"{semantic_attempt + 1}/{self.r2r_template_retry_attempts}: "
                            f"{feedback}"
                        )
                    else:
                        print(f"R2R template accepted after failed validation: {feedback}")
                if (
                    not validation_succeeded
                    and self.r2r_fallback_on_validation_fail
                    and sample_log
                ):
                    fallback_template = correct_single_template(sample_log)
                    if fallback_template and verify_template(fallback_template):
                        print(
                            "R2R template validation failed after retries; "
                            f"use sample fallback. Feedback: {last_validation_feedback}"
                        )
                        template = fallback_template
                llm_called = response is not None
            else:
                answer, latency = self.chat(messages)
                template = post_process(answer) if answer is not None else None
                llm_called = answer is not None
                if answer is not None:
                    prompt_tokens = count_message_tokens(messages, 'gpt-4o-mini')
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 0,
                        "total_tokens": prompt_tokens,
                    }

            if answer is None:
                print("LLM request abandoned, use sample_log as fallback.")
                answer = sample_log
                template = post_process(answer)
            else:
                if not recorded_usage:
                    self._record_llm_usage(usage, latency)
                if runtime_metrics is not None:
                    if not recorded_usage:
                        runtime_metrics.record_model_call(
                            response_obj=response_for_metrics,
                            messages=messages,
                            answer=answer,
                            latency_sec=latency,
                            is_r2r="r2r" in self.model,
                            router_trigger_count=router_trigger_count,
                            routed_token_count=routed_token_count,
                            token_trace_count=len(token_trace),
                        )
            if self.verbose_llm_io:
                print(messages)
                print(answer)
            else:
                print(
                    self._answer_preview(
                        answer,
                        template,
                        router_trigger_count,
                        routed_token_count,
                    )
                )
        except Exception as e:
            print("invoke LLM error", e)
            answer = sample_log
            template = post_process(answer)

        if (
            self.ascii_only_templates
            and not self._contains_non_ascii(sample_log)
            and self._contains_non_ascii(template)
        ):
            template = correct_single_template(sample_log)

        if not verify_template(template):
            template = correct_single_template(sample_log)
        
        cluster, new_cluster = prune_from_cluster(template, cluster)
        if new_cluster.size == cluster.size:
            cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
            new_cluster = Cluster()
            template = correct_single_template(sample_log)
        self._emit_correction_signal(
            cache_base,
            last_match,
            llm_called,
            template,
            router_trigger_count,
            routed_token_count,
            token_trace,
            runtime_metrics,
        )
        return template, cluster, new_cluster
