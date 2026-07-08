import json
import os
import time
from openai import OpenAI
from together import Together
from logbatcher.cluster import Cluster
from logbatcher.postprocess import normalize_template_text, post_process
from logbatcher.matching import prune_from_cluster
from logbatcher.parse_trace import CorrectionSignal, RoutedParseResult
from logbatcher.postprocess import correct_single_template
from logbatcher.util import verify_template, count_message_tokens

class Parser:

    LLM_REQUEST_TIMEOUT_SEC = 300
    LLM_MAX_ATTEMPTS = 3

    def __init__(self, model, theme, config, base_url=None):

        self.model = model
        self.theme = theme
        self.base_url_override = (
            base_url
            or os.environ.get("LOGBATCHER_BASE_URL")
            or config.get("base_url")
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
        if config['api_key_from_openai'] == '<OpenAI_API_KEY>' and config['api_key_from_together'] == '<Together_API_KEY>':
            raise ValueError("Please provide your OpenAI API key and Together API key in the config.json file.")
        if 'gpt' in self.model:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key
            )
        elif 'r2r' in self.model:
            self.api_key = "EMPTY"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url_override or "http://localhost:30000/v1",
                # timeout=60.0,
                max_retries=3
            )
        elif 'qwen-local' in self.model:
            self.api_key = "EMPTY"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url_override or "http://localhost:30001/v1",
                # timeout=60.0,
                max_retries=3
            )
        elif 'qwen' in self.model:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=60.0,
                max_retries=3
            )
        else:
            self.api_key = config['api_key_from_together']
            self.client = Together(
                api_key=self.api_key
            )
        print(f"model: {self.model}, base_url: {self.client.base_url}")

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
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip('\n')

    def _chat_full_response(self, messages):
        kwargs = {}
        if "r2r" in self.model:
            kwargs["extra_body"] = {
                "trace_in_content": True,
                "return_trace": True,
            }
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
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

    def chat(self, messages):
        last_error = None
        for attempt in range(1, self.LLM_MAX_ATTEMPTS + 1):
            t0 = time.time()
            try:
                answer = self._chat_once(messages)
                latency = time.time() - t0
                if latency > self.LLM_REQUEST_TIMEOUT_SEC:
                    print(
                        f"Invalid LLM response: latency {latency:.3f}s exceeds "
                        f"{self.LLM_REQUEST_TIMEOUT_SEC}s, retry "
                        f"{attempt}/{self.LLM_MAX_ATTEMPTS}."
                    )
                    continue
                return answer, latency
            except Exception as e:
                latency = time.time() - t0
                last_error = e
                print(
                    f"LLM request failed: attempt {attempt}/{self.LLM_MAX_ATTEMPTS}, "
                    f"latency {latency:.3f}s, error: {e}"
                )

        print(
            f"LLM request abandoned after {self.LLM_MAX_ATTEMPTS} attempts. "
            f"Last error: {last_error}"
        )
        return None, None

    def chat_full_response(self, messages):
        last_error = None
        for attempt in range(1, self.LLM_MAX_ATTEMPTS + 1):
            t0 = time.time()
            try:
                response = self._chat_full_response(messages)
                latency = time.time() - t0
                if latency > self.LLM_REQUEST_TIMEOUT_SEC:
                    print(
                        f"Invalid LLM response: latency {latency:.3f}s exceeds "
                        f"{self.LLM_REQUEST_TIMEOUT_SEC}s, retry "
                        f"{attempt}/{self.LLM_MAX_ATTEMPTS}."
                    )
                    continue
                return response, latency
            except Exception as e:
                latency = time.time() - t0
                last_error = e
                print(
                    f"LLM request failed: attempt {attempt}/{self.LLM_MAX_ATTEMPTS}, "
                    f"latency {latency:.3f}s, error: {e}"
                )

        print(
            f"LLM request abandoned after {self.LLM_MAX_ATTEMPTS} attempts. "
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
        if "r2r" in self.model:
            instruction += (
                "\nReturn a JSON object with final_template, source, "
                "router_trigger_count, routed_token_count, and token_trace. "
                "final_template must be a single template string that uses <*> "
                "placeholders directly. Do not wrap final_template in backticks, "
                "do not use {{...}} placeholders, and do not encode another JSON "
                "object inside final_template."
            )

        # invoke LLM
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": '\n'.join(f'Log[{i+1}]: `{log}`' for i, log in enumerate(logs))}
        ]
        try:
            usage = None
            response_for_metrics = None
            if "r2r" in self.model:
                response, latency = self.chat_full_response(messages)
                response_for_metrics = response
                if response is None:
                    answer = None
                else:
                    answer = response.choices[0].message.content.strip('\n')
                    usage = self._extract_usage(response, messages)
                    routed_result = self._parse_r2r_answer(answer, response)
                    if routed_result is not None:
                        template = normalize_template_text(routed_result.final_template)
                        if not template:
                            template = post_process(answer)
                        router_trigger_count = routed_result.router_trigger_count
                        routed_token_count = routed_result.routed_token_count
                        token_trace = routed_result.token_trace
                        self.r2r_router_trigger_count += router_trigger_count
                        self.r2r_routed_token_count += routed_token_count
                        self.r2r_token_trace_count += len(token_trace)
                    else:
                        template = post_process(answer)
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
                self._record_llm_usage(usage, latency)
                if runtime_metrics is not None:
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
            print(messages)
            print(answer)
        except Exception as e:
            print("invoke LLM error", e)
            answer = sample_log
            template = post_process(answer)

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
