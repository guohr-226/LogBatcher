import time
from openai import OpenAI
from together import Together
from logbatcher.cluster import Cluster
from logbatcher.postprocess import post_process
from logbatcher.matching import prune_from_cluster
from logbatcher.postprocess import correct_single_template
from logbatcher.util import verify_template, count_message_tokens

class Parser:

    LLM_REQUEST_TIMEOUT_SEC = 300
    LLM_MAX_ATTEMPTS = 3

    def __init__(self, model, theme, config):

        self.model = model
        self.theme = theme
        self.dataset = 'null'
        self.token_list = [0,0]
        self.time_consumption_llm = 0
        if config['api_key_from_openai'] == '<OpenAI_API_KEY>' and config['api_key_from_together'] == '<Together_API_KEY>':
            raise ValueError("Please provide your OpenAI API key and Together API key in the config.json file.")
        if 'gpt' in self.model:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key
            )
        elif 'qwen' in self.model:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=60.0,
                max_retries=3
            )
        elif 'r2r' in self.model:
            self.api_key = "EMPTY"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="http://localhost:30000/v1",
                # timeout=60.0,
                max_retries=3
            )
        elif 'qwen-local' in self.model:
            self.api_key = "EMPTY"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="http://localhost:30001/v1",
                # timeout=60.0,
                max_retries=3
            )
        else:
            self.api_key = config['api_key_from_together']
            self.client = Together(
                api_key=self.api_key
            )
        print(f"model: {self.model}, base_url: {self.client.base_url}")

    def _chat_once(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip('\n')

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

    def get_responce(self, cluster, cache_base):

        # initialize
        logs = cluster.batch_logs
        sample_log = cluster.sample_log
        
        # Matching and Pruning
        new_cluster = Cluster()
        for log in cluster.logs:
            template, _, _ = cache_base.match_event(log)
            if template != "NoMatch":
                cluster, new_cluster = prune_from_cluster(
                    template, cluster)
                if new_cluster.size >= 0 and new_cluster.size < cluster.size:
                    return template, cluster, new_cluster
                elif new_cluster.size == cluster.size:
                    cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
                    new_cluster = Cluster()

        # historical variables
        variable_cluster = Cluster()
        variable_cluster.logs = cache_base.variable_candidates
        if variable_cluster.logs != []:
            variable_cluster.varaible_sampling(5)
        variables = variable_cluster.batch_logs

        variable_prompt = f' Historical variables: {variables}.' if variables != [] else ''
        instruction = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. The variable type in log messages can be any of the following: ['url', 'IPv4_port', 'host_port', 'package_host', 'IPv6', 'Mac_address', 'time', 'path', 'id', 'date', 'duration', 'size', 'numerical', 'weekday_months', 'user_name']." + variable_prompt + " Constant text and strings should not be recognized as variables.\nPrint the input log's template delimited by backticks."

        # invoke LLM
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": '\n'.join(f'Log[{i+1}]: `{log}`' for i, log in enumerate(logs))}
        ]
        try:
            answer, latency = self.chat(messages)
            if answer is None:
                print("LLM request abandoned, use sample_log as fallback.")
                answer = sample_log
            else:
                self.token_list[0] += 1
                self.token_list[1] += count_message_tokens(messages, 'gpt-4o-mini')
                self.time_consumption_llm += latency
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
        return template, cluster, new_cluster
