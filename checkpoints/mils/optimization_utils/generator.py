import numpy as np
from typing import Text, Dict, Tuple, List
from random import sample


def strip_line_counters(text):
    lines = text.split("\n")

    cleaned_lines = []
    for line in lines:
        if ". " in line[:5]:  # A huristic to only count
            period_index = line.find(".")
            cleaned_line = line[period_index + 2 :]
            text = (
                cleaned_line.split("(")[0]
                .strip()
                .replace("<|endoftext|>", "")
                .replace("<pad>", "")
            )
            text = text.replace("!", "")
            if text and text not in cleaned_lines:
                cleaned_lines.append(text)

    return set(cleaned_lines)


class Generator():
    
    def __init__(
        self,
        text_pipeline,
        text_model_name,
        requested_number: int = 100,
        keep_previous: int = 100,
        prompt: Text = "",
        key=lambda x: x[0],
        batch_size: int = 1,
        max_new_tokens: int = 3000,
        device: Text = "cuda:0",
        post_text_function=None,
        verbose: int = 1,
        exploration: float = 0.0,
        ):
        self.key = key
        self.text_model_name = text_model_name
        self.exploration = exploration
        self.batch_size = batch_size
        self.keep_previous = keep_previous
        self.max_new_tokens = max_new_tokens
        self.requested_number = requested_number
        self.prompt = prompt
        self.text_pipeline = text_pipeline
        self.device = device
        self.terminators = [
            self.text_pipeline.tokenizer.eos_token_id,
            self.text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        if "llama" in text_model_name:
            self.text_pipeline.tokenizer.pad_token_id = (
                self.text_pipeline.model.config.eos_token_id[0]
            )
        self.post_text_function = post_text_function
        self.verbose = verbose
        self._post_text_cached = {}
        
        
    def _get_descriptions(self, task, list_of_pairs, **kwargs):
        """
        
        """
        descriptions = ""
        lines = set()
        
        for key, v in list_of_pairs:
            if isinstance(key, float) or isinstance(key, np.float32):
                descriptions += f"{float(key):.3f}"
            else:
                print(key)
                for k in key:
                    descriptions += f"{float(k):.3f}, "
                descriptions = descriptions[:-2]
            if isinstance(v, tuple):
                v = v[0]
            lines.add(v)
            descriptions += f": {v}\n"
        
        new_prompt = self.prompt.replace("{descriptions}", descriptions).replace(
            "{requested_number}", str(self.requested_number)
        )
        
        for k, v in kwargs.items():
            new_prompt = new_prompt.replace("{" + k + "}", v[task])
        if self.verbose > 0:
            print(new_prompt)
            
        return new_prompt, lines
    
    def _get_requests_and_messages(self, task_dict, **kwargs):
        messages = []
        requests = []
        lines_set = []
        
        for task, value in task_dict.items():
            requests.append(task)
            assert len(value)
            assert len(value[0]) == 2, value[0]
            
            list_of_pairs = self.sort_with_exploration(value)
            new_prompt, lines = self._get_descriptions(task, list_of_pairs, **kwargs)
            lines_set.append(lines)
            
            if "llama" in self.text_model_name:
                messages += [
                    self.text_pipeline.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": new_prompt},
                        ],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                ]
            else:
                messages.append(
                    [
                        {"role": "user", "content": new_prompt}
                    ]
                )
        return requests, messages, lines_set
    
    
    def _run_post_text_function(self, descriptions_per_caption):
        data = []
        for j in range(0, len(descriptions_per_caption), self.batch_size):
            # data can be a list of images
            data += self._cached_post_text(
                descriptions_per_caption[j : j + self.batch_size]
            )
        assert len(data) == len(descriptions_per_caption)
        return data
       
        
    def sort_with_exploration(self, values):
        if not self.exploration or self.keep_previous >= len(values):
            return sorted(values, key=self.key)[:self.keep_previous]
        s = sorted(values, key=self.key)
        data = s[: int(self.keep_previous * (1 - self.exploration))]
        return data + sample(
            s[int(self.keep_previous * (1 - self.exploration)) :],
            int(self.keep_previous * self.exploration)
        )
        
    
    def run_on_message_batch(self, current_messages_batch):
        if "llama" in self.text_model_name:
            outputs = self.text_pipeline(
                current_messages_batch,
                batch_size = len(current_messages_batch),
                max_new_tokens = self.max_new_tokens,
                eos_token_id = self.terminators,
                temperature = 0.5,
                top_p = 0.9
            )
            return [
                strip_line_counters(
                    text[0]["generated_text"].split("assistant<|end_header_id|>")[1]
                ) for text in outputs
            ]
        
        outputs = self.text_pipeline(
            current_messages_batch,
            batch_size = len(current_messages_batch),
            max_new_tokens = self.max_new_tokens,
        )
        return [
            strip_line_counters(
                text[0]["generated_text"][-1]["content"]
            ) for text in outputs
        ]
        
    
    def __call__(self, task_dict: Dict[Text, List[Tuple[float, Text]]], **kwargs):
        
        requests, messages, lines_set = self._get_requests_and_messages(
            task_dict=task_dict,
            **kwargs
        )
        assert len(requests) == len(messages)
        
        all_responses = []
        
        for i in range(0, len(messages), self.batch_size):
            current_messages_batch = messages[i : i+ self.batch_size]
            outputs = self.run_on_message_batch(current_messages_batch)
            texts = [list(text - lines_set[i]) for i, text in enumerate(outputs)]
            
            assert len(texts) == len(current_messages_batch), (len(texts), len(current_messages_batch))
            
            if self.post_text_function is not None:
                
                if self.verbose > 0:
                    print(texts)
                    
                for descriptions_per_caption in texts:
                    all_responses.append(
                        list(
                            zip(
                                descriptions_per_caption,
                                self._run_post_text_function(descriptions_per_caption) 
                            )
                        )
                    )
            else:
                
                if self.verbose > 0:
                    print(texts)
                    
                all_responses += texts
                
            if self.verbose > 0:
                print(all_responses)
                
        assert len(all_responses) == len(requests), (len(all_responses), len(requests))
        new_data = {r: m for (r, m) in zip(requests, all_responses)}
        
        for r, val in task_dict.items():
            list_of_pairs = sorted(val, key=self.key)[: self.keep_previous]
            
            for i in list_of_pairs:
                if i[1] not in new_data[r]:
                    new_data[r].append(i[1])
        return new_data