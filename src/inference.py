import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class VLLMModel():
    def __init__(self, args):
        self.args = args
        num_gpus = torch.cuda.device_count()
        print(f"num_gpus: {num_gpus}")

        self.llm = LLM(model=args.model_name,
                       tensor_parallel_size=num_gpus,
                       trust_remote_code=args.trust_remote_code,
                       # max_num_batched_tokens=args.max_num_batched_tokens,
                       max_model_len=args.max_model_len)
        print(">>>>>> model loaded")

        self.sampling_params = SamplingParams(temperature=args.temperature,
                                              top_p=args.top_p,
                                              max_tokens=args.max_tokens,
                                              stop=args.stop,
                                              seed=args.seed)
        print(self.sampling_params)

    def generate(self, prompts=None, prompt_token_ids=None):
        if self.args.chat:
            outputs = self.llm.chat(prompts, self.sampling_params)
        elif self.args.chat_continuation:
            outputs = self.llm.generate(prompts=prompts, sampling_params=self.sampling_params,
                                        prompt_token_ids=prompt_token_ids)
        else:
            outputs = self.llm.generate(prompts, self.sampling_params)
        # sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
        print(">>>>>> generation done")
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_filename', type=str)
    parser.add_argument('--output_filename', type=str)

    parser.add_argument('--trust_remote_code', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_num_batched_tokens', type=int, default=32768)
    parser.add_argument('--max_model_len', type=int, default=None)

    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--stop', type=str, nargs='+',
                        help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--logprobs', type=int, default=None)
    parser.add_argument('--prompt_logprobs', type=int, default=None)

    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--chat_continuation', action='store_true')
    parser.add_argument('--fixed_token', type=str, default=None)
    parser.add_argument('--remove_system_prompt', action='store_true')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)

    model = VLLMModel(args)
    
    data = [json.loads(line) for line in open(args.input_filename)]

    # Preprocess data: remove system prompt if needed
    if args.remove_system_prompt:
        for instance in data:
            assert instance['messages'][0]['role'] == 'system'
            instance['messages'].pop(0)

    if args.chat_continuation:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        prompt_token_ids = []
        for instance in data:
            if args.fixed_token is not None:
                instance['messages'][-1]['content'] += args.fixed_token

            _prompt_token_ids = tokenizer.apply_chat_template(instance['messages'],
                                                               continue_final_message=True,
                                                               tokenize=True,
                                                               return_dict=True,
                                                               )['input_ids']
            prompt_token_ids.append(_prompt_token_ids)
        prompts = tokenizer.batch_decode(prompt_token_ids)
        assert len(prompts) == len(prompt_token_ids)
    else:
        prompts = [instance['messages'] if args.chat else instance['prompt'] for instance in data]
        prompt_token_ids = None

    if args.debug:
        prompts = prompts[:10]
        prompt_token_ids = prompt_token_ids[:10] if prompt_token_ids is not None else None

    print(prompts[0])
    if prompt_token_ids is not None:
        print(prompt_token_ids[0])

    outputs = model.generate(prompts=prompts, prompt_token_ids=prompt_token_ids)
    print(outputs[0].prompt)
    print(outputs[0].outputs[0].text)

    if not os.path.exists(os.path.dirname(args.output_filename)):
        os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
    with open(args.output_filename, 'w') as f:
        for instance, output in zip(data, outputs):
            if args.chat or args.chat_continuation:
                instance['prompt'] = output.prompt
            else:
                assert instance['prompt'] == output.prompt
            response = output.outputs[0].text
            instance['response'] = response
            if args.fixed_token is not None:
                instance['response'] = args.fixed_token + instance['response']

            f.write(json.dumps(instance) + '\n')

    print("Saved to {}".format(args.output_filename))
