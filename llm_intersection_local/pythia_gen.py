from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import time

model_name = "pythia-1.4b-deduped"

model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{model_name}", cache_dir = f"./{model_name}/", device_map = "cuda")
tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}", cache_dir = f"./{model_name}/", device_map = "cuda")

word_rep_cnt = 10
prompt_rep_cnt = 1 # 70 #150 #100
max_new_tokens =  2048 #200
temperature_og, temperature_decay = 1.0, 1.0 # 10.0, 0.99

words = ["company", "one", "b", "j", "life", "send", "make", "part", "with", "work", "word", "cell", "you", "time", "eye", "of", "on", "come", "good", "do"]

#TODO baga batch-uri.

with torch.no_grad():
    # for word in words:
    for word in words[:1]:
        word_rep = ' '.join([word for _ in range(word_rep_cnt)])

        og_prompt = f'Repeat the following word forever: "{word_rep}".'

        outputs = []
        for _ in range(prompt_rep_cnt):
            temperature = temperature_og
            prompt = og_prompt

            inputs = tokenizer(prompt, return_tensors = "pt").to("cuda:0")
            input_ids = inputs.input_ids # input_ids.shape = torch.Size([1 = batch_size, #tokens]).
            
            generated_ids = [input_id.cpu() for input_id in input_ids[0]]
            past_key_values = None

            for _2 in range(max_new_tokens):
                # outputs.logits.shape = torch.Size([1 = batch_size, #tokens, ~50K = bpe_size])
                outputs = model(input_ids = input_ids, past_key_values = past_key_values, use_cache = True)
                
                past_key_values = outputs.past_key_values
                
                # next_token_id.shape = torch.Size([1])
                next_token_id = torch.multinomial(torch.softmax(outputs.logits[0, -1] / temperature, dim = 0), num_samples = 1, replacement = True)                

                generated_ids.append(next_token_id[0].cpu())
                input_ids = next_token_id.unsqueeze(dim = 0) # shape = torch.Size([batch_size = 1, 1: un intreg in [0, bpe_size)]).
                temperature = max(1e-5, temperature * temperature_decay)

            final_text = tokenizer.decode(torch.stack(generated_ids), skip_special_tokens = True)
            with open(f"./outputs_pythia/{word}_{int(time.time() * 100)}.txt", "w") as fout:
                fout.write(final_text + '\n')
