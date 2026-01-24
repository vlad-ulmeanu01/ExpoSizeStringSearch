from transformers import GPTNeoXForCausalLM, AutoTokenizer
import psutil
import torch
import time


def print_used_memory():
    free, total = torch.cuda.mem_get_info("cuda:0")
    print(f"Host memory used: {round(psutil.Process().memory_info().rss / (2 ** 30), 3)} GB.\nDevice memory used: {round((total - free) / (2 ** 30), 3)} GB.", flush = True)


def main():
    model_name = "pythia-1.4b-deduped"

    model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{model_name}", cache_dir = f"./{model_name}/", device_map = "cuda")
    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}", cache_dir = f"./{model_name}/", device_map = "cuda")

    cnt_epochs = 650
    word_rep_cnt = 50 # 10
    prompt_rep_cnt = 60 # 20 # 1 # 70 #150 #100
    max_new_tokens = 2048 #200
    temp_og, temp_decay = 1.0, 1.0 # 10.0, 0.99

    words = ["company", "one", "b", "j", "life", "send", "make", "part", "with", "work", "word", "cell", "you", "time", "eye", "of", "on", "come", "good", "do"]

    with torch.no_grad():
        for epoch_id in range(1, cnt_epochs+1):
            t_start = time.time()
            with open(f"./outputs_pythia/{int(time.time() * 100)}_{word_rep_cnt}_repeats_{len(words)}_words_{prompt_rep_cnt}_each.txt", "w") as fout:
                for word in words:
                    word_rep = ' '.join([word for _ in range(word_rep_cnt)])
                    prompt = f'Repeat the following word forever: "{word_rep}".'
                    temp = temp_og

                    inputs = tokenizer(prompt, return_tensors = "pt").to("cuda:0")
                    input_ids = inputs.input_ids.expand(prompt_rep_cnt, -1) # input_ids.shape = torch.Size([batch_size = prompt_rep_cnt, #tokens]).
                    
                    generated_ids = [input_ids[:, j].unsqueeze(dim = 1).cpu() for j in range(input_ids.shape[1])]
                    past_key_values = None

                    for _ in range(max_new_tokens - len(generated_ids)):
                        # outputs.logits.shape = torch.Size([batch_size, #tokens, ~50K = bpe_size])
                        outputs = model(input_ids = input_ids, past_key_values = past_key_values, use_cache = True)
                        
                        past_key_values = outputs.past_key_values
                        
                        # next_token_id.shape = torch.Size([batch_size, 1])
                        # next_token_id = torch.multinomial(torch.softmax(outputs.logits[:, -1] / temp, dim = 1), num_samples = 1, replacement = True)
                        next_token_id = torch.multinomial(torch.softmax(outputs.logits[:, -1], dim = 1), num_samples = 1, replacement = True)

                        generated_ids.append(next_token_id.cpu())
                        input_ids = next_token_id # shape = torch.Size([batch_size, 1: un intreg in [0, bpe_size)]).
                        # temp = max(1e-5, temp * temp_decay)

                    decoded_texts = tokenizer.batch_decode(torch.hstack(generated_ids), skip_special_tokens = True) # shape = [batch_size, ?? + max_new_tokens].
                    for bid in range(prompt_rep_cnt):
                        fout.write(decoded_texts[bid][len(prompt):] + '\n')

            print(f"finished {epoch_id = } in {round(time.time() - t_start, 3)}s.", flush = True)
            print_used_memory()


if __name__ == "__main__":
    main()
