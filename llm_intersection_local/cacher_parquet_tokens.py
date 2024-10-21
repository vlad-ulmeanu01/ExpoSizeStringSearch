from transformers import AutoTokenizer
import pyarrow.parquet as pq
import pyarrow as pa
import itertools
import time

def main():
    model_name = "pythia-1.4b-deduped"
    root_folder = "/export/home/acs/stud/v/vlad_adrian.ulmeanu/E3S_local/llm_copyright"
    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}", cache_dir = f"{root_folder}/{model_name}/", device_map = "cuda")

    cnt_pq_files = 1650
    itv_used_pq_files = (0, 3)
    get_pq_fname = lambda id: f"{root_folder}/the_pile_deduplicated/train-{str(id).zfill(5)}-of-01650.parquet"

    cast_ch = lambda ch: ch if ord(ch) < 256 else chr(255)
    cast_string = lambda s: ''.join(map(cast_ch, [ch for ch in s]))
    cnt_tokens_per_string = 10
    united_lengths = {}

    for table_id in range(itv_used_pq_files[0], itv_used_pq_files[1] + 1):
        table = pq.read_table(get_pq_fname(table_id))

        print(f"{table.shape = }, {table.schema = }")

        t1 = time.time()

        lines = [line.as_py().strip() for line in table["text"]]
        arr_linetokens = [tokenizer.batch_decode(ids, skip_special_tokens = True) for ids in tokenizer.batch_encode_plus(lines)["input_ids"]]

        all_splitstrings = []
        for arr_tokens, cnt_line in zip(arr_linetokens, range(1, table.shape[0] + 1)):
            arr_tokens = list(map(cast_string, arr_tokens))
            united_tokens = [''.join([arr_tokens[j] for j in range(i, min(len(arr_tokens), i + cnt_tokens_per_string))])
                                for i in range(0, len(arr_tokens), cnt_tokens_per_string)]
            
            all_splitstrings.extend(united_tokens)

            for ut in united_tokens:
                if len(ut) not in united_lengths:
                    united_lengths[len(ut)] = 0
                united_lengths[len(ut)] += 1

            if cnt_line % 1000 == 0 or cnt_line == table.shape[0]:
                print(f"{cnt_line = }, {round(time.time() - t1, 3) = }")

        pq.write_table(
            pa.table(data = [pa.array(all_splitstrings)], names = ["text"]),
            f"{root_folder}/the_pile_10token_strings_cached/train-split-{str(table_id).zfill(5)}.parquet"
        )

        print(time.time() - t1)

    print(united_lengths)


if __name__ == "__main__":
    main()
