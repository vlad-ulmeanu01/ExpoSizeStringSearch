import re
import argparse
from random import shuffle
from tqdm import tqdm as tqdm
import json


def retrieve_LLM_samples(label, num_samples):

    with open(f"./LLMSamples/{label}.json", "r") as file:
        content = json.loads(file.read())
    return content["samples"][:num_samples]

def preprocess_string(sample):

    string = " ".join(sample.split())
    return ''.join([char.lower() if char.isalpha() else '' for char in string])

def retrieve_bookcorpus(book_corpus= 1, num_queries= 100000):

    dictionary = []
    with open(f"./bookcorpus/books_large_p{book_corpus}.txt", "r") as file:        
        for line in tqdm(file):
            dictionary.append(preprocess_string(line))
            if len(dictionary) > num_queries:
                break
    return dictionary


def insert_substrings(sample, dictionary, min_step=2, max_step=5, min_size=32, max_size=512):

    for step in range(min_step, max_step, 2):
        for k in range(min_size, max_size):
            for index in range(0, len(sample) - k, step):
                phrase = sample[index: index + k]
                dictionary.append(phrase)

def create_test_instance(targets, dictionary, test_instance):

    with open(test_instance, 'w') as file:

        file.write("{}\n".format(len(dictionary)))
        for idx, query in enumerate(dictionary):
            file.write(query + "\n")

        file.write(f"{len(targets)}\n")
        for target in targets:
            file.write(target + '\n')
        
    print(f"Done creating test instance!")


def prepare_input(label, num_samples, test_instance='input.txt', num_queries=100000, repeat_experiment=100):

    samples = " ".join(retrieve_LLM_samples(label, num_samples))
    samples = preprocess_string(samples)

    dictionary = retrieve_bookcorpus(num_queries=num_queries)
    shuffle(dictionary)
    insert_substrings(samples, dictionary)

    samples = [samples] * repeat_experiment
    create_test_instance(samples, dictionary, test_instance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate toy benchmark test.")
    parser.add_argument("--numSamples", type=int, default=10, help="Number of samples (s)")
    parser.add_argument("--numQueries", type=int, default=100000, help="Number of book phrases (t)")
    parser.add_argument("--repeatExp", type=int, default=100, help="Repeat the experiment with the same data for X times")

    args = parser.parse_args()

    # Available labels: ["hobbit"]
    prepare_input("hobbit", num_samples=args.numSamples, num_queries=args.numQueries, repeat_experiment=args.repeatExp)

