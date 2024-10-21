#include <iostream>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdint>

#include "E3S.h"

#define TIMER(x) stop = std::chrono::steady_clock::now(); std::cout << "(TIMER) " << x << ' ' << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000000.0 << '\n' << std::flush; start = std::chrono::steady_clock::now();

std::vector<std::vector<uint8_t>> read_pile_10token_strings(std::string fname) {
    constexpr int buffsize = 65536;

    int pipe_subproc_read[2], pipe_subproc_write[2]; ///read from [0], write to [1].
    pipe(pipe_subproc_read); ///the subprocess should read through this pipe. we write to this.
    pipe(pipe_subproc_write); ///subprocess writes. we read.

    pid_t pid = fork();

    if (pid == 0) { ///Child process:
        dup2(pipe_subproc_read[0], STDIN_FILENO);
        close(pipe_subproc_read[0]);
        close(pipe_subproc_read[1]);

        dup2(pipe_subproc_write[1], STDOUT_FILENO);
        close(pipe_subproc_write[0]);
        close(pipe_subproc_write[1]);

        system(("python E3S/parquet_parser.py " + fname).c_str());
        exit(0);
    }
    
    std::vector<std::vector<uint8_t>> texts;
    uint8_t buffer[buffsize], ready = '\n';
    std::fill(buffer, buffer + buffsize, 0);

    while (true) {
        int cnt_bytes = read(pipe_subproc_write[0], buffer, sizeof(buffer));
        
        if (cnt_bytes <= 1) break; ///TODO poate fixezi asta..

        texts.push_back(std::vector<uint8_t>(cnt_bytes));
        std::copy(buffer, buffer + cnt_bytes, texts.back().begin());

        std::fill(buffer, buffer + cnt_bytes, 0);
        write(pipe_subproc_read[1], &ready, 1); ///signal that we are ready for the next string.
        fsync(pipe_subproc_read[1]);
    }

    return texts;
}

int main() {
    auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

    ExpoSizeStrSrc *E3S = new ExpoSizeStrSrc,
                   *E3S_partial = new ExpoSizeStrSrc(E3S); ///we don't expect to find the majority of the strings that we search with E3S, E3S_partial is used to cull them.

    int cnt_parquet_files = 4;
    auto getParquetFpath = [](int parquet_id) {
        return "/export/home/acs/stud/v/vlad_adrian.ulmeanu/E3S_local/llm_copyright/the_pile_10token_strings_cached/train-split-0000" + std::to_string(parquet_id) + ".parquet";
    };

    for (int parquet_id = 0; parquet_id < cnt_parquet_files; parquet_id++) {
        for (std::vector<uint8_t> &text: read_pile_10token_strings(getParquetFpath(parquet_id))) {
            E3S_partial->insertQueriedString(text, text.size(), true);
        }    
    }
    TIMER("(E3S_partial) Loaded parquets")

    std::vector<uint8_t> buffer(E3S->maxn + 1, 0);

    std::ifstream fin("/export/home/acs/stud/v/vlad_adrian.ulmeanu/E3S_local/llm_copyright/outputs_pythia_run1/outputs_concat.txt"); ///std::ios::binary?
    int cnt_last_read_bytes = -1;

    while (cnt_last_read_bytes != 0) {
        fin.read((char *)buffer.data(), E3S->maxn);
        cnt_last_read_bytes = fin.gcount();
        
        if (cnt_last_read_bytes != 0) {
            TIMER("Starting E3S->updateText, buffer size = " + std::to_string(cnt_last_read_bytes))
            E3S->updateText(buffer, cnt_last_read_bytes);
            TIMER("Finished E3S->updateText")

            ///do pre-run only with the first link of the chains attached to the trie.
            E3S_partial->cntStarterNodeChildren = E3S->cntStarterNodeChildren; E3S_partial->n = E3S->n;
            std::fill(E3S_partial->massSearchResults.begin(), E3S_partial->massSearchResults.end(), 0);
            E3S_partial->trieRoot->clear();
            E3S_partial->trieRoot->idLevsCurrentlyHere.emplace_back(-1, INT_MAX);

            E3S_partial->massSearch(E3S_partial->trieRoot);
            TIMER("(E3S_partial) Finished E3S->massSearch")

            ///put into dictionary the whole chains for words we could partially find, then rerun massSearch.

            ///E3S' trie is empty here.
            for (int parquet_id = 0, i = 0; parquet_id < cnt_parquet_files; parquet_id++) {
                for (std::vector<uint8_t> &text: read_pile_10token_strings(getParquetFpath(parquet_id))) {
                    if (E3S_partial->massSearchResults[i] != 0) E3S->insertQueriedString(text, text.size(), false);
                    i++;
                }
            }

            TIMER("Loaded parquets")

            E3S->trieRoot->idLevsCurrentlyHere.emplace_back(-1, INT_MAX);
            E3S->massSearch(E3S->trieRoot);
            TIMER("Finished E3S->massSearch")

            ///do secondary check with sequences of united finds.
            {
                ExpoSizeStrSrc *E3S_check = new ExpoSizeStrSrc(E3S); ///we use this to check for sequences of matches from E3S.
                std::vector<std::vector<uint8_t>> seqs(1);

                int cnt_consecutive = 0;
                for (int parquet_id = 0, i = 0; parquet_id < cnt_parquet_files; parquet_id++) {
                    for (std::vector<uint8_t> &text: read_pile_10token_strings(getParquetFpath(parquet_id))) {
                        if (E3S->massSearchResults[i] != 0) {
                            seqs.back().insert(seqs.back().end(), text.begin(), text.end());
                            cnt_consecutive++;
                        } else if (seqs.back().size() > 0) {
                            if (cnt_consecutive >= 5) E3S_check->insertQueriedString(seqs.back(), seqs.back().size(), false);
                            else seqs.pop_back();

                            seqs.push_back(std::vector<uint8_t>());
                            cnt_consecutive = 0;
                        }

                        i++;
                    }
                }

                if (seqs.back().size() > 0) {
                    if (cnt_consecutive >= 5) E3S_check->insertQueriedString(seqs.back(), seqs.back().size(), false);
                    else seqs.pop_back();
                }

                TIMER("Finished inserting sequences of E3S matches")

                E3S_check->trieRoot->idLevsCurrentlyHere.emplace_back(-1, INT_MAX);
                E3S_check->massSearch(E3S_check->trieRoot);

                TIMER("Finished E3S_check mass search")

                int i = 0;
                for (std::vector<uint8_t> &seq: seqs) {
                    if (E3S_check->massSearchResults[i] != 0) {
                        std::cout << i << ", ";
                        for (uint8_t ch: seq) std::cout << ch;
                        std::cout << '\n';
                    }

                    i++;
                }
            }

            TIMER("Finished printing matched strings.")

            ///we will redo E3S' whole trie next time. (E3S_partial may give different info)
            E3S->massSearchResults.clear();
            E3S->trieBuffersFree();
            E3S->trieRoot = E3S->trieNodeAlloc();
        }
    }

    return 0;
}
