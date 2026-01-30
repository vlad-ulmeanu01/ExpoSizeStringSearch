#pragma once
#include "../utils.h"
#include "../readers.h"
#include "../e3s_sawnoff.h"

///bucatile de intersectie gasite.
struct Result {
    std::string t;
    int count;
    std::pair<int, int> maxlen; ///posibil sa avem doar o estimare pentru maxlen. valoarea adevarata e in [.fi, .se].

    Result(const std::string& t, int count, int l, int r);
};

void base_pq_solve(std::string TYPE_SOLVER, int CNT_PARQUET_FILES, int CNT_ATTACK_FILES, std::string OUTFILE);
