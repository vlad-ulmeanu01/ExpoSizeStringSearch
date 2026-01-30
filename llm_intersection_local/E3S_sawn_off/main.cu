#include "solvers/cses_solver.h"
#include "solvers/base_pq_solver.h"

int main(int argc, char **argv) {
    std::string TYPE_SOLVER = std::string(argv[1]);    
    DBG(TYPE_SOLVER);

    if (TYPE_SOLVER == "cses") {
        assert(argc == 4);
        std::string INFILE = std::string(argv[2]);
        std::string OUTFILE = std::string(argv[3]);

        cses_solve(INFILE, OUTFILE);
    } else if (TYPE_SOLVER == "base_pq" || TYPE_SOLVER == "iter_attack_pq") {
        assert(argc == 5);

        int CNT_PARQUET_FILES = atoi(argv[2]);
        int CNT_ATTACK_FILES = atoi(argv[3]);
        std::string OUTFILE = std::string(argv[4]);

        DBG(CNT_PARQUET_FILES);
        DBG(CNT_ATTACK_FILES);

        base_pq_solve(TYPE_SOLVER, CNT_PARQUET_FILES, CNT_ATTACK_FILES, OUTFILE);
    } else {
        assert(false);
    }

    return 0;
}
