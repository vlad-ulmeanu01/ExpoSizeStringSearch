#include "cses_solver.h"

void cses_solve(std::string INFILE, std::string OUTFILE) {
    TIMER_START

    std::vector<uint8_t> s;
    std::unique_ptr<DictReader> dire;

    std::ifstream fin(INFILE);
    std::string str_s; fin >> str_s;
    s.resize(str_s.size());
    std::copy(str_s.begin(), str_s.end(), s.begin());
    dire = std::make_unique<CsesReader>(std::move(fin));

    DBG(dire->get_q())
    TIMER_SPLIT("read attack files, computed q")

    E3S_sawnoff e3ss;

    e3ss.update_s(s, false);
    TIMER_SPLIT("update_s")

    e3ss.read_ts(std::move(dire)); ///pentru refacere tb alt dire, nu tb resetat dev_ts_info (o sa fie suprascris).
    TIMER_SPLIT("read_ts")

    e3ss.compute_ts_counts(); ///rezultatul folosibil este in hst_ts_count.
    TIMER_SPLIT("compute_ts_counts")

    std::ofstream fout(OUTFILE);
    for (int cnt: e3ss.hst_ts_count) fout << cnt << '\n';
}
