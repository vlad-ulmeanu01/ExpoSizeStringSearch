#include "utils.h"
#include "readers.h"
#include "e3s_sawnoff.h"

int main(int argc, char **argv) {
    assert(argc >= 2);
    std::ofstream fout(argv[1]);

    std::vector<uint8_t> s;

    std::unique_ptr<DictReader> dire;
    if (CNT_PARQUET_FILES < 0) {
        assert(argc == 3);
        std::ifstream fin(argv[2]);
        std::string str_s; fin >> str_s;
        s.resize(str_s.size());
        std::copy(str_s.begin(), str_s.end(), s.begin());
        dire = std::make_unique<CsesReader>(std::move(fin));
    } else {
        assert(argc == 2);
        ///todo citeste str atac.
        dire = std::make_unique<ParquetChunkReader>(); ///q este citit aici.
    }

    E3S_sawnoff e3ss;

    e3ss.update_s(s);
    e3ss.read_ts(std::move(dire)); ///pentru refacere tb alt dire, nu tb resetat dev_ts_info (o sa fie suprascris).
    e3ss.compute_ts_counts(); ///rezultatul folosibil este in hst_ts_count.

    for (int cnt: e3ss.hst_ts_count) fout << cnt << '\n';

    return 0;
}
