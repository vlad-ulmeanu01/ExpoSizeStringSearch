#include "utils.h"
#include "readers.h"
#include "e3s_sawnoff.h"

int main(int argc, char **argv) {
    TIMER_START

    ///cses test structure: ./main CNT_PARQUET_FILES CNT_ATTACK_FILES OUTFILE INFILE
    ///parquet structure: ./main CNT_PARQUET_FILES CNT_ATTACK_FILES OUTFILE 
    assert(argc >= 4);

    int CNT_PARQUET_FILES = atoi(argv[1]); ///daca este < 0 => format CSES.
    int CNT_ATTACK_FILES = atoi(argv[2]); ///trebuie sa fie >= 1.
    DBG(CNT_PARQUET_FILES); DBG(CNT_ATTACK_FILES);

    std::string OUTFILE = std::string(argv[3]);
    std::ofstream fout(OUTFILE);

    std::vector<uint8_t> s;

    std::unique_ptr<DictReader> dire;
    if (CNT_PARQUET_FILES < 0) {
        assert(argc == 5);
        std::string INFILE = std::string(argv[4]);
        std::ifstream fin(INFILE);
        std::string str_s; fin >> str_s;
        s.resize(str_s.size());
        std::copy(str_s.begin(), str_s.end(), s.begin());
        dire = std::make_unique<CsesReader>(std::move(fin));
    } else {
        assert(argc == 4);

        s.reserve(CNT_ATTACK_FILES * BATCH_FILE_MAX_SIZE);

        for (int i = 0; i < CNT_ATTACK_FILES; i++) {
            std::ifstream fin(ATTACK_DIR + "batch_" + std::to_string(i) + ".txt");
            std::string line;
            while (std::getline(fin, line)) {
                line += '\n';
                s.insert(s.end(), line.begin(), line.end());
            }
        }

        dire = std::make_unique<ParquetChunkReader>(CNT_PARQUET_FILES); ///q este citit aici.
    }

    DBG(dire->get_q());
    TIMER_SPLIT("read attack files, computed q")

    E3S_sawnoff e3ss;

    e3ss.update_s(s);
    TIMER_SPLIT("update_s")

    e3ss.read_ts(std::move(dire)); ///pentru refacere tb alt dire, nu tb resetat dev_ts_info (o sa fie suprascris).
    TIMER_SPLIT("read_ts")

    e3ss.compute_ts_counts(); ///rezultatul folosibil este in hst_ts_count.
    TIMER_SPLIT("compute_ts_counts")

    {
        std::unique_ptr<DictReader> dire2 = std::make_unique<ParquetChunkReader>(CNT_PARQUET_FILES);

        for (int i = 0; i < dire2->get_q(); i++) {
            std::vector<uint8_t> vt = dire2->get_next_t();
            if (e3ss.hst_ts_count[i] > 0) {
                std::string t(vt.begin(), vt.end());
                fout << e3ss.hst_ts_count[i] << '\n' << t << "\n---\n";
            }
        }
    }

    return 0;
}
