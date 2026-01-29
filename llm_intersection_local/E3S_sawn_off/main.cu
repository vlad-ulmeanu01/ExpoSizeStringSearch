#include "utils.h"
#include "readers.h"
#include "e3s_sawnoff.h"

///bucatile de intersectie gasite.
struct Result {
    std::string t;
    int count;
    std::pair<int, int> maxlen; ///posibil sa avem doar o estimare pentru maxlen. valoarea adevarata e in [.fi, .se].

    Result(const std::string& t, int count, int l, int r): t(t), count(count), maxlen(l, r) {}
};


int main(int argc, char **argv) {
    TIMER_START

    ///cses test structure: ./main CNT_PARQUET_FILES CNT_ATTACK_FILES OUTFILE INFILE
    ///parquet structure: ./main CNT_PARQUET_FILES CNT_ATTACK_FILES OUTFILE 
    assert(argc >= 4);

    int CNT_PARQUET_FILES = atoi(argv[1]); ///daca este < 0 => format CSES.
    int CNT_ATTACK_FILES = atoi(argv[2]); ///trebuie sa fie >= 1.
    DBG(CNT_PARQUET_FILES); DBG(CNT_ATTACK_FILES);

    std::string OUTFILE = std::string(argv[3]);
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

    DBG(dire->get_q())
    TIMER_SPLIT("read attack files, computed q")

    E3S_sawnoff e3ss;

    e3ss.update_s(s, false);
    TIMER_SPLIT("update_s")

    e3ss.read_ts(std::move(dire)); ///pentru refacere tb alt dire, nu tb resetat dev_ts_info (o sa fie suprascris).
    TIMER_SPLIT("read_ts")

    e3ss.compute_ts_counts(); ///rezultatul folosibil este in hst_ts_count.
    TIMER_SPLIT("compute_ts_counts")

    if (CNT_PARQUET_FILES < 0) {
        std::ofstream fout(OUTFILE);
        for (int cnt: e3ss.hst_ts_count) fout << cnt << '\n';
        return 0;
    }

    e3ss.dire->reset_iter();
    std::unique_ptr<DictReader> dire_inter = std::make_unique<InterReader>();
    
    std::vector<uint8_t> chain_vt, prev_vt0; ///prev_vt0: the last chunk with a count of 0.
    for (int i = 0; i < e3ss.dire->get_q(); i++) {
        std::vector<uint8_t> vt = e3ss.dire->get_next_t();

        if (e3ss.hst_ts_count[i] > 0) {
            chain_vt.insert(chain_vt.end(), vt.begin(), vt.end());
        }

        if (chain_vt.size() > 0 && (e3ss.hst_ts_count[i] == 0 || i+1 == e3ss.dire->get_q())) {
            dire_inter->receive_t(chain_vt, prev_vt0, (e3ss.hst_ts_count[i] == 0? vt: std::vector<uint8_t>()));
            chain_vt.clear();
        }

        if (e3ss.hst_ts_count[i] == 0) prev_vt0 = vt;
    }

    DBG(dire_inter->get_q()) TIMER_SPLIT("fill dire_inter")
    e3ss.read_ts(std::move(dire_inter)); TIMER_SPLIT("dire_inter read_ts")
    e3ss.compute_ts_counts(); TIMER_SPLIT("dire_inter compute_ts_counts")

    e3ss.dire->reset_iter();
    dire_inter = std::make_unique<InterReader>();
    for (int i = 0; i < e3ss.dire->get_q(); i++) {
        std::vector<uint8_t> vt = e3ss.dire->get_next_t();
        auto [vt_l, vt_r] = e3ss.dire->get_next_neighs();

        if (e3ss.hst_ts_count[i] > 0) {
            dire_inter->receive_t(vt, vt_l, vt_r); ///incerc sa expandez si bucata asta, dar doar peste bucatile de 100 adiacente pe care nu le-am gasit.
        } else { ///altfel iau toate bucatile de 100 si incerc sa le expandez (e.g. incerc doar sa le aduc peste 200).
            for (int j = 0; j < (int)vt.size(); j += PARQUET_BYTES_PER_READ) {
                dire_inter->receive_t(
                    std::vector<uint8_t>(vt.begin() + j, vt.begin() + std::min((int)vt.size(), j + PARQUET_BYTES_PER_READ)),
                    (j == 0? vt_l: std::vector<uint8_t>(vt.begin() + j - PARQUET_BYTES_PER_READ, vt.begin() + j)),
                    (j + PARQUET_BYTES_PER_READ >= (int)vt.size()?
                        vt_r:
                        std::vector<uint8_t>(vt.begin() + j + PARQUET_BYTES_PER_READ, vt.begin() + std::min((int)vt.size(), j + 2 * PARQUET_BYTES_PER_READ))
                    )
                );
            }
        }
    }

    DBG(dire_inter->get_q())
    TIMER_SPLIT("build dire_inter with sentinels")

    std::vector<int> offsets_l(dire_inter->get_q()), offsets_r(dire_inter->get_q()), count_l(dire_inter->get_q()), count_r(dire_inter->get_q());
    for (int i = 0; i < dire_inter->get_q(); i++) count_l[i] = count_r[i] = e3ss.hst_ts_count[i]; ///count_l/r tine minte count pt offset maxim pe l/r.

    for (int pas = get_msb(PARQUET_BYTES_PER_READ); pas; pas >>= 1) {
        std::unique_ptr<DictReader> dire_tmp = std::make_unique<InterReader>();

        dire_inter->reset_iter();
        for (int i = 0; i < dire_inter->get_q(); i++) {
            std::vector<uint8_t> vt = dire_inter->get_next_t();
            auto [vt_l, vt_r] = dire_inter->get_next_neighs();

            for (int dir = 0; dir < 2; dir++) { ///0 pentru left, 1 pentru right.
                std::vector<uint8_t> bs;

                if (dir == 0) {
                    bs.insert(bs.end(), vt_l.end() - std::min(offsets_l[i] + pas, (int)vt_l.size()), vt_l.end());
                    bs.insert(bs.end(), vt.begin(), vt.end());
                } else {
                    bs.insert(bs.end(), vt.begin(), vt.end());
                    bs.insert(bs.end(), vt_r.begin(), vt_r.begin() + std::min(offsets_r[i] + pas, (int)vt_r.size()));
                }

                dire_tmp->receive_t(bs, std::vector<uint8_t>(), std::vector<uint8_t>());
            }
        }

        e3ss.read_ts(std::move(dire_tmp));
        e3ss.compute_ts_counts();

        dire_inter->reset_iter();
        for (int i = 0; i < dire_inter->get_q(); i++) {
            auto [vt_l, vt_r] = dire_inter->get_next_neighs();
            for (int dir = 0; dir < 2; dir++) {
                if (e3ss.hst_ts_count[2*i+dir] > 0) {
                    if (dir == 0) {
                        offsets_l[i] = std::min((int)vt_l.size(), offsets_l[i] + pas);
                        count_l[i] = e3ss.hst_ts_count[2*i+dir];
                    } else {
                        offsets_r[i] = std::min((int)vt_r.size(), offsets_r[i] + pas);
                        count_r[i] = e3ss.hst_ts_count[2*i+dir];
                    }
                }
            }
        }

        DBG(pas)
        TIMER_SPLIT("finished bs step")
    }

    ///incerc sa unesc ext maxima la st cu originalul cu ext maxima la dr.
    std::unique_ptr<DictReader> dire_try = std::make_unique<InterReader>();
    dire_inter->reset_iter();
    for (int i = 0; i < dire_inter->get_q(); i++) {
        std::vector<uint8_t> vt = dire_inter->get_next_t();
        auto [vt_l, vt_r] = dire_inter->get_next_neighs();

        std::vector<uint8_t> bs;

        bs.insert(bs.end(), vt_l.end() - offsets_l[i], vt_l.end());
        bs.insert(bs.end(), vt.begin(), vt.end());
        bs.insert(bs.end(), vt_r.begin(), vt_r.begin() + offsets_r[i]);

        dire_try->receive_t(bs, std::vector<uint8_t>(), std::vector<uint8_t>());
    }

    e3ss.read_ts(std::move(dire_try));
    e3ss.compute_ts_counts();

    TIMER_SPLIT("search with extremes united")

    std::vector<Result> results;
    dire_inter->reset_iter();
    e3ss.dire->reset_iter();
    for (int i = 0; i < dire_inter->get_q(); i++) {
        std::vector<uint8_t> vt = e3ss.dire->get_next_t();
        std::vector<uint8_t> vti = dire_inter->get_next_t();
        auto [vti_l, vti_r] = dire_inter->get_next_neighs();

        if (e3ss.hst_ts_count[i] > 0) { ///am gasit si extensia maxima.
            std::string t(vt.begin(), vt.end());
            if (t.size() >= 2 * PARQUET_BYTES_PER_READ) {
                results.emplace_back(t, e3ss.hst_ts_count[i], t.size(), t.size());
            }
        } else { ///altfel ma multumesc cu gasirea de lungime max(offsets_l[i], offsets_r[i]) + dire_inter->ts[i].size().
            std::string t;
            if (offsets_l[i] >= offsets_r[i]) {
                t.insert(t.end(), vti_l.end() - offsets_l[i], vti_l.end());
                t.insert(t.end(), vti.begin(), vti.end());
            } else {
                t.insert(t.end(), vti.begin(), vti.end());
                t.insert(t.end(), vti_r.begin(), vti_r.begin() + offsets_r[i]);
            }

            if ((int)vt.size() - 1 >= 2 * PARQUET_BYTES_PER_READ) {
                results.emplace_back(t, (offsets_l[i] >= offsets_r[i]? count_l[i]: count_r[i]), t.size(), (int)vt.size() - 1);
            }
        }
    }

    std::ofstream fout(OUTFILE);
    for (const Result& r: results) {
        fout << "count = " << r.count << ", maxlen_lo = " << r.maxlen.first << ", maxlen_hi = " << r.maxlen.second << '\n' << r.t << "\n---\n";
    }
    TIMER_SPLIT("fout write")

    // {
    //     std::shared_ptr<arrow::io::FileOutputStream> outfile;

    //     PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(OUTFILE));

    //     parquet::WriterProperties::Builder builder;
    //     std::shared_ptr<parquet::schema::GroupNode> schema;

    //     parquet::schema::NodeVector fields = {
    //         parquet::schema::PrimitiveNode::Make("count", parquet::Repetition::REQUIRED, parquet::LogicalType::Int(32, true), parquet::Type::INT32),
    //         parquet::schema::PrimitiveNode::Make("maxlen_lo", parquet::Repetition::REQUIRED, parquet::LogicalType::Int(32, true), parquet::Type::INT32),
    //         parquet::schema::PrimitiveNode::Make("maxlen_hi", parquet::Repetition::REQUIRED, parquet::LogicalType::Int(32, true), parquet::Type::INT32),
    //         parquet::schema::PrimitiveNode::Make("t", parquet::Repetition::REQUIRED, parquet::LogicalType::String(), parquet::Type::BYTE_ARRAY)
    //     };

    //     schema = std::static_pointer_cast<parquet::schema::GroupNode>(parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));

    //     parquet::StreamWriter os(parquet::ParquetFileWriter::Open(outfile, schema, builder.build()));
    //     for (const Result& r: results) {
    //         os << r.count << r.maxlen.first << r.maxlen.second << r.t << parquet::EndRow;
    //     }
        
    //     TIMER_SPLIT("parquet write")
    // }

    return 0;
}
