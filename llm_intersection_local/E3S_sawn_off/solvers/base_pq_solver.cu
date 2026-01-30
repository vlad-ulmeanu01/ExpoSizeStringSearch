#include "base_pq_solver.h"

Result::Result(const std::string& t, int count, int l, int r): t(t), count(count), maxlen(l, r) {}

void base_pq_solve(std::string TYPE_SOLVER, int CNT_PARQUET_FILES, int CNT_ATTACK_FILES, std::string OUTFILE) {
    TIMER_START

    std::unique_ptr<DictReader> dire = std::make_unique<ParquetChunkReader>(CNT_PARQUET_FILES); ///q este citit aici.

    DBG(dire->get_q())
    TIMER_SPLIT("computed q")

    E3S_sawnoff e3ss_first, e3ss_others;

    std::vector<uint8_t> s;
    int ATTACKS_PER_ITER = (TYPE_SOLVER == "base_pq"? CNT_ATTACK_FILES: 1);
    s.reserve(ATTACKS_PER_ITER * BATCH_FILE_MAX_SIZE);
    for (int attack_offset = 0; attack_offset < CNT_ATTACK_FILES; attack_offset += ATTACKS_PER_ITER) {
        s.clear();
        for (int i = attack_offset; i < attack_offset + ATTACKS_PER_ITER; i++) {
            std::ifstream fin(ATTACK_DIR + "batch_" + std::to_string(i) + ".txt");
            std::string line;
            while (std::getline(fin, line)) {
                line += '\n';
                s.insert(s.end(), line.begin(), line.end());
            }
        }

        TIMER_SPLIT("read attack files")
        
        e3ss_first.update_s(s, attack_offset > 0);
        e3ss_others.update_s(s, attack_offset > 0);
        TIMER_SPLIT("update_s")

        if (attack_offset == 0) {
            ///tin e3ss exclusiv pentru prima cautare. are sens spargerea doar pt > 1 iteratie, e.g. revin asupra dict cu alt s.
            e3ss_first.read_ts(std::move(dire)); ///! in modul in care e scris e3ss, trebuie dev_base_pws pentru read_ts.
            TIMER_SPLIT("e3ss_first read_ts")
        }

        e3ss_first.compute_ts_counts(); ///rezultatul folosibil este in hst_ts_count.
        TIMER_SPLIT("compute_ts_counts")

        e3ss_first.dire->reset_iter();
        std::unique_ptr<DictReader> dire_inter = std::make_unique<InterReader>();
        
        std::vector<uint8_t> chain_vt, prev_vt0; ///prev_vt0: the last chunk with a count of 0.
        for (int i = 0; i < e3ss_first.dire->get_q(); i++) {
            std::vector<uint8_t> vt = e3ss_first.dire->get_next_t();

            if (e3ss_first.hst_ts_count[i] > 0) {
                chain_vt.insert(chain_vt.end(), vt.begin(), vt.end());
            }

            if (chain_vt.size() > 0 && (e3ss_first.hst_ts_count[i] == 0 || i+1 == e3ss_first.dire->get_q())) {
                dire_inter->receive_t(chain_vt, prev_vt0, (e3ss_first.hst_ts_count[i] == 0? vt: std::vector<uint8_t>()));
                chain_vt.clear();
            }

            if (e3ss_first.hst_ts_count[i] == 0) prev_vt0 = vt;
        }

        DBG(dire_inter->get_q()) TIMER_SPLIT("fill dire_inter")
        e3ss_others.read_ts(std::move(dire_inter)); TIMER_SPLIT("dire_inter read_ts")
        e3ss_others.compute_ts_counts(); TIMER_SPLIT("dire_inter compute_ts_counts")

        e3ss_others.dire->reset_iter();
        dire_inter = std::make_unique<InterReader>();

        int pas = get_msb(PARQUET_BYTES_PER_READ);
        for (int i = 0; i < e3ss_others.dire->get_q(); i++) {
            std::vector<uint8_t> vt = e3ss_others.dire->get_next_t();
            auto [vt_l, vt_r] = e3ss_others.dire->get_next_neighs();

            if (e3ss_others.hst_ts_count[i] > 0) {
                dire_inter->receive_t(vt, vt_l, vt_r); ///incerc sa expandez si bucata asta, dar doar peste bucatile de 100 adiacente pe care nu le-am gasit.
            } else { ///altfel iau toate bucatile de 100 si incerc sa le expandez (cat de mult pot, eg pana la santilele, nu incerc doar sa le aduc peste 200).
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
        for (int i = 0; i < dire_inter->get_q(); i++) count_l[i] = count_r[i] = e3ss_others.hst_ts_count[i]; ///count_l/r tine minte count pt offset maxim pe l/r.

        for (; pas; pas >>= 1) {
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

            e3ss_others.read_ts(std::move(dire_tmp));
            e3ss_others.compute_ts_counts();

            dire_inter->reset_iter();
            for (int i = 0; i < dire_inter->get_q(); i++) {
                auto [vt_l, vt_r] = dire_inter->get_next_neighs();
                for (int dir = 0; dir < 2; dir++) {
                    if (e3ss_others.hst_ts_count[2*i+dir] > 0) {
                        if (dir == 0) {
                            offsets_l[i] = std::min((int)vt_l.size(), offsets_l[i] + pas);
                            count_l[i] = e3ss_others.hst_ts_count[2*i+dir];
                        } else {
                            offsets_r[i] = std::min((int)vt_r.size(), offsets_r[i] + pas);
                            count_r[i] = e3ss_others.hst_ts_count[2*i+dir];
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

        e3ss_others.read_ts(std::move(dire_try));
        e3ss_others.compute_ts_counts();

        TIMER_SPLIT("search with extremes united")

        std::vector<Result> results;
        dire_inter->reset_iter();
        e3ss_others.dire->reset_iter();
        for (int i = 0; i < dire_inter->get_q(); i++) {
            std::vector<uint8_t> vt = e3ss_others.dire->get_next_t();
            std::vector<uint8_t> vti = dire_inter->get_next_t();
            auto [vti_l, vti_r] = dire_inter->get_next_neighs();

            if (e3ss_others.hst_ts_count[i] > 0) { ///am gasit si extensia maxima.
                std::string t(vt.begin(), vt.end());
                if (t.size() >= 2 * PARQUET_BYTES_PER_READ) {
                    results.emplace_back(t, e3ss_others.hst_ts_count[i], t.size(), t.size());
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

        std::string outfile = OUTFILE;
        if (TYPE_SOLVER == "iter_attack_pq") {    
            while (outfile.back() != '.') outfile.pop_back();
            outfile.pop_back();
            outfile += "_" + std::to_string(attack_offset) + ".txt";
        }
        std::ofstream fout(outfile);

        for (const Result& r: results) {
            fout << "count = " << r.count << ", maxlen_lo = " << r.maxlen.first << ", maxlen_hi = " << r.maxlen.second << '\n' << r.t << "\n---\n";
        }

        DBG(attack_offset)
        TIMER_SPLIT("fout write")
    }
}
