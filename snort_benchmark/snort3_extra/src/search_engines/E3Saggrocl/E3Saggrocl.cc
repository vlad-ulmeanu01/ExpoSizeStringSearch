/*
*   An abstracted interface to the Multi-Pattern Matching routines,
*   that's why we're passing 'void *' objects around.
*
*/

#include "log/messages.h"
#include "framework/mpse.h"

#include "E3Saggrocl_utils.h"

using namespace snort;

//-------------------------------------------------------------------------
// "E3Saggrocl"
//-------------------------------------------------------------------------

class E3SaggroclMpse : public Mpse
{
private:
    static constexpr uint32_t ct229 = (1 << 29) - 1;
    static constexpr uint64_t M61 = (1ULL << 61) - 1, M61_2x = M61 * 2;
    static constexpr int maxn = 65'535, max_ml2 = 16;

    struct PatternInfo {
        ChainInfo ci;
        void *user;
        void *tree;
        void *neg_list;
        bool no_case;
        bool negated;

        PatternInfo(void *user, void *tree, void *neg_list, bool no_case, bool negated) {
            this->user = user;
            this->tree = tree;
            this->neg_list = neg_list;
            this->no_case = no_case;
            this->negated = negated;
        }
    };

    const MpseAgent *agent;
    SharedInfo *sharedInfo;
    ExpoSizeStrSrc *E3Saggrocl;

    std::vector<PatternInfo> patterns;
    std::unordered_set<uint64_t> patternHashes; ///the hashes of the patterns. don't want to work with duplicates.
    std::vector<LinkInfo> connections; ///the patterns' hash chain connections I am interested in finding out whether they exist or not in the DAG.

    std::vector<uint8_t> tmp_buffer;
    uint8_t tolower_lookup[256];

    ///the two functions below are exclusively used for preprocessQueriedString.
    ///a1 = a1 * b1 % M61.
    ///a2 = a2 * b2 % M61.
    static void mul2x(uint64_t &a1, const uint64_t &b1, uint64_t &a2, const uint64_t &b2) {
        uint64_t a1_hi = a1 >> 32, a1_lo = (uint32_t)a1, b1_hi = b1 >> 32, b1_lo = (uint32_t)b1,
                 a2_hi = a2 >> 32, a2_lo = (uint32_t)a2, b2_hi = b2 >> 32, b2_lo = (uint32_t)b2,
                 ans_1 = 0, ans_2 = 0, tmp_1 = 0, tmp_2 = 0;

        tmp_1 = a1_hi * b1_lo + a1_lo * b1_hi;
        tmp_2 = a2_hi * b2_lo + a2_lo * b2_hi;

        tmp_1 = ((tmp_1 & ct229) << 32) + (tmp_1 >> 29);
        tmp_2 = ((tmp_2 & ct229) << 32) + (tmp_2 >> 29);

        tmp_1 += (a1_hi * b1_hi) << 3;
        tmp_2 += (a2_hi * b2_hi) << 3;

        ans_1 = (tmp_1 >> 61) + (tmp_1 & M61);
        ans_2 = (tmp_2 >> 61) + (tmp_2 & M61);

        tmp_1 = a1_lo * b1_lo;
        tmp_2 = a2_lo * b2_lo;

        ans_1 += (tmp_1 >> 61) + (tmp_1 & M61);
        ans_2 += (tmp_2 >> 61) + (tmp_2 & M61);

        ans_1 = (ans_1 >= M61_2x? ans_1 - M61_2x: (ans_1 >= M61? ans_1 - M61: ans_1));
        ans_2 = (ans_2 >= M61_2x? ans_2 - M61_2x: (ans_2 >= M61? ans_2 - M61: ans_2));

        a1 = ans_1;
        a2 = ans_2;
    }

    ///128bit hash ---(xorshift)---> uniform spread that fits in 8 bytes.
    static uint64_t reduceHash(std::pair<uint64_t, uint64_t> &hh, std::pair<uint64_t, uint64_t> otp) {
        otp.first ^= hh.first;
        otp.second ^= hh.second;

        ///Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)
        ///https://prng.di.unimi.it/splitmix64.c
        ///https://prng.di.unimi.it/xoshiro256plusplus.c
        ///https://vigna.di.unimi.it/ftp/papers/xorshift.pdf

        //xoshiro256pp::seed_and_get(otp.first, otp.second):
        otp.first += 0x9e3779b97f4a7c15; ///s[0] from xoshiro256plusplus.
        otp.first = (otp.first ^ (otp.first >> 30)) * 0xbf58476d1ce4e5b9;
        otp.first = (otp.first ^ (otp.first >> 27)) * 0x94d049bb133111eb;
        otp.first = otp.first ^ (otp.first >> 31);

        otp.second += 0x3c6ef372fe94f82a; ///s[3] from xoshiro256plusplus. 0x9e3779b97f4a7c15 * 2 - 2**64.
        otp.second = (otp.second ^ (otp.second >> 30)) * 0xbf58476d1ce4e5b9;
        otp.second = (otp.second ^ (otp.second >> 27)) * 0x94d049bb133111eb;
        otp.second = otp.second ^ (otp.second >> 31);

        //rotl(hh1 + hh2, 23) + hh1:
        otp.second += otp.first;
        return ((otp.second << 23) | (otp.second >> 41)) + otp.first;
    }

public:
    E3SaggroclMpse(const MpseAgent* agent, SharedInfo *sharedInfo, ExpoSizeStrSrc *E3Saggrocl) : Mpse("E3Saggrocl") {
        this->agent = agent;

        std::iota(tolower_lookup, tolower_lookup + 256, 0);
        for (int i = 'A'; i <= (int)'Z'; i++) tolower_lookup[i] ^= 32;

        this->sharedInfo = sharedInfo;
        this->E3Saggrocl = E3Saggrocl;
    }

    ~E3SaggroclMpse() override {

    }

    /**
    * aggrocl is inherently online. since we will search the same string over different texts, we should remember in the pre-search phase the hash chain.
    * @param t the string in question: t[0 .. lengthT - 1]. t already comes in case non-sensitive format.
    * @param lengthT. The length of t.
    * @param ci.fullHash. the entire hash of t.
    * @param ci.chainLength. must be filled in this function. the length of the hash chain.
    * @param ci.exponents. also must be filled here. the powers of two that make lengthT, in decreasing order. (i.e. for 11, [8, 2, 1])
    * @param ci.t_hashes. also must be filled here. the elements of the hash chain.
    */
    void preprocessQueriedString(const std::vector<uint8_t> &t, int lengthT, ChainInfo &ci) {
        ci.chainLength = __builtin_popcount(lengthT);

        ///split t into a substring chain, each substring having a distinct power of 2 length.

        int i, j, z, k = 0;
        std::pair<uint64_t, uint64_t> hh, fullHh128(0, 0);
        for (i = max_ml2, z = 0; i >= 0; i--) {
            if (lengthT & (1<<i)) {
                hh = std::make_pair(0, 0);
                for (j = z + (1<<i); z < j; z++) {
                    mul2x(hh.first, sharedInfo->base.first, hh.second, sharedInfo->base.second);
                    mul2x(fullHh128.first, sharedInfo->base.first, fullHh128.second, sharedInfo->base.second);

                    hh.first += (int)t[z] + 1;
                    hh.second += (int)t[z] + 1;

                    fullHh128.first += (int)t[z] + 1;
                    fullHh128.second += (int)t[z] + 1;

                    hh.first = (hh.first >= M61? hh.first - M61: hh.first);
                    hh.second = (hh.second >= M61? hh.second - M61: hh.second);

                    fullHh128.first = (fullHh128.first >= M61? fullHh128.first - M61: fullHh128.first);
                    fullHh128.second = (fullHh128.second >= M61? fullHh128.second - M61: fullHh128.second);
                }

                ci.exponents[k] = i;
                ci.t_hashes[k++] = reduceHash(hh, sharedInfo->logOtp[i]);
            }
        }

        ci.fullHash = reduceHash(fullHh128, std::make_pair(0, 0));
    }

    /**
     * @param P the new pattern.
     * @param m length of the pattern.
     * @param desc found in mpse.h. info about no_case, negated, literal, multi_match, flags.
     * @param user ???
     * @return 0 for succes.
     */
    int add_pattern(const uint8_t* P, unsigned m, const PatternDescriptor& desc, void* user) override {
        patterns.emplace_back(user, nullptr, nullptr, desc.no_case, desc.negated);

        if (m > (int)tmp_buffer.size()) {
            tmp_buffer.resize(m);
        }

        std::copy(P, P + m, tmp_buffer.begin());
        for (int i = 0; i < m; i++) tmp_buffer[i] = tolower_lookup[tmp_buffer[i]];

        preprocessQueriedString(tmp_buffer, m, patterns.back().ci);

        if (patternHashes.count(patterns.back().ci.fullHash)) {
            patterns.pop_back();
        } else {
            patternHashes.insert(patterns.back().ci.fullHash);
        }

        for (int i = 0; i < patterns.back().ci.chainLength - 1; i++) {
            connections.emplace_back(patterns.back().ci.t_hashes[i], patterns.back().ci.t_hashes[i+1], patterns.back().ci.exponents[i+1]);
        }

        if (patterns.back().ci.chainLength == 1) {
            ///we must still check for the existence of the head of the chain. add a false link.
            ///there is no link that looks like (hh1, hh1) (it would imply that both nodes' strings have the same length).
            connections.emplace_back(patterns.back().ci.t_hashes[0], patterns.back().ci.t_hashes[0]);
        }

        return 0;
    }

    /**
     * probably called after all add_pattern().
     * @param sc found in ??? (snort::SnortConfig). Doesn't seem modified anywhere by lowmem.
     * @return 0 for succes.
     */
    int prep_patterns(SnortConfig* sc) override {
        //std::cout << "prep_patterns: nr patternuri = " << (int)patterns.size() << '\n' << std::flush;

        std::sort(connections.begin(), connections.end());
        connections.resize(std::unique(connections.begin(), connections.end()) - connections.begin());

        for (auto &p: patterns) {
            if (p.user) {
                if (!p.negated) agent->build_tree(sc, p.user, &p.tree);
                else agent->negate_list(p.user, &p.neg_list);
            }

            for (int i = 0; i < p.ci.chainLength - 1; i++) {
                p.ci.massSearchIds[i] = std::lower_bound(connections.begin(), connections.end(), LinkInfo(p.ci.t_hashes[i], p.ci.t_hashes[i+1])) - connections.begin();
            }

            if (p.ci.chainLength == 1) {
                p.ci.massSearchIds[0] = std::lower_bound(connections.begin(), connections.end(), LinkInfo(p.ci.t_hashes[0], p.ci.t_hashes[0])) - connections.begin();
            }
        }

        return 0;
    }

    /**
     *
     * @param T text in which we have to search all patterns.
     * @param n length of T.
     * @param match extern call, in case we find a match. may exit the search quicker. matches that begin sooner are preferred.
     * @param context
     * @param current_state
     * @return count of found patterns. if the first pattern was found 2x, and the second one 1x, return 3.
     */
    int _search(const uint8_t* T, int n, MpseMatch match, void* context, int* current_state) override {
        if (n > (int)tmp_buffer.size()) {
            tmp_buffer.resize(n);
        }

        std::copy(T, T + n, tmp_buffer.begin());
        for (int i = 0; i < n; i++) tmp_buffer[i] = tolower_lookup[tmp_buffer[i]];

        E3Saggrocl->updateText(tmp_buffer, n, connections);
        E3Saggrocl->massSearch(connections);

        int matches = 0;
        for (auto &p: patterns) {
            bool found = true;
            for (int i = 0; i < p.ci.chainLength - 1 && found; i++) {
                found &= connections[p.ci.massSearchIds[i]].found;
            }

            if (p.ci.chainLength == 1) {
                found = connections[p.ci.massSearchIds[0]].found;
            }

            if (found) {
                matches++;
                if (match(p.user, p.tree, 0, context, p.neg_list) > 0) {
                    return matches;
                }
            }
        }

        // std::cout << "E3Saggrocl matches = " << matches << '\n' << std::flush;

        return matches;
    }

    /**
     * how many times was add_pattern called?
     * @return how many patterns do I have.
     */
    int get_pattern_count() const override {
        return (int)patterns.size();
    }
};

//-------------------------------------------------------------------------
// api
//-------------------------------------------------------------------------

ExpoSizeStrSrc *E3Saggrocl = nullptr;
SharedInfo *sharedInfo = nullptr;
int refcnt_mpse = 0;

static Mpse* e3saggrocl_ctor(const SnortConfig*, class Module*, const MpseAgent* agent)
{
    ///snort wishes to split the dictionary into multiple parts. it does so by creating multiple Mpse classes.
    refcnt_mpse++;

    return new E3SaggroclMpse(agent, sharedInfo, E3Saggrocl);
}

static void e3saggrocl_dtor(Mpse* p)
{
    delete p;
    
    refcnt_mpse--;
    if (refcnt_mpse == 0) {
        delete E3Saggrocl;
        delete sharedInfo;
    }
}

static void e3saggrocl_init()
{
    ///called only once.
    sharedInfo = new SharedInfo;
    E3Saggrocl = new ExpoSizeStrSrc(sharedInfo);
}

static void e3saggrocl_print()
{
    LogMessage("E3Saggrocl.");
}

static const MpseApi e3saggrocl_api =
{
    {
        PT_SEARCH_ENGINE,
        sizeof(MpseApi),
        SEAPI_VERSION,
        0,
        API_RESERVED,
        API_OPTIONS,
        "E3Saggrocl",
        "E3Saggrocl MPSE",
        nullptr,
        nullptr
    },
    MPSE_BASE,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    e3saggrocl_ctor,
    e3saggrocl_dtor,
    e3saggrocl_init,
    e3saggrocl_print,
    nullptr
};

const BaseApi* se_E3Saggrocl = &e3saggrocl_api.base;
