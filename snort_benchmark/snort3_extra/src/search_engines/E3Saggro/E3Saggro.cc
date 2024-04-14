/*
*   An abstracted interface to the Multi-Pattern Matching routines,
*   that's why we're passing 'void *' objects around.
*
*/

#include "log/messages.h"
#include "framework/mpse.h"

#include "E3Saggro_utils.h"

using namespace snort;

//-------------------------------------------------------------------------
// "E3Saggro"
//-------------------------------------------------------------------------

class E3SaggroMpse : public Mpse
{
private:
    struct PatternInfo {
        ///hash chain info. initialized in ExpoSizeStrSrc::preprocessQueriedString. used in ExpoSizeStrSrc::queryString.
        int chainLength;
        std::array<int, ExpoSizeStrSrc::max_ml2> exponents;
        std::array<uint64_t, ExpoSizeStrSrc::max_ml2> t_hashes;

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

    std::vector<PatternInfo> patterns;
    ExpoSizeStrSrc *E3Saggro;

    std::vector<uint8_t> tmp_buffer;
    uint8_t tolower_lookup[256];

public:
    E3SaggroMpse(const MpseAgent* agent) : Mpse("E3Saggro") {
        this->agent = agent;

        std::iota(tolower_lookup, tolower_lookup + 256, 0);
        for (int i = 'A'; i <= (int)'Z'; i++) tolower_lookup[i] ^= 32;

        E3Saggro = new ExpoSizeStrSrc;
    }

    ~E3SaggroMpse() override {
        delete E3Saggro;
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

        E3Saggro->preprocessQueriedString(tmp_buffer, m, patterns.back().chainLength, patterns.back().exponents, patterns.back().t_hashes);

        return 0;
    }

    /**
     * probably called after all add_pattern().
     * @param sc found in ??? (snort::SnortConfig). Doesn't seem modified anywhere by lowmem.
     * @return 0 for succes.
     */
    int prep_patterns(SnortConfig* sc) override {
        for (auto &p: patterns) {
            if (p.user) {
                if (!p.negated) agent->build_tree(sc, p.user, &p.tree);
                else agent->negate_list(p.user, &p.neg_list);
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

        E3Saggro->updateText(tmp_buffer, n);

        int matches = 0;
        for (auto &p: patterns) {
            if (E3Saggro->queryString(p.chainLength, p.exponents, p.t_hashes)) { ///TODO query si pe s[1..)?
                matches++;
                if (match(p.user, p.tree, 0, context, p.neg_list) > 0) {
                    return matches;
                }
            }
        }

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

static Mpse* e3saggro_ctor(const SnortConfig*, class Module*, const MpseAgent* agent)
{
    return new E3SaggroMpse(agent);
}

static void e3saggro_dtor(Mpse* p)
{
    delete p;
}

static void e3saggro_init()
{

}

static void e3saggro_print()
{
    LogMessage("E3Saggro.");
}

static const MpseApi e3saggro_api =
{
    {
        PT_SEARCH_ENGINE,
        sizeof(MpseApi),
        SEAPI_VERSION,
        0,
        API_RESERVED,
        API_OPTIONS,
        "E3Saggro",
        "E3Saggro MPSE",
        nullptr,
        nullptr
    },
    MPSE_BASE,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    e3saggro_ctor,
    e3saggro_dtor,
    e3saggro_init,
    e3saggro_print,
    nullptr
};

const BaseApi* se_E3Saggro = &e3saggro_api.base;

