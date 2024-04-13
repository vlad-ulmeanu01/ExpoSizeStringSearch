/*
*   An abstracted interface to the Multi-Pattern Matching routines,
*   that's why we're passing 'void *' objects around.
*
*/

#include "log/messages.h"
#include "framework/mpse.h"

#include "E3S_utils.h"

using namespace snort;

//-------------------------------------------------------------------------
// "E3S"
//-------------------------------------------------------------------------

class E3SMpse : public Mpse
{
private:
    struct PatternInfo {
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
    ExpoSizeStrSrc *E3S;

    std::vector<uint8_t> tmp_buffer;
    uint8_t tolower_lookup[256];

public:
    E3SMpse(const MpseAgent* agent) : Mpse("E3S") {
        this->agent = agent;

        std::iota(tolower_lookup, tolower_lookup + 256, 0);
        for (int i = 'A'; i <= (int)'Z'; i++) tolower_lookup[i] ^= 32;

        E3S = new ExpoSizeStrSrc;
    }

    ~E3SMpse() override {
        E3S->trieBuffersFree();
        delete E3S;
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

        E3S->insertQueriedString(tmp_buffer, m);

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

        E3S->linearizeMaps(E3S->trieRoot);

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

        E3S->updateText(tmp_buffer, n);
        E3S->massSearch(E3S->trieRoot);

        for(int i = 0; i < (int)patterns.size(); i++) {
            if (E3S->massSearchResults[i] && match(patterns[i].user, patterns[i].tree, 0, context, patterns[i].neg_list)) {
                return E3S->massSearchCntMatches;
            }
        }

        //printf("E3S matches = %d\n", E3S->massSearchCntMatches);

        return E3S->massSearchCntMatches;
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

static Mpse* e3s_ctor(const SnortConfig*, class Module*, const MpseAgent* agent)
{
    return new E3SMpse(agent);
}

static void e3s_dtor(Mpse* p)
{
    delete p;
}

static void e3s_init()
{

}

static void e3s_print()
{
    LogMessage("E3S.");
}

static const MpseApi e3s_api =
{
    {
        PT_SEARCH_ENGINE,
        sizeof(MpseApi),
        SEAPI_VERSION,
        0,
        API_RESERVED,
        API_OPTIONS,
        "E3S",
        "E3S MPSE",
        nullptr,
        nullptr
    },
    MPSE_BASE,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    e3s_ctor,
    e3s_dtor,
    e3s_init,
    e3s_print,
    nullptr
};

const BaseApi* se_E3S = &e3s_api.base;

