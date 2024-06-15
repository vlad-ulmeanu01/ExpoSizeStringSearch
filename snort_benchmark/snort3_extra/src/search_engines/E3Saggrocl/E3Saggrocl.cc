/*
*   An abstracted interface to the Multi-Pattern Matching routines,
*   that's why we're passing 'void *' objects around.
*
*/

#include "log/messages.h"
#include "framework/mpse.h"
// #include "framework/base_api.h"
// #include "main/snort_types.h"

#include "E3Saggrocl_utils.h"

using namespace snort;

//-------------------------------------------------------------------------
// "E3Saggrocl"
//-------------------------------------------------------------------------

class E3SaggroclMpse : public Mpse
{
private:
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

    std::vector<PatternInfo> patterns;
    ExpoSizeStrSrc *E3Saggrocl;
    std::unordered_set<uint64_t> patternHashes; ///the hashes of the patterns. don't want to work with duplicates.
    std::vector<LinkInfo> connections; ///the patterns' hash chain connections I am interested in finding out whether they exist or not in the DAG.

    std::vector<uint8_t> tmp_buffer;
    uint8_t tolower_lookup[256];

public:
    E3SaggroclMpse(const MpseAgent* agent) : Mpse("E3Saggrocl") {
        this->agent = agent;

        std::iota(tolower_lookup, tolower_lookup + 256, 0);
        for (int i = 'A'; i <= (int)'Z'; i++) tolower_lookup[i] ^= 32;

        E3Saggrocl = new ExpoSizeStrSrc;
    }

    ~E3SaggroclMpse() override {
        delete E3Saggrocl;
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

        E3Saggrocl->preprocessQueriedString(tmp_buffer, m, patterns.back().ci);

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

        //printf("E3Saggrocl matches = %d\n", matches);

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

static Mpse* e3saggrocl_ctor(const SnortConfig*, class Module*, const MpseAgent* agent)
{
    return new E3SaggroclMpse(agent);
}

static void e3saggrocl_dtor(Mpse* p)
{
    delete p;
}

static void e3saggrocl_init()
{

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

// SO_PUBLIC const snort::BaseApi* snort_plugins[] =
// {
//     &e3saggrocl_api.base,
//     nullptr
// };
