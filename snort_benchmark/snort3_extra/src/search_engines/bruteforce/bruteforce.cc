/*
*   An abstracted interface to the Multi-Pattern Matching routines,
*   that's why we're passing 'void *' objects around.
*
*/

#include "log/messages.h"
#include "framework/mpse.h"

#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <cstring>
#include <cstdio>

using namespace snort;

//-------------------------------------------------------------------------
// "bruteforce"
//-------------------------------------------------------------------------

class BruteforceMpse : public Mpse
{
private:
    struct PatternInfo {
        std::string pat;
        void *user; ///?? de la cine am pattern-ul?
        void *tree; ///detection_option_tree_root_t. am nevoie de cate un obiect tree pentru fiecare match (ca sa stie snort ce pattern-uri sunt gasite).

        ///TODO mare grija cu pointerii. PatternInfo e folosit in std::vector<>...
        PatternInfo(std::string pat, void *user, void *tree) {
            this->pat = pat;
            this->user = user;
            this->tree = tree;
        }
    };

    const MpseAgent *agent;

    std::vector<PatternInfo> patterns; ///tine minte si un pointer catre user (???).
    //FILE *dbg;

    void debug(std::string s) const {
        //fprintf(dbg, "%s\n", s.c_str());
        printf("%s\n", s.c_str());
    }

public:
    BruteforceMpse(const MpseAgent* agent) : Mpse("bruteforce") {
        //dbg = fopen("/home/vlad/Documents/SublimeMerge/snort3_demo/tests/search_engines/bruteforce/debug.txt", "w");

        this->agent = agent;
    }

    ~BruteforceMpse() override {
        //puts("03/14-06:42:08.719162, 8, TCP, stream_tcp, 238, S2C, 10.9.8.7:80, 10.1.2.3:48620, 1:1:0, allow");
        //fclose(dbg);
    }

    /**
     * @param P the new pattern.
     * @param m length of the pattern.
     * @param desc found in mpse.h. info about no_case, negated, literal, multi_match, flags.
     * @param user ???
     * @return 0 for succes.
     */
    int add_pattern(const uint8_t* P, unsigned m, const PatternDescriptor& desc, void* user) override {
        patterns.emplace_back(std::string((char *) P), user, nullptr);

        //debug("add_pattern: " + patterns.back().pat);
        //debug("no_case: " + std::to_string(desc.no_case));
        //debug("negated: " + std::to_string(desc.negated));
        //debug("literal: " + std::to_string(desc.literal));
        //debug("multi_match: " + std::to_string(desc.multi_match));

        return 0;
    }

    /**
     * probably called after all add_pattern().
     * @param sc found in ??? (snort::SnortConfig). Doesn't seem modified anywhere by lowmem.
     * @return 0 for succes.
     */
    int prep_patterns(SnortConfig* sc) override {
        debug("prep_patterns called. patterns size = " + std::to_string(patterns.size()));

        for (auto &p: patterns) {
            agent->build_tree(sc, p.user, &p.tree);

            /*
            if (p->user) {
                if (p->negative) ts->agent->negate_list(p->user, &root->pkeyword->neg_list);
                else ts->agent->build_tree(sc, p->user, &root->pkeyword->rule_option_tree);
            }
            */
        }

        return 0;
    }

    /**
     *
     * @param T text in which we have to search all patterns.
     * @param n length of T.
     * @param match TODO cred ca trebuie apelata asta.
     * @param context
     * @param current_state
     * @return count of found patterns. (?) if the first pattern was found 2x, and the second one 1x, return 3.
     */
    int _search(const uint8_t* T, int n, MpseMatch match, void* context, int* current_state) override {
        //debug("search called: T = " + std::string((char *)T) + "END");
        int matches = 0;
        for (auto &p: patterns) {
            for (int i = 0; i + p.pat.size() - 1 < n; i++) {
                int j = 0;
                while (j < (int)p.pat.size() && T[i + j] == p.pat[j]) j++;

                if (j >= (int)p.pat.size()) {
                    matches++;
                    if (match(p.user, p.tree, i, context, nullptr) > 0) { ///(???) cred ca trebuie sa marchez pozitia match-urilor. dc ret > 0, pot sa termin inainte.
                        debug("search finished by match return.");
                        return matches;
                    }
                }
            }
        }


        debug("search finished: matches = " + std::to_string(matches));

        return matches;
    }

    /**
     * how many times was add_pattern called?
     * @return how many patterns do I have.
     */
    int get_pattern_count() const override {
        debug("get_pattern_count called: " + std::to_string(patterns.size()));

        return (int)patterns.size();
    }

    ///mai jos: functii virtuale mostenite, dar care nu apar in scheletul de la lowmem. poate e apelata una dintre ele.

//    void _search(MpseBatch&, MpseType) override {
//        debug("_search batch called.");
//    }

//    MpseRespType receive_responses(MpseBatch&, MpseType) override {
//        debug("receive_responses called.");
//        return MPSE_RESP_COMPLETE_SUCCESS;
//    }
};

//-------------------------------------------------------------------------
// api
//-------------------------------------------------------------------------

static Mpse* bf_ctor(const SnortConfig*, class Module*, const MpseAgent* agent)
{
    return new BruteforceMpse(agent);
}

static void bf_dtor(Mpse* p)
{
    delete p;
}

static void bf_init()
{

}

static void bf_print()
{
    LogMessage("bruteforce idk.");
}

static const MpseApi bf_api =
{
    {
        PT_SEARCH_ENGINE,
        sizeof(MpseApi),
        SEAPI_VERSION,
        0,
        API_RESERVED,
        API_OPTIONS,
        "bruteforce",
        "Bruteforce MPSE",
        nullptr,
        nullptr
    },
    MPSE_BASE,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    bf_ctor,
    bf_dtor,
    bf_init,
    bf_print,
    nullptr
};

const BaseApi* se_bruteforce = &bf_api.base;

