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
#include <cctype>

using namespace snort;

//-------------------------------------------------------------------------
// "bruteforce"
//-------------------------------------------------------------------------

class BruteforceMpse : public Mpse
{
private:
    struct PatternInfo {
        std::vector<uint8_t> pat;
        void *user; ///?? de la cine am pattern-ul?
        void *tree; ///detection_option_tree_root_t. am nevoie de cate un obiect tree pentru fiecare match (ca sa stie snort ce pattern-uri sunt gasite).
        void *neg_list; ///varul lui tree pentru negated.
        bool no_case; ///true <=> trebuie sa caut case insensitive.
        bool negated; ///true <=> vreau sa NU gasesc pattern-ul.

        ///TODO mare grija cu pointerii. PatternInfo e folosit in std::vector<>...
        PatternInfo(std::vector<uint8_t> pat, void *user, void *tree, void *neg_list, bool no_case, bool negated) {
            this->pat = pat;

            no_case = true; ///?? pare intotdeauna adevarat (?)

            if (no_case) { ///cu ! crapa mai tarziu...
                for (int i = 0; i < (int)pat.size(); i++) this->pat[i] = tolower(this->pat[i]);
            }

            this->user = user;
            this->tree = tree;
            this->neg_list = neg_list;
            this->no_case = no_case;
            this->negated = negated;
        }

        bool operator == (const PatternInfo &oth) {
            return no_case == oth.no_case && pat == oth.pat && negated == oth.negated;
        }
    };

    const MpseAgent *agent;

    std::vector<PatternInfo> patterns;

public:
    BruteforceMpse(const MpseAgent* agent) : Mpse("bruteforce") {
        this->agent = agent;
    }

    ~BruteforceMpse() override {
    }

    /**
     * @param P the new pattern.
     * @param m length of the pattern.
     * @param desc found in mpse.h. info about no_case, negated, literal, multi_match, flags.
     * @param user ???
     * @return 0 for succes.
     */
    int add_pattern(const uint8_t* P, unsigned m, const PatternDescriptor& desc, void* user) override {
        patterns.emplace_back(std::vector<uint8_t>(P, P+m), user, nullptr, nullptr, desc.no_case, desc.negated); ///std::string((char *) P, m)

        for (int i = 0; i < (int)patterns.size() - 1; i++) {
            if (patterns[i] == patterns.back()) {
                patterns.pop_back();
                break;
            }
        }

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
     * @param match TODO cred ca trebuie apelata asta.
     * @param context
     * @param current_state
     * @return count of found patterns. (?) if the first pattern was found 2x, and the second one 1x, return 3.
     */
    int _search(const uint8_t* T, int n, MpseMatch match, void* context, int* current_state) override {
        int matches = 0;

        std::vector<uint8_t> T_nocase(T, T+n);
        for (int i = 0; i < n; i++) {
            T_nocase[i] = tolower(T_nocase[i]);
        }

        ///DA. conteaza ordinea forurilor. se pare ca prefera intai match-uri care se termina mai devreme in T.
        for (int i = 0; i < n; i++) {
            for (auto &p: patterns) {
                if (i + (int)p.pat.size() - 1 < n) {
                    int j = 0;

                    ///(bug snort?) se pare ca pune toate pattern-urile cu p.no_case == true.
                    if (p.no_case) {
                        while (j < (int)p.pat.size() && T_nocase[i + j] == p.pat[j]) j++;
                    } else {
                        while (j < (int)p.pat.size() && T[i + j] == p.pat[j]) j++;
                    }

                    if (j >= (int)p.pat.size()) { ///cred ca negated conteaza doar la triere in tree / neg_list?
                        matches++;

                        if (match(p.user, p.tree, i + j, context, p.neg_list) > 0) {
                            return matches;
                        }
                    }
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


