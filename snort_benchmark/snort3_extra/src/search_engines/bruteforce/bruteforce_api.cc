#include "framework/base_api.h"
#include "main/snort_types.h"

extern const snort::BaseApi* se_bruteforce;

SO_PUBLIC const snort::BaseApi* snort_plugins[] =
{
    se_bruteforce,
    nullptr
};

