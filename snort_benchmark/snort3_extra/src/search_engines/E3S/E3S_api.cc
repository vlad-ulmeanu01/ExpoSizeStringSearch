#include "framework/base_api.h"
#include "main/snort_types.h"

extern const snort::BaseApi* se_E3S;

SO_PUBLIC const snort::BaseApi* snort_plugins[] =
{
    se_E3S,
    nullptr
};

