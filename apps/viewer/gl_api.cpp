#include "gl_api.h"

#define PBD_GL_DEFINE(ret, name, args) PFN_##name name = nullptr;
PBD_GL_FUNCTIONS(PBD_GL_DEFINE)
#undef PBD_GL_DEFINE

namespace gl
{
static PROC Load(const char* name)
{
    PROC p = wglGetProcAddress(name);
    // Some drivers return small sentinel values instead of nullptr for missing functions.
    const INT_PTR v = reinterpret_cast<INT_PTR>(p);
    if (v == 0 || v == 1 || v == 2 || v == 3 || v == -1)
    {
        static HMODULE module = LoadLibraryA("opengl32.dll");
        p = GetProcAddress(module, name);
    }
    return p;
}

const char* LoadFunctions()
{
#define PBD_GL_LOAD(ret, name, args)                                                                              \
    name = reinterpret_cast<PFN_##name>(reinterpret_cast<void*>(Load(#name)));                                                             \
    if (!name)                                                                                                    \
    {                                                                                                             \
        return #name;                                                                                             \
    }
    PBD_GL_FUNCTIONS(PBD_GL_LOAD)
#undef PBD_GL_LOAD
    return nullptr;
}
} // namespace gl
