// Minimal OpenGL 3.3 function loader for Windows (no GLEW / GLAD dependency).
// Only the entry points the viewer uses are declared and loaded.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <GL/gl.h>
#include <cstddef>

typedef char GLchar;
typedef std::ptrdiff_t GLsizeiptr;
typedef std::ptrdiff_t GLintptr;

#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_MULTISAMPLE 0x809D

#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002

#define PBD_GL_FUNCTIONS(X)                                                                                       \
    X(GLuint, glCreateShader, (GLenum type))                                                                      \
    X(void, glShaderSource, (GLuint shader, GLsizei count, const GLchar* const* source, const GLint* length))   \
    X(void, glCompileShader, (GLuint shader))                                                                     \
    X(void, glGetShaderiv, (GLuint shader, GLenum pname, GLint* params))                                         \
    X(void, glGetShaderInfoLog, (GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog))             \
    X(void, glDeleteShader, (GLuint shader))                                                                      \
    X(GLuint, glCreateProgram, (void))                                                                            \
    X(void, glAttachShader, (GLuint program, GLuint shader))                                                     \
    X(void, glLinkProgram, (GLuint program))                                                                      \
    X(void, glGetProgramiv, (GLuint program, GLenum pname, GLint* params))                                       \
    X(void, glGetProgramInfoLog, (GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog))           \
    X(void, glUseProgram, (GLuint program))                                                                       \
    X(void, glDeleteProgram, (GLuint program))                                                                    \
    X(GLint, glGetUniformLocation, (GLuint program, const GLchar* name))                                         \
    X(void, glUniformMatrix4fv, (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value))     \
    X(void, glUniform3f, (GLint location, GLfloat v0, GLfloat v1, GLfloat v2))                                   \
    X(void, glUniform1f, (GLint location, GLfloat v0))                                                            \
    X(void, glUniform1i, (GLint location, GLint v0))                                                              \
    X(void, glGenVertexArrays, (GLsizei n, GLuint * arrays))                                                     \
    X(void, glBindVertexArray, (GLuint array))                                                                    \
    X(void, glDeleteVertexArrays, (GLsizei n, const GLuint* arrays))                                             \
    X(void, glGenBuffers, (GLsizei n, GLuint * buffers))                                                         \
    X(void, glBindBuffer, (GLenum target, GLuint buffer))                                                        \
    X(void, glBufferData, (GLenum target, GLsizeiptr size, const void* data, GLenum usage))                     \
    X(void, glBufferSubData, (GLenum target, GLintptr offset, GLsizeiptr size, const void* data))              \
    X(void, glDeleteBuffers, (GLsizei n, const GLuint* buffers))                                                 \
    X(void, glVertexAttribPointer,                                                                                \
      (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer))        \
    X(void, glEnableVertexAttribArray, (GLuint index))

#define PBD_GL_DECLARE(ret, name, args)                                                                           \
    typedef ret(APIENTRY* PFN_##name) args;                                                                       \
    extern PFN_##name name;
PBD_GL_FUNCTIONS(PBD_GL_DECLARE)
#undef PBD_GL_DECLARE

namespace gl
{
// Loads every function above from the current context. Returns the name of the
// first missing entry point, or nullptr on success.
const char* LoadFunctions();
} // namespace gl
