#include "cloth_renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace viewer
{
namespace
{
const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aDisp;
layout(location = 3) in float aActive;
uniform mat4 uMvp;
uniform float uDispScale;
out vec3 vPos;
out vec3 vNormal;
out float vDisp;
out float vActive;
void main()
{
    vPos = aPos;
    vNormal = aNormal;
    vDisp = clamp(aDisp * uDispScale, 0.0, 1.0);
    vActive = aActive;
    gl_Position = uMvp * vec4(aPos, 1.0);
}
)";

const char* kFragmentShader = R"(#version 330 core
in vec3 vPos;
in vec3 vNormal;
in float vDisp;
in float vActive;
uniform vec3 uEye;
uniform vec3 uLight;
uniform vec3 uColor;
uniform int uMode;        // 0 = shaded cloth, 1 = flat colour
uniform int uShowActive;
out vec4 fragColor;

vec3 palette(float t)     // grey -> teal -> violet -> orange, the portfolio colours
{
    vec3 a = vec3(0.85, 0.87, 0.89), b = vec3(0.12, 0.58, 0.66);
    vec3 c = vec3(0.43, 0.38, 0.72), d = vec3(0.95, 0.33, 0.11);
    if (t < 0.33) return mix(a, b, t / 0.33);
    if (t < 0.66) return mix(b, c, (t - 0.33) / 0.33);
    return mix(c, d, (t - 0.66) / 0.34);
}

void main()
{
    if (uMode == 1) { fragColor = vec4(uColor, 1.0); return; }
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;                    // two-sided lighting
    vec3 v = normalize(uEye - vPos);
    vec3 l = normalize(uLight);
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(n, normalize(l + v)), 0.0), 48.0) * 0.22;
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.22;
    vec3 base = palette(vDisp);
    if (uShowActive == 1 && vActive > 0.5) base = mix(base, vec3(1.0, 0.70, 0.52), 0.25);
    fragColor = vec4(base * (0.34 + 0.66 * diff) + vec3(spec) + rim * vec3(0.85, 0.92, 1.0), 1.0);
}
)";

GLuint Compile(GLenum type, const char* src, const char** error)
{
    static char log[2048];
    const GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        *error = log;
        glDeleteShader(s);
        return 0;
    }
    return s;
}

void SetupClothAttributes()
{
    const GLsizei stride = sizeof(float) * 8;
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 3));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 6));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 7));
}
} // namespace

ClothRenderer::~ClothRenderer()
{
    Release();
    if (program)
    {
        glDeleteProgram(program);
    }
}

bool ClothRenderer::Init(const char** error)
{
    const GLuint vs = Compile(GL_VERTEX_SHADER, kVertexShader, error);
    if (!vs)
    {
        return false;
    }
    const GLuint fs = Compile(GL_FRAGMENT_SHADER, kFragmentShader, error);
    if (!fs)
    {
        glDeleteShader(vs);
        return false;
    }
    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        static char log[2048];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        *error = log;
        return false;
    }
    uMvp = glGetUniformLocation(program, "uMvp");
    uEye = glGetUniformLocation(program, "uEye");
    uLight = glGetUniformLocation(program, "uLight");
    uMode = glGetUniformLocation(program, "uMode");
    uColor = glGetUniformLocation(program, "uColor");
    uDispScale = glGetUniformLocation(program, "uDispScale");
    uShowActive = glGetUniformLocation(program, "uShowActive");

    glGenVertexArrays(1, &grabVao);
    glGenBuffers(1, &grabVbo);
    glBindVertexArray(grabVao);
    glBindBuffer(GL_ARRAY_BUFFER, grabVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * 2, nullptr, GL_DYNAMIC_DRAW);
    SetupClothAttributes();
    glBindVertexArray(0);
    return true;
}

void ClothRenderer::Release()
{
    const GLuint buffers[] = {clothVbo, triEbo, wireEbo, blockEbo, floorVbo};
    for (GLuint b : buffers)
    {
        if (b)
        {
            glDeleteBuffers(1, &b);
        }
    }
    if (clothVao)
    {
        glDeleteVertexArrays(1, &clothVao);
    }
    if (floorVao)
    {
        glDeleteVertexArrays(1, &floorVao);
    }
    clothVao = clothVbo = triEbo = wireEbo = blockEbo = floorVao = floorVbo = 0;
}

void ClothRenderer::Rebuild(const pbd::Cloth& rest)
{
    Release();
    nx = rest.nx;
    ny = rest.ny;
    restPosition = rest.position;
    vertices.assign(rest.position.size(), Vertex{});

    std::vector<unsigned> tris, wire;
    tris.reserve(static_cast<size_t>(nx - 1) * (ny - 1) * 6);
    for (int j = 0; j + 1 < ny; ++j)
    {
        for (int i = 0; i + 1 < nx; ++i)
        {
            const unsigned a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
            tris.insert(tris.end(), {a, c, b, b, c, d});
        }
    }
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            const unsigned a = j * nx + i;
            if (i + 1 < nx)
            {
                wire.insert(wire.end(), {a, a + 1});
            }
            if (j + 1 < ny)
            {
                wire.insert(wire.end(), {a, a + static_cast<unsigned>(nx)});
            }
        }
    }
    triCount = static_cast<GLsizei>(tris.size());
    wireCount = static_cast<GLsizei>(wire.size());

    glGenVertexArrays(1, &clothVao);
    glBindVertexArray(clothVao);
    glGenBuffers(1, &clothVbo);
    glBindBuffer(GL_ARRAY_BUFFER, clothVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), nullptr, GL_DYNAMIC_DRAW);
    SetupClothAttributes();
    glGenBuffers(1, &triEbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * tris.size(), tris.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &wireEbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wireEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * wire.size(), wire.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &blockEbo);
    glBindVertexArray(0);

    float minY = 0.0f, maxExtent = 0.0f;
    for (const auto& p : rest.position)
    {
        minY = std::min(minY, p.y);
        maxExtent = std::max(maxExtent, std::max(std::fabs(p.x), std::fabs(p.z)));
    }
    BuildFloor(minY - 0.6f, std::max(1.5f, maxExtent * 2.0f));
}

void ClothRenderer::BuildFloor(float y, float halfSize)
{
    floorY = y;
    std::vector<Vertex> lines;
    const int n = 16;
    for (int k = -n; k <= n; ++k)
    {
        const float t = halfSize * static_cast<float>(k) / n;
        lines.push_back({-halfSize, y, t, 0, 1, 0, 0, 0});
        lines.push_back({halfSize, y, t, 0, 1, 0, 0, 0});
        lines.push_back({t, y, -halfSize, 0, 1, 0, 0, 0});
        lines.push_back({t, y, halfSize, 0, 1, 0, 0, 0});
    }
    floorCount = static_cast<GLsizei>(lines.size());
    glGenVertexArrays(1, &floorVao);
    glBindVertexArray(floorVao);
    glGenBuffers(1, &floorVbo);
    glBindBuffer(GL_ARRAY_BUFFER, floorVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * lines.size(), lines.data(), GL_STATIC_DRAW);
    SetupClothAttributes();
    glBindVertexArray(0);
}

void ClothRenderer::Update(const pbd::Cloth& cloth, const std::vector<unsigned char>& active, int blockSize,
                           Vec3 offset)
{
    if (cloth.nx != nx || cloth.ny != ny)
    {
        return;
    }
    const int n = nx * ny;
    for (int k = 0; k < n; ++k)
    {
        Vertex& v = vertices[k];
        const pbd::Vec3 p = cloth.position[k] + offset;
        v.px = p.x;
        v.py = p.y;
        v.pz = p.z;
        v.nx = v.ny = v.nz = 0.0f;
        v.displacement = pbd::length(cloth.position[k] - restPosition[k]);
        v.active = 0.0f;
    }

    // Area-weighted vertex normals from the two triangles of every quad.
    auto pos = [&](int k) { return cloth.position[k]; };
    auto accumulate = [&](int a, int b, int c) {
        const Vec3 fn = Cross(pos(b) - pos(a), pos(c) - pos(a));
        for (int k : {a, b, c})
        {
            vertices[k].nx += fn.x;
            vertices[k].ny += fn.y;
            vertices[k].nz += fn.z;
        }
    };
    for (int j = 0; j + 1 < ny; ++j)
    {
        for (int i = 0; i + 1 < nx; ++i)
        {
            const int a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
            accumulate(a, c, b);
            accumulate(b, c, d);
        }
    }

    // Active blocks: tint their particles and outline each block along the cloth.
    blockLines.clear();
    if (blockSize > 0 && !active.empty())
    {
        const int bx = (nx + blockSize - 1) / blockSize;
        for (size_t b = 0; b < active.size(); ++b)
        {
            if (!active[b])
            {
                continue;
            }
            const int i0 = static_cast<int>(b % bx) * blockSize, j0 = static_cast<int>(b / bx) * blockSize;
            const int i1 = std::min(i0 + blockSize, nx - 1), j1 = std::min(j0 + blockSize, ny - 1);
            for (int j = j0; j < std::min(j0 + blockSize, ny); ++j)
            {
                for (int i = i0; i < std::min(i0 + blockSize, nx); ++i)
                {
                    vertices[j * nx + i].active = 1.0f;
                }
            }
            for (int i = i0; i < i1; ++i)
            {
                blockLines.insert(blockLines.end(), {static_cast<unsigned>(j0 * nx + i), static_cast<unsigned>(j0 * nx + i + 1)});
                blockLines.insert(blockLines.end(), {static_cast<unsigned>(j1 * nx + i), static_cast<unsigned>(j1 * nx + i + 1)});
            }
            for (int j = j0; j < j1; ++j)
            {
                blockLines.insert(blockLines.end(), {static_cast<unsigned>(j * nx + i0), static_cast<unsigned>((j + 1) * nx + i0)});
                blockLines.insert(blockLines.end(), {static_cast<unsigned>(j * nx + i1), static_cast<unsigned>((j + 1) * nx + i1)});
            }
        }
    }
    blockCount = static_cast<GLsizei>(blockLines.size());

    glBindVertexArray(clothVao);
    glBindBuffer(GL_ARRAY_BUFFER, clothVbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertex) * vertices.size(), vertices.data());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, blockEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * (blockLines.empty() ? 1 : blockLines.size()),
                 blockLines.empty() ? nullptr : blockLines.data(), GL_DYNAMIC_DRAW);
    glBindVertexArray(0);
}

void ClothRenderer::SetGrab(bool on, Vec3 particle, Vec3 target)
{
    grabOn = on;
    grabParticle = particle;
    grabTarget = target;
}

void ClothRenderer::Draw(const OrbitCamera& camera, int width, int height, const DrawOptions& o)
{
    if (!program || !clothVao || width <= 0 || height <= 0)
    {
        return;
    }
    const Mat4 mvp = camera.Projection(static_cast<float>(width) / height) * camera.View();
    const Vec3 eye = camera.Eye();

    glUseProgram(program);
    glUniformMatrix4fv(uMvp, 1, GL_FALSE, mvp.m);
    glUniform3f(uEye, eye.x, eye.y, eye.z);
    glUniform3f(uLight, 0.35f, 0.9f, 0.55f);
    glUniform1f(uDispScale, o.displacementScale);
    glUniform1i(uShowActive, o.showBlocks ? 1 : 0);

    // floor
    glUniform1i(uMode, 1);
    if (o.drawFloor)
    {
        glUniform3f(uColor, 0.80f, 0.82f, 0.85f);
        glBindVertexArray(floorVao);
        glDrawArrays(GL_LINES, 0, floorCount);
    }

    // cloth surface (pushed back slightly so lines drawn on it stay visible)
    glUniform1i(uMode, 0);
    glBindVertexArray(clothVao);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triEbo);
    glDrawElements(GL_TRIANGLES, triCount, GL_UNSIGNED_INT, nullptr);
    glDisable(GL_POLYGON_OFFSET_FILL);

    glUniform1i(uMode, 1);
    if (o.wireframe)
    {
        glUniform3f(uColor, 0.30f, 0.34f, 0.38f);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wireEbo);
        glDrawElements(GL_LINES, wireCount, GL_UNSIGNED_INT, nullptr);
    }
    if (o.showBlocks && blockCount > 0)
    {
        glUniform3f(uColor, 0.95f, 0.33f, 0.11f);
        glLineWidth(1.6f);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, blockEbo);
        glDrawElements(GL_LINES, blockCount, GL_UNSIGNED_INT, nullptr);
        glLineWidth(1.0f);
    }

    // grab tether: particle -> mouse target
    if (grabOn)
    {
        const Vertex pts[2] = {{grabParticle.x, grabParticle.y, grabParticle.z, 0, 1, 0, 0, 0},
                               {grabTarget.x, grabTarget.y, grabTarget.z, 0, 1, 0, 0, 0}};
        glBindVertexArray(grabVao);
        glBindBuffer(GL_ARRAY_BUFFER, grabVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(pts), pts);
        glUniform3f(uColor, 0.95f, 0.33f, 0.11f);
        glDisable(GL_DEPTH_TEST);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, 2);
        glPointSize(9.0f);
        glDrawArrays(GL_POINTS, 1, 1);
        glLineWidth(1.0f);
        glEnable(GL_DEPTH_TEST);
    }
    glBindVertexArray(0);
    glUseProgram(0);
}
} // namespace viewer
