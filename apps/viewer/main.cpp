// pbd_viewer: real-time OpenGL viewer for the CPU / GPU / active-block PBD solvers.
// Win32 + WGL + OpenGL 3.3 (compatibility profile for the text overlay); no external libraries.
//
//   [LMB drag] pull the cloth     [RMB drag] orbit     [wheel] zoom
//   [1] cpu  [2] cpu-active  [3] gpu  [4] gpu-active
//   [P] scene (drum / curtain)  [N] resolution  [Space] pause  [R] reset
//   [C] compare: global and active solvers side by side on the same input
//   [W] wireframe  [B] active blocks  [Esc] quit
#include "gl_api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "camera.h"
#include "cloth_renderer.h"
#include "pbd/solver.h"

namespace
{
using pbd::Vec3;
using viewer::OrbitCamera;
using viewer::Ray;

typedef HGLRC(WINAPI* PFN_wglCreateContextAttribsARB)(HDC, HGLRC, const int*);
typedef BOOL(WINAPI* PFN_wglSwapIntervalEXT)(int);

const int kResolutions[] = {48, 64, 96, 128, 160};

struct Scene
{
    const char* name;
    pbd::PinMode pin;
    float gravity;
};
const Scene kScenes[] = {{"drum (framed, no gravity)", pbd::PinMode::Frame, 0.0f},
                         {"curtain (top edge, gravity)", pbd::PinMode::TopEdge, 9.81f}};

struct App
{
    HWND hwnd = nullptr;
    HDC hdc = nullptr;
    HGLRC glrc = nullptr;
    int width = 1280, height = 800;
    GLuint fontBase = 0;
    bool running = true;

    // simulation
    pbd::SolverConfig config;
    pbd::Cloth rest, cloth;
    pbd::Solver solver;
    pbd::SolverType type = pbd::CPU_GLOBAL;
    pbd::StepStats stats;
    int sceneIndex = 0, resolutionIndex = 2;
    bool paused = false;
    bool gpuAvailable = false;
    std::string message;
    double messageUntil = 0.0;

    // compare mode: `solver` runs the global variant, `solverB` the active one
    bool compare = false;
    pbd::Solver solverB;
    pbd::Cloth clothB;
    pbd::StepStats statsB;
    float maxDifference = 0.0f; // max |x_global - x_active| over all particles (m)
    int frameCounter = 0;

    // interaction
    OrbitCamera camera;
    viewer::DrawOptions draw;
    std::unique_ptr<viewer::ClothRenderer> renderer, rendererB;
    int grabIndex = -1;
    int grabSide = 0; // 0 = left / single cloth, 1 = right cloth in compare mode
    Vec3 grabTarget{0, 0, 0}, grabPlanePoint{0, 0, 0}, grabPlaneNormal{0, 0, 1};
    bool orbiting = false;
    int lastX = 0, lastY = 0;

    // smoothed numbers for the overlay
    struct Average
    {
        double step = 0, solve = 0, iterations = 0, constraints = 0, active = 0;
        bool primed = false;
        void Reset()
        {
            primed = false;
        }
        void Add(const pbd::StepStats& s)
        {
            const double activePct = s.totalBlocks ? 100.0 * s.activeBlocks / s.totalBlocks : 100.0;
            const double a = primed ? 0.06 : 1.0;
            step += a * (s.stepMs - step);
            solve += a * (s.solveMs - solve);
            iterations += a * (s.iterations - iterations);
            constraints += a * (s.solvedConstraints - constraints);
            active += a * (activePct - active);
            primed = true;
        }
    } avgA, avgB;

    // timing
    LARGE_INTEGER frequency{}, start{};
    double fps = 0.0, fpsTimer = 0.0;
    int fpsFrames = 0;
};

App* gApp = nullptr;

double Now(const App& app)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return static_cast<double>(t.QuadPart - app.start.QuadPart) / static_cast<double>(app.frequency.QuadPart);
}

void Flash(App& app, const std::string& text)
{
    app.message = text;
    app.messageUntil = Now(app) + 2.5;
}

const char* TypeLabel(pbd::SolverType t)
{
    switch (t)
    {
    case pbd::CPU_GLOBAL:
        return "cpu";
    case pbd::CPU_ACTIVE:
        return "cpu-active";
    case pbd::GPU_GLOBAL:
        return "gpu";
    default:
        return "gpu-active";
    }
}

const float kCompareGap = 0.9f; // half distance between the two cloths in compare mode

Vec3 Offset(const App& app, int side)
{
    if (!app.compare)
    {
        return Vec3{0, 0, 0};
    }
    return Vec3{side == 0 ? -kCompareGap : kCompareGap, 0.0f, 0.0f};
}

bool IsGpu(pbd::SolverType t)
{
    return t == pbd::GPU_GLOBAL || t == pbd::GPU_ACTIVE;
}

// Compare mode: global (left) and active (right) of the same backend start from one state.
bool StartCompare(App& app, bool gpu, const pbd::Cloth& state)
{
    if (gpu && !app.gpuAvailable)
    {
        Flash(app, "compare on GPU needs a CUDA device; using the CPU pair");
        gpu = false;
    }
    const pbd::SolverType global = gpu ? pbd::GPU_GLOBAL : pbd::CPU_GLOBAL;
    const pbd::SolverType active = gpu ? pbd::GPU_ACTIVE : pbd::CPU_ACTIVE;
    if (!app.solver.Init(global, app.config, state) || !app.solverB.Init(active, app.config, state))
    {
        Flash(app, "could not start the compare pair");
        return false;
    }
    app.type = global;
    app.cloth = state;
    app.clothB = state;
    app.grabIndex = -1;
    app.solver.SetGrab(-1, Vec3{0, 0, 0});
    app.solverB.SetGrab(-1, Vec3{0, 0, 0});
    app.avgA.Reset();
    app.avgB.Reset();
    app.maxDifference = 0.0f;
    return true;
}

// (Re)start the solver from `state`, keeping positions and velocities.
bool StartSolver(App& app, pbd::SolverType t, const pbd::Cloth& state)
{
    if ((t == pbd::GPU_GLOBAL || t == pbd::GPU_ACTIVE) && !app.gpuAvailable)
    {
        Flash(app, std::string(TypeLabel(t)) + ": no CUDA device in this build or machine");
        return false;
    }
    if (!app.solver.Init(t, app.config, state))
    {
        Flash(app, std::string("could not start ") + TypeLabel(t));
        return false;
    }
    app.type = t;
    app.grabIndex = -1;
    app.solver.SetGrab(-1, Vec3{0, 0, 0});
    app.avgA.Reset();
    return true;
}

void BuildScene(App& app)
{
    const Scene& s = kScenes[app.sceneIndex];
    const int n = kResolutions[app.resolutionIndex];
    pbd::ClothDesc desc;
    desc.nx = desc.ny = n;
    desc.spacing = 1.6f / static_cast<float>(n);
    desc.pin = s.pin;
    app.rest = pbd::Cloth::makeGrid(desc);
    app.cloth = app.rest;
    app.config.gravity = Vec3{0.0f, -s.gravity, 0.0f};
    app.config.damping = s.gravity > 0.0f ? 0.02f : 0.1f;

    // frame the cloth: centre of the grid, distance from its size
    app.camera.target = Vec3{0.0f, -0.8f, 0.0f};
    app.camera.distance = app.compare ? 4.6f : 3.0f;
    app.renderer->Rebuild(app.rest);
    app.rendererB->Rebuild(app.rest);
    if (app.compare)
    {
        StartCompare(app, IsGpu(app.type), app.cloth);
    }
    else if (!StartSolver(app, app.type, app.cloth))
    {
        StartSolver(app, pbd::CPU_GLOBAL, app.cloth);
    }
    app.renderer->Update(app.cloth, {}, 0, Offset(app, 0));
    app.rendererB->Update(app.clothB, {}, 0, Offset(app, 1));
}

void ToggleCompare(App& app)
{
    if (!app.compare)
    {
        app.solver.GetCloth(app.cloth);
        app.compare = true;
        if (!StartCompare(app, IsGpu(app.type), app.cloth))
        {
            app.compare = false;
            return;
        }
        app.camera.distance = 4.6f;
        Flash(app, std::string("compare: ") + (IsGpu(app.type) ? "gpu" : "cpu") + " global (left) vs active (right)");
    }
    else
    {
        // keep running the active variant
        app.solverB.GetCloth(app.clothB);
        const pbd::SolverType active = IsGpu(app.type) ? pbd::GPU_ACTIVE : pbd::CPU_ACTIVE;
        app.compare = false;
        app.cloth = app.clothB;
        StartSolver(app, active, app.cloth);
        app.camera.distance = 3.0f;
        Flash(app, "compare off");
    }
}

// ----------------------------------------------------------------------------------
// interaction
// ----------------------------------------------------------------------------------
void BeginGrab(App& app, int x, int y)
{
    const Ray ray = app.camera.ScreenRay(static_cast<float>(x), static_cast<float>(y), app.width, app.height);
    int best = -1, bestSide = 0;
    float bestAngle = 0.03f * 0.03f; // accept particles within ~1.7 degrees of the ray
    for (int side = 0; side < (app.compare ? 2 : 1); ++side)
    {
        const pbd::Cloth& c = side == 0 ? app.cloth : app.clothB;
        const Vec3 offset = Offset(app, side);
        for (int k = 0; k < c.particleCount(); ++k)
        {
            if (c.invMass[k] == 0.0f)
            {
                continue;
            }
            const Vec3 p = c.position[k] + offset;
            const float t = pbd::dot(p - ray.origin, ray.dir);
            if (t <= 0.0f)
            {
                continue;
            }
            const float angle = viewer::DistanceToRaySq(ray, p) / (t * t);
            if (angle < bestAngle)
            {
                bestAngle = angle;
                best = k;
                bestSide = side;
            }
        }
    }
    if (best < 0)
    {
        return;
    }
    app.grabIndex = best;
    app.grabSide = bestSide;
    app.grabPlanePoint = (bestSide == 0 ? app.cloth : app.clothB).position[best] + Offset(app, bestSide);
    app.grabPlaneNormal = app.camera.Forward();
    app.grabTarget = app.grabPlanePoint;
    SetCapture(app.hwnd);
}

void DragGrab(App& app, int x, int y)
{
    const Ray ray = app.camera.ScreenRay(static_cast<float>(x), static_cast<float>(y), app.width, app.height);
    Vec3 hit;
    if (viewer::IntersectPlane(ray, app.grabPlanePoint, app.grabPlaneNormal, hit))
    {
        app.grabTarget = hit;
    }
}

void EndGrab(App& app)
{
    app.grabIndex = -1;
    app.solver.SetGrab(-1, Vec3{0, 0, 0});
    app.solverB.SetGrab(-1, Vec3{0, 0, 0});
    ReleaseCapture();
}

void OnKey(App& app, WPARAM key)
{
    switch (key)
    {
    case VK_ESCAPE:
        app.running = false;
        break;
    case VK_SPACE:
        app.paused = !app.paused;
        break;
    case '1':
    case '2':
    case '3':
    case '4':
    {
        const pbd::SolverType t = static_cast<pbd::SolverType>(key - '1');
        app.solver.GetCloth(app.cloth);
        if (app.compare)
        {
            if (StartCompare(app, IsGpu(t), app.cloth))
            {
                Flash(app, std::string("compare: ") + (IsGpu(app.type) ? "gpu" : "cpu") + " global vs active");
            }
        }
        else if (StartSolver(app, t, app.cloth))
        {
            Flash(app, std::string("solver: ") + TypeLabel(t));
        }
        break;
    }
    case 'C':
        ToggleCompare(app);
        break;
    case 'R':
        BuildScene(app);
        Flash(app, "reset");
        break;
    case 'P':
        app.sceneIndex = (app.sceneIndex + 1) % 2;
        BuildScene(app);
        Flash(app, std::string("scene: ") + kScenes[app.sceneIndex].name);
        break;
    case 'N':
        app.resolutionIndex = (app.resolutionIndex + 1) % static_cast<int>(sizeof(kResolutions) / sizeof(int));
        BuildScene(app);
        Flash(app, "resolution: " + std::to_string(kResolutions[app.resolutionIndex]) + " x " +
                       std::to_string(kResolutions[app.resolutionIndex]));
        break;
    case 'W':
        app.draw.wireframe = !app.draw.wireframe;
        break;
    case 'B':
        app.draw.showBlocks = !app.draw.showBlocks;
        break;
    default:
        break;
    }
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    App* app = gApp;
    if (!app)
    {
        return DefWindowProcA(hwnd, msg, wp, lp);
    }
    const int x = static_cast<short>(LOWORD(lp)), y = static_cast<short>(HIWORD(lp));
    switch (msg)
    {
    case WM_CLOSE:
        app->running = false;
        return 0;
    case WM_SIZE:
        app->width = LOWORD(lp);
        app->height = HIWORD(lp);
        return 0;
    case WM_KEYDOWN:
        OnKey(*app, wp);
        return 0;
    case WM_LBUTTONDOWN:
        BeginGrab(*app, x, y);
        return 0;
    case WM_LBUTTONUP:
        if (app->grabIndex >= 0)
        {
            EndGrab(*app);
        }
        return 0;
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
        app->orbiting = true;
        app->lastX = x;
        app->lastY = y;
        SetCapture(hwnd);
        return 0;
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
        app->orbiting = false;
        ReleaseCapture();
        return 0;
    case WM_MOUSEMOVE:
        if (app->grabIndex >= 0)
        {
            DragGrab(*app, x, y);
        }
        else if (app->orbiting)
        {
            app->camera.Orbit(static_cast<float>(x - app->lastX), static_cast<float>(y - app->lastY));
            app->lastX = x;
            app->lastY = y;
        }
        return 0;
    case WM_MOUSEWHEEL:
        app->camera.Zoom(static_cast<float>(GET_WHEEL_DELTA_WPARAM(wp)) / WHEEL_DELTA);
        return 0;
    default:
        return DefWindowProcA(hwnd, msg, wp, lp);
    }
}

// ----------------------------------------------------------------------------------
// window + context
// ----------------------------------------------------------------------------------
bool CreateGlWindow(App& app, HINSTANCE inst)
{
    WNDCLASSA wc = {};
    wc.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = "PbdViewerWindow";
    RegisterClassA(&wc);

    RECT r = {0, 0, app.width, app.height};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    app.hwnd = CreateWindowA(wc.lpszClassName, "CUDA PBD Sandbox", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                             r.right - r.left, r.bottom - r.top, nullptr, nullptr, inst, nullptr);
    if (!app.hwnd)
    {
        return false;
    }
    app.hdc = GetDC(app.hwnd);

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    const int format = ChoosePixelFormat(app.hdc, &pfd);
    if (!format || !SetPixelFormat(app.hdc, format, &pfd))
    {
        return false;
    }

    // A legacy context is needed first to fetch wglCreateContextAttribsARB.
    HGLRC legacy = wglCreateContext(app.hdc);
    wglMakeCurrent(app.hdc, legacy);
    auto createAttribs =
        reinterpret_cast<PFN_wglCreateContextAttribsARB>(reinterpret_cast<void*>(wglGetProcAddress("wglCreateContextAttribsARB")));
    app.glrc = legacy;
    if (createAttribs)
    {
        const int attribs[] = {WGL_CONTEXT_MAJOR_VERSION_ARB, 3, WGL_CONTEXT_MINOR_VERSION_ARB, 3,
                               WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB, 0};
        HGLRC modern = createAttribs(app.hdc, nullptr, attribs);
        if (modern)
        {
            wglMakeCurrent(app.hdc, modern);
            wglDeleteContext(legacy);
            app.glrc = modern;
        }
    }
    if (auto swapInterval = reinterpret_cast<PFN_wglSwapIntervalEXT>(reinterpret_cast<void*>(wglGetProcAddress("wglSwapIntervalEXT"))))
    {
        swapInterval(1); // vsync
    }

    // Bitmap font for the overlay text.
    HFONT font = CreateFontA(-16, 0, 0, 0, FW_MEDIUM, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_TT_PRECIS,
                             CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FF_DONTCARE | DEFAULT_PITCH, "Segoe UI");
    SelectObject(app.hdc, font);
    app.fontBase = glGenLists(96);
    wglUseFontBitmapsA(app.hdc, 32, 96, app.fontBase);
    return true;
}

void DrawHudText(const App& app, int x, int y, const std::string& text, float r, float g, float b)
{
    glColor3f(r, g, b);
    glRasterPos2i(x, y);
    glListBase(app.fontBase - 32);
    glCallLists(static_cast<GLsizei>(text.size()), GL_UNSIGNED_BYTE, text.c_str());
}

void DrawPanel(int x0, int y0, int x1, int y1)
{
    glColor4f(1.0f, 1.0f, 1.0f, 0.84f);
    glBegin(GL_QUADS);
    glVertex2i(x0, y0);
    glVertex2i(x1, y0);
    glVertex2i(x1, y1);
    glVertex2i(x0, y1);
    glEnd();
    glColor4f(0.95f, 0.33f, 0.11f, 1.0f);
    glBegin(GL_QUADS);
    glVertex2i(x0, y0);
    glVertex2i(x0 + 4, y0);
    glVertex2i(x0 + 4, y1);
    glVertex2i(x0, y1);
    glEnd();
}

// World point -> pixel (top-left origin); false when behind the camera.
bool ProjectToScreen(const App& app, Vec3 p, int& sx, int& sy)
{
    const viewer::Mat4 m =
        app.camera.Projection(static_cast<float>(app.width) / (app.height > 0 ? app.height : 1)) * app.camera.View();
    const float x = m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12];
    const float y = m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13];
    const float w = m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15];
    if (w <= 0.0f)
    {
        return false;
    }
    sx = static_cast<int>((x / w * 0.5f + 0.5f) * app.width);
    sy = static_cast<int>((1.0f - (y / w * 0.5f + 0.5f)) * app.height);
    return true;
}

void DrawSinglePanel(App& app)
{
    DrawPanel(16, 16, 480, 150);
    char line[256];
    const App::Average& a = app.avgA;
    std::snprintf(line, sizeof(line), "%s%s%s", app.solver.GetName(),
                  app.stats.globalFallback ? "   -> global fallback (most blocks awake)" : "",
                  app.paused ? "   (paused)" : "");
    DrawHudText(app, 34, 42, line, 0.08f, 0.09f, 0.11f);
    std::snprintf(line, sizeof(line), "step %.2f ms   solve %.2f ms   iterations %.0f", a.step, a.solve, a.iterations);
    DrawHudText(app, 34, 68, line, 0.25f, 0.28f, 0.31f);
    std::snprintf(line, sizeof(line), "active blocks %.1f %%   constraints %.0f", a.active, a.constraints);
    DrawHudText(app, 34, 90, line, 0.25f, 0.28f, 0.31f);
    std::snprintf(line, sizeof(line), "%d x %d cloth   %.0f fps   %s", app.rest.nx, app.rest.ny, app.fps,
                  kScenes[app.sceneIndex].name);
    DrawHudText(app, 34, 112, line, 0.25f, 0.28f, 0.31f);
    DrawHudText(app, 34, 136, "1-4 solver  C compare  LMB pull  RMB orbit  P scene  N size  W wire  B blocks", 0.45f,
                0.48f, 0.52f);
}

void DrawComparePanel(App& app)
{
    DrawPanel(16, 16, 520, 236);
    const App::Average& a = app.avgA;
    const App::Average& b = app.avgB;
    const int c0 = 34, c1 = 220, c2 = 340;
    char line[256], va[64], vb[64];
    std::snprintf(line, sizeof(line), "compare  %s%s", IsGpu(app.type) ? "gpu" : "cpu", app.paused ? "   (paused)" : "");
    DrawHudText(app, c0, 42, line, 0.08f, 0.09f, 0.11f);
    DrawHudText(app, c1, 66, "global (left)", 0.43f, 0.38f, 0.72f);
    DrawHudText(app, c2, 66, app.statsB.globalFallback ? "active -> global" : "active (right)", 0.12f, 0.58f,
                0.66f);

    struct Row
    {
        const char* label;
        double a, b;
        const char* fmt;
    } rows[] = {{"step ms", a.step, b.step, "%.2f"},
                {"solve ms", a.solve, b.solve, "%.2f"},
                {"iterations", a.iterations, b.iterations, "%.0f"},
                {"constraints", a.constraints, b.constraints, "%.0f"},
                {"active blocks %", 100.0, b.active, "%.1f"}};
    int y = 90;
    for (const Row& r : rows)
    {
        std::snprintf(va, sizeof(va), r.fmt, r.a);
        std::snprintf(vb, sizeof(vb), r.fmt, r.b);
        DrawHudText(app, c0, y, r.label, 0.45f, 0.48f, 0.52f);
        DrawHudText(app, c1, y, va, 0.15f, 0.17f, 0.19f);
        DrawHudText(app, c2, y, vb, 0.15f, 0.17f, 0.19f);
        y += 22;
    }
    const double speedup = b.step > 1e-6 ? a.step / b.step : 0.0;
    std::snprintf(line, sizeof(line), "speedup  x%.2f  (step time, global / active)", speedup);
    DrawHudText(app, c0, y + 4, line, 0.95f, 0.33f, 0.11f);
    std::snprintf(line, sizeof(line), "max |x_global - x_active|  %.2f mm   %.0f fps", app.maxDifference * 1000.0f,
                  app.fps);
    DrawHudText(app, c0, y + 28, line, 0.25f, 0.28f, 0.31f);

    // labels above the two cloths
    int sx, sy;
    if (ProjectToScreen(app, Offset(app, 0) + Vec3{-0.75f, 0.12f, 0.0f}, sx, sy))
    {
        DrawHudText(app, sx, sy, "GLOBAL", 0.43f, 0.38f, 0.72f);
    }
    if (ProjectToScreen(app, Offset(app, 1) + Vec3{-0.75f, 0.12f, 0.0f}, sx, sy))
    {
        DrawHudText(app, sx, sy, "ACTIVE", 0.12f, 0.58f, 0.66f);
    }
}

void DrawOverlay(App& app)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, app.width, app.height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (app.compare)
    {
        DrawComparePanel(app);
    }
    else
    {
        DrawSinglePanel(app);
    }
    if (!app.message.empty() && Now(app) < app.messageUntil)
    {
        DrawHudText(app, 24, app.height - 24, app.message, 0.95f, 0.33f, 0.11f);
    }
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void Frame(App& app)
{
    const float dt = 1.0f / 60.0f;
    // The same grab input (in each cloth's own coordinates) drives both solvers.
    const Vec3 localTarget = app.grabTarget - Offset(app, app.grabSide);
    if (!app.paused)
    {
        app.solver.SetGrab(app.grabIndex, localTarget);
        app.stats = app.solver.Step(dt);
        app.avgA.Add(app.stats);
        if (app.compare)
        {
            app.solverB.SetGrab(app.grabIndex, localTarget);
            app.statsB = app.solverB.Step(dt);
            app.avgB.Add(app.statsB);
        }
    }
    app.solver.GetCloth(app.cloth);
    app.renderer->Update(app.cloth, app.solver.GetActiveBlocks(), app.solver.GetBlockSize(), Offset(app, 0));
    if (app.compare)
    {
        app.solverB.GetCloth(app.clothB);
        app.rendererB->Update(app.clothB, app.solverB.GetActiveBlocks(), app.solverB.GetBlockSize(), Offset(app, 1));
        if (++app.frameCounter % 15 == 0)
        {
            float d = 0.0f;
            for (int k = 0; k < app.cloth.particleCount(); ++k)
            {
                d = std::max(d, pbd::length(app.cloth.position[k] - app.clothB.position[k]));
            }
            app.maxDifference = d;
        }
    }

    app.renderer->SetGrab(false, Vec3{0, 0, 0}, Vec3{0, 0, 0});
    app.rendererB->SetGrab(false, Vec3{0, 0, 0}, Vec3{0, 0, 0});
    if (app.grabIndex >= 0)
    {
        const pbd::Cloth& c = app.grabSide == 0 ? app.cloth : app.clothB;
        viewer::ClothRenderer& r = app.grabSide == 0 ? *app.renderer : *app.rendererB;
        r.SetGrab(true, c.position[app.grabIndex] + Offset(app, app.grabSide), app.grabTarget);
    }

    glViewport(0, 0, app.width, app.height);
    glClearColor(0.953f, 0.961f, 0.969f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    app.renderer->Draw(app.camera, app.width, app.height, app.draw);
    if (app.compare)
    {
        viewer::DrawOptions second = app.draw;
        second.drawFloor = false;
        app.rendererB->Draw(app.camera, app.width, app.height, second);
    }
    DrawOverlay(app);
    SwapBuffers(app.hdc);

    ++app.fpsFrames;
    const double t = Now(app);
    if (t - app.fpsTimer >= 0.5)
    {
        app.fps = app.fpsFrames / (t - app.fpsTimer);
        app.fpsFrames = 0;
        app.fpsTimer = t;
    }
}
} // namespace

int WINAPI WinMain(HINSTANCE inst, HINSTANCE, LPSTR, int show)
{
    App app;
    gApp = &app;
    QueryPerformanceFrequency(&app.frequency);
    QueryPerformanceCounter(&app.start);

    if (!CreateGlWindow(app, inst))
    {
        MessageBoxA(nullptr, "Could not create an OpenGL window.", "pbd_viewer", MB_ICONERROR);
        return 1;
    }
    if (const char* missing = gl::LoadFunctions())
    {
        MessageBoxA(nullptr, (std::string("OpenGL 3.3 function missing: ") + missing).c_str(), "pbd_viewer",
                    MB_ICONERROR);
        return 1;
    }
    app.renderer.reset(new viewer::ClothRenderer());
    app.rendererB.reset(new viewer::ClothRenderer());
    const char* error = nullptr;
    if (!app.renderer->Init(&error) || !app.rendererB->Init(&error))
    {
        MessageBoxA(nullptr, error ? error : "shader error", "pbd_viewer", MB_ICONERROR);
        return 1;
    }
#if defined(PBD_WITH_CUDA)
    app.gpuAvailable = pbd::cudaDeviceAvailable();
    app.type = app.gpuAvailable ? pbd::GPU_ACTIVE : pbd::CPU_ACTIVE;
#else
    app.type = pbd::CPU_ACTIVE;
#endif
    app.config.substeps = 8;
    app.config.maxIterations = 10;
    BuildScene(app);
    ShowWindow(app.hwnd, show);

    while (app.running)
    {
        MSG msg;
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (!app.running)
        {
            break;
        }
        Frame(app);
    }

    app.renderer.reset(); // GL objects must go before the context
    app.rendererB.reset();
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(app.glrc);
    ReleaseDC(app.hwnd, app.hdc);
    DestroyWindow(app.hwnd);
    gApp = nullptr;
    return 0;
}
