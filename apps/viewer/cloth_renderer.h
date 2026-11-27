// Draws the cloth with OpenGL 3.3: shaded two-sided surface coloured by
// out-of-plane displacement, optional wireframe, the outlines of the blocks the
// active solver touched, a floor grid and the grab tether.
#pragma once
#include <vector>

#include "camera.h"
#include "gl_api.h"
#include "pbd/cloth.h"

namespace viewer
{
struct DrawOptions
{
    bool wireframe = false;
    bool showBlocks = true;
    bool drawFloor = true;
    float displacementScale = 10.0f; // colour saturates at 1/scale metres
};

class ClothRenderer
{
  public:
    ClothRenderer() = default;
    ~ClothRenderer();
    ClothRenderer(const ClothRenderer&) = delete;
    ClothRenderer& operator=(const ClothRenderer&) = delete;

    bool Init(const char** error);                      // compile shaders, create floor
    void Rebuild(const pbd::Cloth& rest);               // new topology (resolution / scene changed)
    // `offset` shifts the drawn mesh in world space (used to place two cloths side by side).
    void Update(const pbd::Cloth& cloth, const std::vector<unsigned char>& activeBlocks, int blockSize,
                Vec3 offset = Vec3{0, 0, 0});
    void SetGrab(bool on, Vec3 particle, Vec3 target);
    void Draw(const OrbitCamera& camera, int width, int height, const DrawOptions& options);

  private:
    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
        float displacement;
        float active;
    };

    GLuint program = 0;
    GLint uMvp = -1, uEye = -1, uLight = -1, uMode = -1, uColor = -1, uDispScale = -1, uShowActive = -1;

    GLuint clothVao = 0, clothVbo = 0, triEbo = 0, wireEbo = 0, blockEbo = 0;
    GLuint floorVao = 0, floorVbo = 0;
    GLuint grabVao = 0, grabVbo = 0;
    GLsizei triCount = 0, wireCount = 0, blockCount = 0, floorCount = 0;
    float floorY = -1.0f;

    int nx = 0, ny = 0;
    std::vector<pbd::Vec3> restPosition;
    std::vector<Vertex> vertices;
    std::vector<unsigned> blockLines;
    bool grabOn = false;
    Vec3 grabParticle{0, 0, 0}, grabTarget{0, 0, 0};

    void BuildFloor(float y, float halfSize);
    void Release();
};
} // namespace viewer
