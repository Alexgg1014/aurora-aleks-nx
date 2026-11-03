// v1.1 (aleks2): mods' gfx service on the deko3d renderer (dk_modgfx.hpp). MIT, this project's.
//
// With Dawn, a mod's runtime draw type gets the scene pass's WGPURenderPassEncoder and its encoder task the
// frame's WGPUCommandEncoder, and the mod records into them with the wgpu* C API. With deko3d, Dawn is the Null
// device: the objects a mod creates still exist and validate there, but nothing it records reaches the GPU,
// so until now any gfx-service draw sent the game back to Vulkan (deko3d_policy::note_gfx_mod).
//
// Here the mod's wgpu* imports are bound (by the game's loader, aurora_dk_wgpu_proc) to functions that, while
// deko3d renders, do the work on the render worker's deko3d command buffer and otherwise call Dawn's:
//   objects     textures, views, samplers and bind groups through Aurora's --wrap'ed creators (their deko3d
//               shadows, dk_objects.cpp); buffers get a deko3d block of their own (Buf); a shader module
//               keeps its WGSL; a render or compute pipeline is translated like Aurora's own shaders (Tint
//               with the Sequential binding scheme, then the uam worker: shader_translate.h) into ModPipe.
//               Pipeline creation waits for uam, as Dawn creates pipelines synchronously, so work a mod
//               pushes once is not lost to a shader still compiling.
//   encoders    the pass a draw type gets, and the encoder a task gets, are our own handles (a magic word
//               first, never handed to Dawn); a task's BeginRenderPass / BeginComputePass return more of
//               them. Every wgpu*PassEncoder* and wgpuCommandEncoder* function is ours, so none of Dawn's
//               ever sees one: SetPipeline / SetBindGroup / Draw / Dispatch / the copies become deko3d
//               commands, debug groups and labels do nothing, and the rest count as failures. A pipeline's
//               WGSL bindings are looked up in the bound groups' shadows (held from SetBindGroup until the
//               pass ends) and bound to the slots the translation gave them
//   writes      QueueWriteBuffer / QueueWriteTexture go through a staging copy recorded in the command
//               stream (inside a callback at that point, otherwise at the render worker's next op), never
//               into memory a frame in flight still reads
//   readback    MapAsync completes once the render worker has recorded the game frame the request was made
//               in and the GPU has finished it, from ProcessEvents (the mod's or Aurora's per-frame pump);
//               the mapped range is the block itself (CPU-uncached)
// Anything not done (a present target, a shader that does not translate or compile, a call with no deko3d
// equivalent here) counts in aurora_dk_modgfx_failures, per thread, so the game moves the mod whose callback
// it was to Vulkan as before.
#include "dk_modgfx.hpp"

#include "dk_encode.hpp"
#include "dk_objects.hpp"
#include "dk_pipeline.hpp"
#include "dk_runtime.hpp"
#include "dksh_file.h"
#include "shader_translate.h"
#include "../thread.hpp"

#include "gfx/frame.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/resources.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aurora::gfx::dk {
using namespace aurora::gfx::detail;
using swdk::dklog;
using swdk::R;
using swdk::ShaderStage;
using swdk::SlotBinding;

namespace {

constexpr uint64_t kMagicPass = 0x73736170646f6d31ull;     // "1modpass"
constexpr uint64_t kMagicCompute = 0x706d6f63646f6d31ull;  // "1modcomp"
constexpr uint64_t kMagicEncoder = 0x636e6564646f6d31ull;  // "1moddenc"
constexpr uint32_t kMaxGroups = 4;
constexpr uint32_t kMaxUniformSize = 0x10000;
constexpr uint32_t kGpuWritten = DkInvalidateFlags_Image | DkInvalidateFlags_Shader;

// per thread: the game compares it around one mod's callback, on the thread that runs it
thread_local uint32_t t_failures = 0;
thread_local bool t_inCallback = false;  // the render worker is inside a mod's draw or task callback
std::atomic<uint64_t> g_encodedFrame{0};  // the newest frame the render worker recorded mods' work into

void count_failure() { t_failures++; }

void fail_once(const std::string& what) {
    count_failure();
    static std::mutex m;
    static std::vector<std::string> seen;
    std::lock_guard<std::mutex> lock(m);
    if (std::find(seen.begin(), seen.end(), what) != seen.end()) return;
    seen.push_back(what);
    dklog("mods' gfx: %s (not done with deko3d; that mod goes to Vulkan from the next start)", what.c_str());
}

void note_once(const char* what) {
    static std::mutex m;
    static std::vector<std::string> seen;
    std::lock_guard<std::mutex> lock(m);
    for (const auto& s : seen)
        if (s == what) return;
    seen.emplace_back(what);
    dklog("mods' gfx: %s", what);
}

std::string view_string(WGPUStringView s) {
    if (!s.data) return {};
    return s.length == WGPU_STRLEN ? std::string(s.data) : std::string(s.data, s.length);
}

uint64_t recording_frame() { return R.frame + 1; }

// ---- game frame -> deko3d frame (render worker writes, game thread reads; g_ringMutex)
struct IndexFrame {
    uint32_t index = UINT32_MAX;
    uint64_t dkFrame = 0;
};
constexpr uint32_t kRing = 64;
std::mutex g_ringMutex;
IndexFrame g_ring[kRing];
uint32_t g_encodedIndex = UINT32_MAX;  // the newest game frame the render worker has started recording

// the deko3d frame that recorded game frame index, 0 if not recorded yet; UINT64_MAX: too long ago to know
// (anything that frame did is long finished)
uint64_t dk_frame_of(uint32_t index) {
    std::lock_guard<std::mutex> lock(g_ringMutex);
    if (index == UINT32_MAX || g_encodedIndex == UINT32_MAX) return UINT64_MAX;
    const IndexFrame& e = g_ring[index % kRing];
    if (e.index == index) return e.dkFrame;
    return int32_t(g_encodedIndex - index) > 0 ? UINT64_MAX : 0;
}

// ---- objects (any thread; g_mutex)
std::shared_mutex g_mutex;

struct Module {
    int32_t refs = 1;
    std::string wgsl;
};
std::unordered_map<const void*, Module> g_modules;

// bindings with a dynamic offset, in binding order (WebGPU's order of the offsets)
std::unordered_map<const void*, std::vector<uint32_t>> g_layoutDynamic;

struct Buf {
    std::atomic<int32_t> refs{1};
    WGPUBuffer handle = nullptr;
    swdk::Block block;
    uint64_t size = 0, usage = 0;
    std::atomic<uint64_t> lastFrame{0};  // the newest frame whose commands use it
    // mapping (under g_mutex)
    bool mapped = false, dawnMapped = false, mapPending = false;
    uint32_t mapIndex = UINT32_MAX;  // the game frame the request was made in
    WGPUBufferMapCallbackInfo callback{};
};
std::unordered_map<const void*, Buf*> g_buffers;

struct Group {
    std::vector<uint32_t> dynamic;
    std::vector<Buf*> buffers;  // the bind group holds our buffers as Dawn holds its own
};
std::unordered_map<const void*, Group> g_groups;

struct StageCode {
    swdk::ShaderEntry* entry = nullptr;
    std::vector<SlotBinding> bindings;
    std::vector<swdk::SamplerSource> samplers;
};
struct ModPipe {
    bool compute = false;
    StageCode vs, fs, cs;
    DkPrimitive primitive = DkPrimitive_Triangles;
    DkRasterizerState rasterizer{};
    DkColorState color{};
    DkColorWriteState colorWrite{};
    DkBlendState blend[8]{};
    uint32_t targets = 0;
    DkDepthStencilState depthStencil{};
    std::vector<DkVtxAttribState> attribs;
    std::vector<DkVtxBufferState> vertexBuffers;
    std::atomic<bool> failed{false};
};
// pipelines live as long as the process (a few hundred bytes each; a queued frame may still name one)
std::unordered_map<const void*, ModPipe*> g_pipes;

// buffers whose last reference is gone, freed once the GPU is past the frames that may use them
struct Dead {
    uint64_t frame = 0;
    Buf* buf = nullptr;
};
std::mutex g_deadMutex;
std::vector<Dead> g_dead;

void bury(Buf* b) {
    swdk::block_free_deferred(b->block.block);
    std::lock_guard<std::mutex> lock(g_deadMutex);
    g_dead.push_back({g_encodedFrame.load(std::memory_order_relaxed) + swdk::kFrames + 1, b});
}

void dig_up() {
    const uint64_t done = swdk::frames_completed();
    std::lock_guard<std::mutex> lock(g_deadMutex);
    auto it = std::remove_if(g_dead.begin(), g_dead.end(), [&](const Dead& d) {
        if (d.frame > done) return false;
        delete d.buf;
        return true;
    });
    g_dead.erase(it, g_dead.end());
}

void unref(Buf* b) {
    if (b->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    {
        std::unique_lock<std::shared_mutex> lock(g_mutex);
        auto it = g_buffers.find(b->handle);
        if (it != g_buffers.end() && it->second == b) g_buffers.erase(it);
    }
    bury(b);
}

template <class T> T* find_in(const std::unordered_map<const void*, T*>& map, const void* key) {
    if (!key) return nullptr;
    std::shared_lock<std::shared_mutex> lock(g_mutex);
    auto it = map.find(key);
    return it == map.end() ? nullptr : it->second;
}

// a bind group reference (Dawn's and its shadow's, through the --wrap'ed functions); the last one also drops
// the references it held on our buffers
void group_addref(WGPUBindGroup g) {
    if (g) wgpuBindGroupAddRef(g);
}
void group_release(WGPUBindGroup g) {
    if (!g) return;
    wgpuBindGroupRelease(g);
    if (!aurora_dk_active() || find_bind_group(g)) return;  // (still referenced)
    std::vector<Buf*> buffers;
    {
        std::unique_lock<std::shared_mutex> lock(g_mutex);
        auto it = g_groups.find(g);
        if (it == g_groups.end()) return;
        buffers = std::move(it->second.buffers);
        g_groups.erase(it);
    }
    for (Buf* b : buffers) unref(b);
}

// ---- pipelines
DkCompareOp compare_op(WGPUCompareFunction f) {
    switch (f) {
    case WGPUCompareFunction_Never: return DkCompareOp_Never;
    case WGPUCompareFunction_Less: return DkCompareOp_Less;
    case WGPUCompareFunction_Equal: return DkCompareOp_Equal;
    case WGPUCompareFunction_LessEqual: return DkCompareOp_Lequal;
    case WGPUCompareFunction_Greater: return DkCompareOp_Greater;
    case WGPUCompareFunction_NotEqual: return DkCompareOp_NotEqual;
    case WGPUCompareFunction_GreaterEqual: return DkCompareOp_Gequal;
    default: return DkCompareOp_Always;
    }
}

DkBlendFactor blend_factor(WGPUBlendFactor f, bool alpha) {
    switch (f) {
    case WGPUBlendFactor_Zero: return DkBlendFactor_Zero;
    case WGPUBlendFactor_One: return DkBlendFactor_One;
    case WGPUBlendFactor_Src: return DkBlendFactor_SrcColor;
    case WGPUBlendFactor_OneMinusSrc: return DkBlendFactor_InvSrcColor;
    case WGPUBlendFactor_SrcAlpha: return DkBlendFactor_SrcAlpha;
    case WGPUBlendFactor_OneMinusSrcAlpha: return DkBlendFactor_InvSrcAlpha;
    case WGPUBlendFactor_Dst: return DkBlendFactor_DstColor;
    case WGPUBlendFactor_OneMinusDst: return DkBlendFactor_InvDstColor;
    case WGPUBlendFactor_DstAlpha: return DkBlendFactor_DstAlpha;
    case WGPUBlendFactor_OneMinusDstAlpha: return DkBlendFactor_InvDstAlpha;
    case WGPUBlendFactor_SrcAlphaSaturated: return DkBlendFactor_SrcAlphaSaturate;
    case WGPUBlendFactor_Constant: return alpha ? DkBlendFactor_ConstAlpha : DkBlendFactor_ConstColor;
    case WGPUBlendFactor_OneMinusConstant: return alpha ? DkBlendFactor_InvConstAlpha : DkBlendFactor_InvConstColor;
    case WGPUBlendFactor_Src1: return DkBlendFactor_Src1Color;
    case WGPUBlendFactor_OneMinusSrc1: return DkBlendFactor_InvSrc1Color;
    case WGPUBlendFactor_Src1Alpha: return DkBlendFactor_Src1Alpha;
    case WGPUBlendFactor_OneMinusSrc1Alpha: return DkBlendFactor_InvSrc1Alpha;
    default: return DkBlendFactor_One;
    }
}

DkBlendOp blend_op(WGPUBlendOperation o) {
    switch (o) {
    case WGPUBlendOperation_Subtract: return DkBlendOp_Sub;
    case WGPUBlendOperation_ReverseSubtract: return DkBlendOp_RevSub;
    case WGPUBlendOperation_Min: return DkBlendOp_Min;
    case WGPUBlendOperation_Max: return DkBlendOp_Max;
    default: return DkBlendOp_Add;
    }
}

DkPrimitive primitive_of(WGPUPrimitiveTopology t) {
    switch (t) {
    case WGPUPrimitiveTopology_PointList: return DkPrimitive_Points;
    case WGPUPrimitiveTopology_LineList: return DkPrimitive_Lines;
    case WGPUPrimitiveTopology_LineStrip: return DkPrimitive_LineStrip;
    case WGPUPrimitiveTopology_TriangleStrip: return DkPrimitive_TriangleStrip;
    default: return DkPrimitive_Triangles;
    }
}

// a WebGPU vertex format as deko3d's attribute size and type; false when there is none
bool vertex_format(WGPUVertexFormat f, DkVtxAttribSize& size, DkVtxAttribType& type) {
    switch (f) {
    case WGPUVertexFormat_Float32: size = DkVtxAttribSize_1x32, type = DkVtxAttribType_Float; return true;
    case WGPUVertexFormat_Float32x2: size = DkVtxAttribSize_2x32, type = DkVtxAttribType_Float; return true;
    case WGPUVertexFormat_Float32x3: size = DkVtxAttribSize_3x32, type = DkVtxAttribType_Float; return true;
    case WGPUVertexFormat_Float32x4: size = DkVtxAttribSize_4x32, type = DkVtxAttribType_Float; return true;
    case WGPUVertexFormat_Uint32: size = DkVtxAttribSize_1x32, type = DkVtxAttribType_Uint; return true;
    case WGPUVertexFormat_Uint32x2: size = DkVtxAttribSize_2x32, type = DkVtxAttribType_Uint; return true;
    case WGPUVertexFormat_Uint32x3: size = DkVtxAttribSize_3x32, type = DkVtxAttribType_Uint; return true;
    case WGPUVertexFormat_Uint32x4: size = DkVtxAttribSize_4x32, type = DkVtxAttribType_Uint; return true;
    case WGPUVertexFormat_Sint32: size = DkVtxAttribSize_1x32, type = DkVtxAttribType_Sint; return true;
    case WGPUVertexFormat_Sint32x2: size = DkVtxAttribSize_2x32, type = DkVtxAttribType_Sint; return true;
    case WGPUVertexFormat_Sint32x3: size = DkVtxAttribSize_3x32, type = DkVtxAttribType_Sint; return true;
    case WGPUVertexFormat_Sint32x4: size = DkVtxAttribSize_4x32, type = DkVtxAttribType_Sint; return true;
    case WGPUVertexFormat_Float16x2: size = DkVtxAttribSize_2x16, type = DkVtxAttribType_Float; return true;
    case WGPUVertexFormat_Float16x4: size = DkVtxAttribSize_4x16, type = DkVtxAttribType_Float; return true;
    case WGPUVertexFormat_Unorm16x2: size = DkVtxAttribSize_2x16, type = DkVtxAttribType_Unorm; return true;
    case WGPUVertexFormat_Unorm16x4: size = DkVtxAttribSize_4x16, type = DkVtxAttribType_Unorm; return true;
    case WGPUVertexFormat_Snorm16x2: size = DkVtxAttribSize_2x16, type = DkVtxAttribType_Snorm; return true;
    case WGPUVertexFormat_Snorm16x4: size = DkVtxAttribSize_4x16, type = DkVtxAttribType_Snorm; return true;
    case WGPUVertexFormat_Uint16x2: size = DkVtxAttribSize_2x16, type = DkVtxAttribType_Uint; return true;
    case WGPUVertexFormat_Uint16x4: size = DkVtxAttribSize_4x16, type = DkVtxAttribType_Uint; return true;
    case WGPUVertexFormat_Sint16x2: size = DkVtxAttribSize_2x16, type = DkVtxAttribType_Sint; return true;
    case WGPUVertexFormat_Sint16x4: size = DkVtxAttribSize_4x16, type = DkVtxAttribType_Sint; return true;
    case WGPUVertexFormat_Unorm8x2: size = DkVtxAttribSize_2x8, type = DkVtxAttribType_Unorm; return true;
    case WGPUVertexFormat_Unorm8x4: size = DkVtxAttribSize_4x8, type = DkVtxAttribType_Unorm; return true;
    case WGPUVertexFormat_Snorm8x2: size = DkVtxAttribSize_2x8, type = DkVtxAttribType_Snorm; return true;
    case WGPUVertexFormat_Snorm8x4: size = DkVtxAttribSize_4x8, type = DkVtxAttribType_Snorm; return true;
    case WGPUVertexFormat_Uint8x2: size = DkVtxAttribSize_2x8, type = DkVtxAttribType_Uint; return true;
    case WGPUVertexFormat_Uint8x4: size = DkVtxAttribSize_4x8, type = DkVtxAttribType_Uint; return true;
    case WGPUVertexFormat_Sint8x2: size = DkVtxAttribSize_2x8, type = DkVtxAttribType_Sint; return true;
    case WGPUVertexFormat_Sint8x4: size = DkVtxAttribSize_4x8, type = DkVtxAttribType_Sint; return true;
    default: return false;
    }
}

// one stage of a module: translated and compiled now (the uam worker; a pipeline of the local DKSH cache is
// ready at once), as Dawn's pipeline creation is synchronous
bool stage_code(WGPUShaderModule module, WGPUStringView entryPoint, ShaderStage stage, StageCode& out,
                std::string& error) {
    std::string wgsl;
    {
        std::shared_lock<std::shared_mutex> lock(g_mutex);
        auto it = g_modules.find(module);
        if (it == g_modules.end()) {
            error = "a shader module that is not WGSL (or was created before the renderer)";
            return false;
        }
        wgsl = it->second.wgsl;
    }
    swdk::TranslateResult r;
    {
        std::lock_guard<std::mutex> lock(translate_mutex());
        std::string entry = view_string(entryPoint);
        // aleks2 (v1.1-beta2): Tint's resolver recurses deep, and a mod may create its pipeline from wherever it draws:
        // DoF HQ does it in its custom draw, on the render worker, whose default libnx stack Tint overflowed
        // (AnalyzeUniformity). Tint runs on a thread of its own with 1 MiB, as the pipeline compilation thread does.
        bool ran = false;
        thread::LargeStackThread translator;
        if (translator.start(1u << 20, [&] {
                if (entry.empty()) entry = swdk::entry_point_of(wgsl, stage, &error);
                if (!entry.empty()) r = swdk::wgsl_to_glsl(wgsl, entry, stage, swdk::BindingScheme::Sequential);
                ran = true;
            })) {
            translator.join();
        }
        if (!ran) {
            if (entry.empty()) entry = swdk::entry_point_of(wgsl, stage, &error);
            if (entry.empty()) return false;
            r = swdk::wgsl_to_glsl(wgsl, entry, stage, swdk::BindingScheme::Sequential);
        }
        if (entry.empty()) return false;
    }
    if (!r.ok) {
        error = r.error.substr(0, 300);
        return false;
    }
    out.entry = swdk::shader_entry(stage, swdk::glsl_hash(r.glsl));
    swdk::shader_compile(out.entry, r.glsl, true);
    if (!swdk::shader_wait(out.entry)) {
        error = "uam does not compile it";
        return false;
    }
    out.bindings = std::move(r.bindings);
    out.samplers = std::move(r.samplers);
    return true;
}

void defaults(ModPipe* p) {
    memset(&p->rasterizer, 0, sizeof p->rasterizer);
    memset(&p->color, 0, sizeof p->color);
    memset(&p->colorWrite, 0, sizeof p->colorWrite);
    memset(p->blend, 0, sizeof p->blend);
    memset(&p->depthStencil, 0, sizeof p->depthStencil);
    dkRasterizerStateDefaults(&p->rasterizer);
    dkColorStateDefaults(&p->color);
    dkColorWriteStateDefaults(&p->colorWrite);
    for (auto& b : p->blend) dkBlendStateDefaults(&b);
    dkDepthStencilStateDefaults(&p->depthStencil);
}

ModPipe* build_render(const WGPURenderPipelineDescriptor& d) {
    auto* p = new ModPipe;
    defaults(p);
    std::string error;
    if (!stage_code(d.vertex.module, d.vertex.entryPoint, ShaderStage::Vertex, p->vs, error)) {
        fail_once("a render pipeline's vertex shader: " + error);
        p->failed = true;
        return p;
    }
    if (!d.fragment) {
        fail_once("a render pipeline without a fragment stage");
        p->failed = true;
        return p;
    }
    if (!stage_code(d.fragment->module, d.fragment->entryPoint, ShaderStage::Fragment, p->fs, error)) {
        fail_once("a render pipeline's fragment shader: " + error);
        p->failed = true;
        return p;
    }
    if (d.vertex.constantCount || d.fragment->constantCount) note_once("pipeline-overridable constants (ignored)");
    p->primitive = primitive_of(d.primitive.topology);
    DkRasterizerState& rs = p->rasterizer;
    rs.cullMode = d.primitive.cullMode == WGPUCullMode_Front  ? DkFace_Front
                  : d.primitive.cullMode == WGPUCullMode_Back ? DkFace_Back
                                                              : DkFace_None;
    const bool ccw = d.primitive.frontFace != WGPUFrontFace_CW;
    rs.frontFace = ccw != swdk::C.flipFront ? DkFrontFace_CCW : DkFrontFace_CW;
    if (d.depthStencil) {
        DkDepthStencilState& ds = p->depthStencil;
        ds.depthTestEnable = true;
        ds.depthWriteEnable = d.depthStencil->depthWriteEnabled == WGPUOptionalBool_True;
        ds.depthCompareOp = compare_op(d.depthStencil->depthCompare);
        if (d.depthStencil->depthBias != 0 || d.depthStencil->depthBiasSlopeScale != 0.f) {
            note_once("a pipeline's depth bias (ignored)");
        }
    } else {
        p->depthStencil.depthTestEnable = false;
        p->depthStencil.depthWriteEnable = false;
    }
    p->targets = uint32_t(std::min<size_t>(d.fragment->targetCount, 8));
    for (uint32_t i = 0; i < 8; i++) dkColorWriteStateSetMask(&p->colorWrite, i, 0);
    for (uint32_t i = 0; i < p->targets; i++) {
        const WGPUColorTargetState& t = d.fragment->targets[i];
        dkColorWriteStateSetMask(&p->colorWrite, i, uint32_t(t.writeMask) & 0xF);
        if (t.blend) {
            dkColorStateSetBlendEnable(&p->color, i, true);
            dkBlendStateSetOps(&p->blend[i], blend_op(t.blend->color.operation), blend_op(t.blend->alpha.operation));
            dkBlendStateSetFactors(&p->blend[i], blend_factor(t.blend->color.srcFactor, false),
                                   blend_factor(t.blend->color.dstFactor, false),
                                   blend_factor(t.blend->alpha.srcFactor, true),
                                   blend_factor(t.blend->alpha.dstFactor, true));
        }
    }
    for (size_t b = 0; b < d.vertex.bufferCount && b < 16; b++) {
        const WGPUVertexBufferLayout& l = d.vertex.buffers[b];
        p->vertexBuffers.push_back({uint32_t(l.arrayStride), l.stepMode == WGPUVertexStepMode_Instance ? 1u : 0u});
        for (size_t a = 0; a < l.attributeCount; a++) {
            const WGPUVertexAttribute& va = l.attributes[a];
            DkVtxAttribState s{};
            DkVtxAttribSize size{};
            DkVtxAttribType type{};
            if (!vertex_format(va.format, size, type)) {
                fail_once("a vertex attribute format");
                p->failed = true;
                return p;
            }
            s.size = size;
            s.type = type;
            s.bufferId = uint32_t(b);
            s.offset = uint32_t(va.offset);
            if (va.shaderLocation >= p->attribs.size()) p->attribs.resize(va.shaderLocation + 1);
            p->attribs[va.shaderLocation] = s;
        }
    }
    for (auto& a : p->attribs) {
        if (a.size == 0) a.isFixed = 1;  // a hole in the locations: a constant zero
    }
    return p;
}

ModPipe* build_compute(const WGPUComputePipelineDescriptor& d) {
    auto* p = new ModPipe;
    defaults(p);
    p->compute = true;
    std::string error;
    if (!stage_code(d.compute.module, d.compute.entryPoint, ShaderStage::Compute, p->cs, error)) {
        fail_once("a compute pipeline's shader: " + error);
        p->failed = true;
    }
    if (d.compute.constantCount) note_once("pipeline-overridable constants (ignored)");
    return p;
}

// ---- encoding (render worker)
struct BoundGroup {
    WGPUBindGroup group = nullptr;  // referenced while bound
    uint32_t dynamic[16] = {};
    uint32_t dynamicCount = 0;
};
struct VertexBinding {
    WGPUBuffer buffer = nullptr;
    uint64_t offset = 0, size = 0;
};
struct EncState {
    ModPipe* pipe = nullptr;
    BoundGroup groups[kMaxGroups];
    VertexBinding vertex[16];
    VertexBinding index;
    DkIdxFormat indexFormat = DkIdxFormat_Uint16;
};
struct ModPass {
    uint64_t magic = kMagicPass;
    std::atomic<int32_t> refs{1};
    bool scene = false;
    bool dead = false;  // its targets could not be bound: nothing it draws may land in what is bound
    bool ended = false;
    uint32_t width = 0, height = 0;
    EncState st;
};
struct ModCompute {
    uint64_t magic = kMagicCompute;
    std::atomic<int32_t> refs{1};
    bool ended = false;
    EncState st;
};
struct ModEncoder {
    uint64_t magic = kMagicEncoder;
};
ModPass g_scenePass;
ModEncoder g_taskEncoder;

template <class T> T* fake(const void* handle, uint64_t magic) {
    return handle && *static_cast<const uint64_t*>(handle) == magic ? static_cast<T*>(const_cast<void*>(handle))
                                                                    : nullptr;
}
ModPass* as_pass(WGPURenderPassEncoder p) { return aurora_dk_active() ? fake<ModPass>(p, kMagicPass) : nullptr; }
ModCompute* as_compute(WGPUComputePassEncoder p) {
    return aurora_dk_active() ? fake<ModCompute>(p, kMagicCompute) : nullptr;
}
ModEncoder* as_encoder(WGPUCommandEncoder e) {
    return aurora_dk_active() ? fake<ModEncoder>(e, kMagicEncoder) : nullptr;
}

void release_groups(EncState& st) {
    for (auto& g : st.groups) {
        group_release(g.group);
        g = BoundGroup{};
    }
}

// a buffer's GPU range: one of Aurora's shared streaming buffers (this frame's staging regions) or a mod's
bool buffer_range(WGPUBuffer buffer, uint64_t offset, uint64_t size, DkGpuAddr& addr, uint64_t& outSize,
                  Buf** outBuf = nullptr) {
    DkGpuAddr base = 0;
    uint64_t total = 0;
    if (!shared_buffer_range(buffer, base, total)) {
        Buf* b = find_in(g_buffers, buffer);
        if (!b || !b->block) return false;
        base = b->block.gpu;
        total = b->size;
        b->lastFrame.store(recording_frame(), std::memory_order_relaxed);
        if (outBuf) *outBuf = b;
    }
    if (offset >= total) return false;
    addr = base + offset;
    outSize = size == WGPU_WHOLE_SIZE ? total - offset : std::min(size, total - offset);
    return outSize > 0;
}

const BindGroup::Entry* group_entry(const EncState& st, uint32_t group, uint32_t binding, uint32_t* dynamicOffset) {
    if (group >= kMaxGroups || !st.groups[group].group) return nullptr;
    BindGroup* g = find_bind_group(st.groups[group].group);
    if (!g) return nullptr;
    *dynamicOffset = 0;
    {
        std::shared_lock<std::shared_mutex> lock(g_mutex);
        auto it = g_groups.find(st.groups[group].group);
        if (it != g_groups.end()) {
            const auto& dyn = it->second.dynamic;
            for (size_t i = 0; i < dyn.size() && i < st.groups[group].dynamicCount; i++)
                if (dyn[i] == binding) *dynamicOffset = st.groups[group].dynamic[i];
        }
    }
    for (const auto& e : g->entries)
        if (e.binding == binding) return &e;
    return nullptr;
}

uint32_t storage_slot(View* v) {
    if (!v || !v->tex || !v->tex->valid) return 0;
    if (v->storageSlot == 0) {
        const uint32_t slot = swdk::image_slot_alloc();
        if (slot == 0) return 0;
        DkImageView iv = v->view;
        iv.pImage = &v->tex->image;
        iv.mipLevelCount = 1;
        swdk::write_storage_image_descriptor(slot, iv);
        v->storageSlot = slot;
    }
    return v->storageSlot;
}

// a view as a render target: its level, layer and format
DkImageView target_of(View* v) {
    DkImageView iv = v->view;
    iv.pImage = &v->tex->image;
    iv.mipLevelCount = 1;
    return iv;
}

// one stage's resources; false (logged once, counted) when one is missing
bool bind_stage(DkStage stage, const StageCode& code, EncState& st) {
    size_t sampler = 0;
    bool ok = true;
    for (const SlotBinding& b : code.bindings) {
        uint32_t dyn = 0;
        switch (b.kind) {
        case SlotBinding::Kind::Uniform:
        case SlotBinding::Kind::Storage: {
            const BindGroup::Entry* e = group_entry(st, b.group, b.binding, &dyn);
            DkGpuAddr addr = 0;
            uint64_t size = 0;
            if (!e || !e->buffer || !buffer_range(e->buffer, e->offset + dyn, e->size, addr, size)) {
                note_once("a buffer binding without a buffer it knows (skipped)");
                ok = false;
                break;
            }
            if (b.kind == SlotBinding::Kind::Uniform && stage == DkStage_Compute && b.slot >= 6) {
                // deko3d gives compute shaders constant buffers 0-5 only (queue_compute.cpp)
                fail_once("a compute shader with more than 6 uniform buffers");
                ok = false;
                break;
            }
            if (b.kind == SlotBinding::Kind::Uniform) {
                dkCmdBufBindUniformBuffer(R.cmd, stage, b.slot, addr, uint32_t(std::min<uint64_t>(size, kMaxUniformSize)));
            } else {
                dkCmdBufBindStorageBuffer(R.cmd, stage, b.slot, addr, uint32_t(std::min<uint64_t>(size, 0xFFFFFFFFu)));
            }
            break;
        }
        case SlotBinding::Kind::Sampler: {
            const BindGroup::Entry* e = group_entry(st, b.group, b.binding, &dyn);
            Samp* samp = nullptr;
            if (sampler < code.samplers.size() && code.samplers[sampler].group != UINT32_MAX) {
                if (const BindGroup::Entry* s = group_entry(st, code.samplers[sampler].group,
                                                            code.samplers[sampler].binding, &dyn)) {
                    samp = s->sampler;
                }
            }
            sampler++;
            if (!e || !e->view) {
                note_once("a texture binding without a texture it knows (an empty one is bound)");
                ok = false;
            }
            dkCmdBufBindTexture(R.cmd, stage, b.slot, texture_handle(e ? e->view : nullptr, samp));
            break;
        }
        case SlotBinding::Kind::Image: {
            const BindGroup::Entry* e = group_entry(st, b.group, b.binding, &dyn);
            const uint32_t slot = e ? storage_slot(e->view) : 0;
            if (!slot) {
                note_once("a storage texture binding without a texture it knows (skipped)");
                ok = false;
                break;
            }
            dkCmdBufBindImage(R.cmd, stage, b.slot, dkMakeImageHandle(slot));
            break;
        }
        case SlotBinding::Kind::Immediates:
            note_once("immediate data in a mod's shader (left as Aurora's)");
            break;
        }
    }
    if (!ok) count_failure();
    return ok;
}

// the stage is drawable; a Failed shader marks its pipeline (the game moves the mod to Vulkan)
bool usable(ModPipe* p, swdk::ShaderEntry* e) {
    if (!e) return false;
    if (e->state.load(std::memory_order_acquire) == swdk::ShaderState::Failed) {
        if (!p->failed.exchange(true)) fail_once("a mod's shader that uam does not compile");
        return false;
    }
    // (outside the per-frame load budget: a mod's shader was compiled when its pipeline was made)
    if (!swdk::shader_usable(e, true)) return false;
    // v1.1: more scratch memory than the queue has would make deko3d abort (compute) or the GPU fault (graphics)
    const uint32_t limit =
        e->stage == ShaderStage::Compute ? swdk::kNumSms * swdk::kPerWarpScratch : swdk::kPerWarpScratch;
    if (e->scratch > limit) {
        if (!p->failed.exchange(true)) fail_once("a mod's shader that needs more scratch memory than deko3d has");
        return false;
    }
    return true;
}

void bind_shaders(const DkShader* const* shaders, uint32_t count, uint32_t mask) {
    if (swdk::code_take_written()) dkCmdBufBarrier(R.cmd, DkBarrier_None, DkInvalidateFlags_Shader);
    dkCmdBufBindShaders(R.cmd, mask, shaders, count);
}

// a draw or dispatch of a pipeline that will never run counts every time, so the game can tell which mod's
// callback it was (it compares the count around each one)
bool runnable(ModPipe* p, bool compute) {
    if (p && p->compute == compute && !p->failed) return true;
    count_failure();
    return false;
}

bool prepare_draw(ModPass& pass) {
    if (pass.dead) {
        count_failure();
        return false;
    }
    EncState& st = pass.st;
    ModPipe* p = st.pipe;
    if (!runnable(p, false)) return false;
    if (!usable(p, p->vs.entry) || !usable(p, p->fs.entry)) return false;
    const DkShader* sh[] = {&p->vs.entry->shader, &p->fs.entry->shader};
    bind_shaders(sh, 2, DkStageFlag_GraphicsMask);
    dkCmdBufBindRasterizerState(R.cmd, &p->rasterizer);
    dkCmdBufBindColorState(R.cmd, &p->color);
    dkCmdBufBindColorWriteState(R.cmd, &p->colorWrite);
    dkCmdBufBindBlendStates(R.cmd, 0, p->blend, std::max(p->targets, 1u));
    dkCmdBufBindDepthStencilState(R.cmd, &p->depthStencil);
    dkCmdBufBindVtxAttribState(R.cmd, p->attribs.data(), uint32_t(p->attribs.size()));
    dkCmdBufBindVtxBufferState(R.cmd, p->vertexBuffers.data(), uint32_t(p->vertexBuffers.size()));
    for (size_t i = 0; i < p->vertexBuffers.size(); i++) {
        DkGpuAddr addr = 0;
        uint64_t size = 0;
        if (st.vertex[i].buffer && buffer_range(st.vertex[i].buffer, st.vertex[i].offset, st.vertex[i].size, addr, size)) {
            dkCmdBufBindVtxBuffer(R.cmd, uint32_t(i), addr, uint32_t(size));
        }
    }
    bind_stage(DkStage_Vertex, p->vs, st);
    bind_stage(DkStage_Fragment, p->fs, st);
    return true;
}

void set_group(EncState& st, uint32_t index, WGPUBindGroup group, size_t count, const uint32_t* offsets) {
    if (index >= kMaxGroups) return;
    BoundGroup& g = st.groups[index];
    group_addref(group);  // (WebGPU lets the mod release it before the draw)
    group_release(g.group);
    g.group = group;
    g.dynamicCount = uint32_t(std::min<size_t>(count, 16));
    for (uint32_t i = 0; i < g.dynamicCount; i++) g.dynamic[i] = offsets[i];
}

// written memory read back by the CPU, sampled, or read by a later shader: out of the GPU's L2 and the
// shader caches first (as the depth peek's copy)
void barrier_after_writes() {
    dkCmdBufBarrier(R.cmd, DkBarrier_Full, kGpuWritten | DkInvalidateFlags_L2Cache);
}

// ---- writes queued by the game thread (QueueWriteBuffer / QueueWriteTexture), copied at the next op
struct PendingWrite {
    swdk::Block block;
    Buf* buffer = nullptr;  // a buffer write (referenced until done), else a texture write
    uint64_t offset = 0;
    uint32_t size = 0;
    WGPUTexture texture = nullptr;
    uint32_t mip = 0, x = 0, y = 0, z = 0, width = 0, height = 0, layers = 1, pitch = 0, rows = 0;
};
std::mutex g_writeMutex;
std::vector<PendingWrite> g_writes;

void copy_into_buffer(DkGpuAddr src, Buf* b, uint64_t offset, uint32_t size) {
    dkCmdBufBarrier(R.cmd, DkBarrier_Full, 0);
    dkCmdBufCopyBuffer(R.cmd, src, b->block.gpu + offset, size);
    barrier_after_writes();
    b->lastFrame.store(recording_frame(), std::memory_order_relaxed);
}

void run_task(const EncoderTask& task) {
    const auto type = find_runtime_encoder_task_type(task.type);
    if (!type) return;
    auto& res = resources();
    const EncoderTaskContext context{
        .device = webgpu::g_device,
        .queue = webgpu::g_queue,
        .vertexBuffer = res.vertexBuffer,
        .indexBuffer = res.indexBuffer,
        .uniformBuffer = res.uniformBuffer,
        .storageBuffer = res.storageBuffer,
    };
    auto encoder = wgpu::CommandEncoder::Acquire(reinterpret_cast<WGPUCommandEncoder>(&g_taskEncoder));
    t_inCallback = true;
    type->callback(context, encoder, task.payload.data(), task.payloadSize, type->userdata);
    t_inCallback = false;
    (void)encoder.MoveToCHandle();
    if (type->afterSubmit) note_once("an encoder task's after-submit callback (not run with deko3d)");
}

void complete_maps() {
    const uint64_t done = swdk::frames_completed();
    std::vector<WGPUBufferMapCallbackInfo> ready;
    {
        std::unique_lock<std::shared_mutex> lock(g_mutex);
        for (auto& [handle, b] : g_buffers) {
            if (!b->mapPending) continue;
            const uint64_t recorded = dk_frame_of(b->mapIndex);
            if (recorded == 0) continue;  // the frame of the request is not recorded yet
            const uint64_t need = std::max(recorded == UINT64_MAX ? 0 : recorded, b->lastFrame.load());
            if (need > done) continue;
            b->mapPending = false;
            b->mapped = true;
            ready.push_back(b->callback);
        }
    }
    for (const auto& cb : ready) {
        if (cb.callback) cb.callback(WGPUMapAsyncStatus_Success, WGPUStringView{nullptr, 0}, cb.userdata1, cb.userdata2);
    }
}

}  // namespace

void modgfx_frame_begin(const FramePacket& frame) {
    {
        std::lock_guard<std::mutex> lock(g_ringMutex);
        g_ring[frame.frameIndex % kRing] = IndexFrame{frame.frameIndex, recording_frame()};
        g_encodedIndex = frame.frameIndex;
    }
    g_encodedFrame.store(recording_frame(), std::memory_order_relaxed);
    dig_up();
}

void modgfx_process_events() { complete_maps(); }

void modgfx_custom_draw(const CustomDrawCommand& draw, const RenderPass& pass, uint32_t tw, uint32_t th) {
    const auto type = find_runtime_draw_type(draw.type);
    if (!type) return;
    auto& res = resources();
    const DrawContext context{
        .device = webgpu::g_device,
        .queue = webgpu::g_queue,
        .vertexBuffer = res.vertexBuffer,
        .indexBuffer = res.indexBuffer,
        .uniformBuffer = res.uniformBuffer,
        .storageBuffer = res.storageBuffer,
        .layout = pass.target_layout(),
    };
    g_scenePass.st = EncState{};
    g_scenePass.scene = true;
    g_scenePass.width = tw;
    g_scenePass.height = th;
    auto encoder = wgpu::RenderPassEncoder::Acquire(reinterpret_cast<WGPURenderPassEncoder>(&g_scenePass));
    t_inCallback = true;
    type->draw(context, encoder, draw.payload.data(), draw.payloadSize, type->userdata);
    t_inCallback = false;
    (void)encoder.MoveToCHandle();
    release_groups(g_scenePass.st);
    invalidate_state();
}

void modgfx_encoder_task(FramePacket& frame, const EncoderTask& task) {
    (void)frame;
    run_task(task);
    invalidate_state();
}

void modgfx_flush_writes() {
    std::vector<PendingWrite> writes;
    {
        std::lock_guard<std::mutex> lock(g_writeMutex);
        if (g_writes.empty()) return;
        writes.swap(g_writes);
    }
    dkCmdBufBarrier(R.cmd, DkBarrier_Full, DkInvalidateFlags_Image);
    for (const PendingWrite& w : writes) {
        if (w.buffer) {
            copy_into_buffer(w.block.gpu, w.buffer, w.offset, w.size);
            unref(w.buffer);
        } else if (Tex* t = find_texture(w.texture); t && t->valid && w.mip < t->mips) {
            const DkCopyBuf src = {w.block.gpu, w.pitch, w.pitch * w.rows};
            const DkImageView view = level_view(t, w.mip);
            const DkImageRect rect = {w.x, w.y, w.z, w.width, w.height, w.layers};
            dkCmdBufCopyBufferToImage(R.cmd, &src, &view, &rect, 0);
        }
        swdk::block_free_deferred(w.block.block);
    }
    barrier_after_writes();
    invalidate_state();
}

}  // namespace aurora::gfx::dk

// ---- the functions a mod's wgpu* imports are bound to (aurora_dk_wgpu_proc)
using namespace aurora::gfx::dk;

namespace {

// objects whose deko3d shadows Aurora's --wrap'ed creators make (dk_objects.cpp): the plain names here are
// those wrappers (the executable links with -Wl,--wrap=<name>), which forward to Dawn when deko3d is off
WGPUTexture dkw_DeviceCreateTexture(WGPUDevice device, const WGPUTextureDescriptor* d) {
    return wgpuDeviceCreateTexture(device, d);
}
void dkw_TextureAddRef(WGPUTexture t) { wgpuTextureAddRef(t); }
void dkw_TextureRelease(WGPUTexture t) { wgpuTextureRelease(t); }
WGPUTextureView dkw_TextureCreateView(WGPUTexture t, const WGPUTextureViewDescriptor* d) {
    return wgpuTextureCreateView(t, d);
}
void dkw_TextureViewAddRef(WGPUTextureView v) { wgpuTextureViewAddRef(v); }
void dkw_TextureViewRelease(WGPUTextureView v) { wgpuTextureViewRelease(v); }
WGPUSampler dkw_DeviceCreateSampler(WGPUDevice device, const WGPUSamplerDescriptor* d) {
    return wgpuDeviceCreateSampler(device, d);
}
void dkw_SamplerAddRef(WGPUSampler s) { wgpuSamplerAddRef(s); }
void dkw_SamplerRelease(WGPUSampler s) { wgpuSamplerRelease(s); }

WGPUBindGroupLayout dkw_DeviceCreateBindGroupLayout(WGPUDevice device, const WGPUBindGroupLayoutDescriptor* d) {
    WGPUBindGroupLayout l = wgpuDeviceCreateBindGroupLayout(device, d);
    if (!l || !d || !aurora_dk_active()) return l;
    std::vector<uint32_t> dynamic;
    for (size_t i = 0; i < d->entryCount; i++)
        if (d->entries[i].buffer.hasDynamicOffset) dynamic.push_back(d->entries[i].binding);
    std::sort(dynamic.begin(), dynamic.end());
    std::unique_lock<std::shared_mutex> lock(g_mutex);
    g_layoutDynamic[l] = std::move(dynamic);
    return l;
}

WGPUBindGroup dkw_DeviceCreateBindGroup(WGPUDevice device, const WGPUBindGroupDescriptor* d) {
    WGPUBindGroup g = wgpuDeviceCreateBindGroup(device, d);
    if (!g || !d || !aurora_dk_active()) return g;
    Group group;
    std::unique_lock<std::shared_mutex> lock(g_mutex);
    if (auto it = g_layoutDynamic.find(d->layout); it != g_layoutDynamic.end()) group.dynamic = it->second;
    for (size_t i = 0; i < d->entryCount; i++) {
        auto it = g_buffers.find(d->entries[i].buffer);
        if (it == g_buffers.end()) continue;
        it->second->refs.fetch_add(1, std::memory_order_relaxed);
        group.buffers.push_back(it->second);
    }
    if (auto old = g_groups.find(g); old != g_groups.end()) {
        // (a stale entry: Dawn reused the address); its buffers stay referenced, a leak rather than a fault
        old->second = std::move(group);
    } else {
        g_groups.emplace(g, std::move(group));
    }
    return g;
}
void dkw_BindGroupAddRef(WGPUBindGroup g) { group_addref(g); }
void dkw_BindGroupRelease(WGPUBindGroup g) { group_release(g); }

WGPUShaderModule dkw_DeviceCreateShaderModule(WGPUDevice device, const WGPUShaderModuleDescriptor* d) {
    WGPUShaderModule m = wgpuDeviceCreateShaderModule(device, d);
    if (!m || !d || !aurora_dk_active()) return m;
    for (const WGPUChainedStruct* c = d->nextInChain; c; c = c->next) {
        if (c->sType != WGPUSType_ShaderSourceWGSL) continue;
        const auto* src = reinterpret_cast<const WGPUShaderSourceWGSL*>(c);
        std::unique_lock<std::shared_mutex> lock(g_mutex);
        g_modules[m] = Module{1, view_string(src->code)};
    }
    return m;
}
void dkw_ShaderModuleAddRef(WGPUShaderModule m) {
    if (aurora_dk_active()) {
        std::unique_lock<std::shared_mutex> lock(g_mutex);
        if (auto it = g_modules.find(m); it != g_modules.end()) it->second.refs++;
    }
    wgpuShaderModuleAddRef(m);
}
void dkw_ShaderModuleRelease(WGPUShaderModule m) {
    if (aurora_dk_active()) {
        std::unique_lock<std::shared_mutex> lock(g_mutex);
        if (auto it = g_modules.find(m); it != g_modules.end() && --it->second.refs <= 0) g_modules.erase(it);
    }
    wgpuShaderModuleRelease(m);
}

WGPURenderPipeline dkw_DeviceCreateRenderPipeline(WGPUDevice device, const WGPURenderPipelineDescriptor* d) {
    WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(device, d);
    if (!p || !d || !aurora_dk_active()) return p;
    ModPipe* pipe = build_render(*d);
    std::unique_lock<std::shared_mutex> lock(g_mutex);
    g_pipes[p] = pipe;
    return p;
}
WGPUComputePipeline dkw_DeviceCreateComputePipeline(WGPUDevice device, const WGPUComputePipelineDescriptor* d) {
    WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(device, d);
    if (!p || !d || !aurora_dk_active()) return p;
    ModPipe* pipe = build_compute(*d);
    std::unique_lock<std::shared_mutex> lock(g_mutex);
    g_pipes[p] = pipe;
    return p;
}

WGPUBuffer dkw_DeviceCreateBuffer(WGPUDevice device, const WGPUBufferDescriptor* d) {
    WGPUBuffer h = wgpuDeviceCreateBuffer(device, d);
    if (!h || !d || !aurora_dk_active()) return h;
    auto* b = new Buf;
    b->handle = h;
    b->size = d->size;
    b->usage = d->usage;
    const uint64_t bytes = std::max<uint64_t>((d->size + 0xFFFu) & ~uint64_t(0xFFF), 0x1000);
    if (bytes > 0xFFFFFFFFull || !(b->block = swdk::block_create(uint32_t(bytes), "a mod's buffer"))) {
        delete b;
        fail_once("a buffer the deko3d memory cannot hold");
        return h;
    }
    memset(b->block.cpu, 0, bytes);
    b->mapped = b->dawnMapped = d->mappedAtCreation;
    std::unique_lock<std::shared_mutex> lock(g_mutex);
    g_buffers[h] = b;
    return h;
}
void dkw_BufferAddRef(WGPUBuffer h) {
    if (Buf* b = aurora_dk_active() ? find_in(g_buffers, h) : nullptr) b->refs.fetch_add(1, std::memory_order_relaxed);
    wgpuBufferAddRef(h);
}
void dkw_BufferRelease(WGPUBuffer h) {
    if (Buf* b = aurora_dk_active() ? find_in(g_buffers, h) : nullptr) unref(b);
    wgpuBufferRelease(h);
}
void* dkw_BufferGetMappedRange(WGPUBuffer h, size_t offset, size_t size) {
    Buf* b = aurora_dk_active() ? find_in(g_buffers, h) : nullptr;
    if (!b) return wgpuBufferGetMappedRange(h, offset, size);
    std::shared_lock<std::shared_mutex> lock(g_mutex);
    return b->mapped && offset < b->size ? b->block.cpu + offset : nullptr;
}
const void* dkw_BufferGetConstMappedRange(WGPUBuffer h, size_t offset, size_t size) {
    Buf* b = aurora_dk_active() ? find_in(g_buffers, h) : nullptr;
    if (!b) return wgpuBufferGetConstMappedRange(h, offset, size);
    std::shared_lock<std::shared_mutex> lock(g_mutex);
    return b->mapped && offset < b->size ? b->block.cpu + offset : nullptr;
}
void dkw_BufferUnmap(WGPUBuffer h) {
    Buf* b = aurora_dk_active() ? find_in(g_buffers, h) : nullptr;
    if (!b) return wgpuBufferUnmap(h);
    bool dawn = false;
    {
        std::unique_lock<std::shared_mutex> lock(g_mutex);
        b->mapped = false;
        dawn = std::exchange(b->dawnMapped, false);
    }
    if (dawn) wgpuBufferUnmap(h);  // (mappedAtCreation: Dawn's own buffer is mapped too)
}
std::atomic<uint64_t> g_futures{0};
WGPUFuture dkw_BufferMapAsync(WGPUBuffer h, WGPUMapMode mode, size_t offset, size_t size,
                              WGPUBufferMapCallbackInfo callback) {
    Buf* b = aurora_dk_active() ? find_in(g_buffers, h) : nullptr;
    if (!b) return wgpuBufferMapAsync(h, mode, offset, size, callback);
    std::unique_lock<std::shared_mutex> lock(g_mutex);
    b->mapPending = true;
    // the game frame being recorded: the work pushed before the request is in it or an earlier one
    b->mapIndex = aurora::gfx::current_frame();
    b->callback = callback;
    return WGPUFuture{g_futures.fetch_add(1, std::memory_order_relaxed) + 1};
}
void dkw_InstanceProcessEvents(WGPUInstance instance) {
    wgpuInstanceProcessEvents(instance);
    if (aurora_dk_active()) complete_maps();
}
void dkw_QueueWriteBuffer(WGPUQueue queue, WGPUBuffer h, uint64_t offset, const void* data, size_t size) {
    Buf* b = aurora_dk_active() ? find_in(g_buffers, h) : nullptr;
    if (!b) return wgpuQueueWriteBuffer(queue, h, offset, data, size);
    if (offset >= b->size || size == 0) return;
    const uint32_t n = uint32_t(std::min<uint64_t>({size, b->size - offset, 0xFFFFFFFFu}));
    if (t_inCallback) {
        // inside a callback: in the command stream at this point, before the draws that follow
        if (const swdk::StreamAlloc s = swdk::stream_alloc(n, 256)) {
            memcpy(s.cpu, data, n);
            copy_into_buffer(s.gpu, b, offset, n);
            return;
        }
    }
    PendingWrite w;
    w.block = swdk::block_create(std::max<uint32_t>((n + 255u) & ~255u, 256), "a mod's buffer write");
    if (!w.block) return;
    memcpy(w.block.cpu, data, n);
    w.buffer = b;
    b->refs.fetch_add(1, std::memory_order_relaxed);
    w.offset = offset;
    w.size = n;
    std::lock_guard<std::mutex> lock(g_writeMutex);
    g_writes.push_back(w);
}
void dkw_QueueWriteTexture(WGPUQueue queue, const WGPUTexelCopyTextureInfo* dst, const void* data, size_t dataSize,
                           const WGPUTexelCopyBufferLayout* layout, const WGPUExtent3D* size) {
    Tex* t = aurora_dk_active() ? find_texture(dst->texture) : nullptr;
    if (!t) return wgpuQueueWriteTexture(queue, dst, data, dataSize, layout, size);
    const FormatInfo f = format_info(t->wformat);
    // compressed formats: rows of blocks
    const uint32_t blocksW = (size->width + f.blockW - 1) / f.blockW;
    const uint32_t blocksH = (size->height + f.blockH - 1) / f.blockH;
    const uint32_t rowBytes = blocksW * f.blockBytes;
    const uint32_t srcPitch = layout->bytesPerRow != WGPU_COPY_STRIDE_UNDEFINED ? layout->bytesPerRow : rowBytes;
    const uint32_t srcRows = layout->rowsPerImage != WGPU_COPY_STRIDE_UNDEFINED ? layout->rowsPerImage : blocksH;
    const uint32_t layers = std::max<uint32_t>(size->depthOrArrayLayers, 1);
    PendingWrite w;
    w.pitch = (rowBytes + 255u) & ~255u;
    w.rows = blocksH;
    w.block = swdk::block_create(std::max<uint32_t>(w.pitch * blocksH * layers, 256), "a mod's texture write");
    if (!w.block) return;
    for (uint32_t l = 0; l < layers; l++) {
        for (uint32_t r = 0; r < blocksH; r++) {
            const uint64_t from = layout->offset + (uint64_t(l) * srcRows + r) * srcPitch;
            if (from + rowBytes > dataSize) break;
            memcpy(w.block.cpu + (size_t(l) * blocksH + r) * w.pitch, static_cast<const uint8_t*>(data) + from, rowBytes);
        }
    }
    w.texture = dst->texture;
    w.mip = dst->mipLevel;
    // dkCmdBufCopyBufferToImage takes compressed rectangles in blocks (as dk_encode's copies)
    w.x = dst->origin.x / f.blockW;
    w.y = dst->origin.y / f.blockH;
    w.z = dst->origin.z;
    w.width = f.compressed ? blocksW : size->width;
    w.height = f.compressed ? blocksH : size->height;
    w.layers = layers;
    std::lock_guard<std::mutex> lock(g_writeMutex);
    g_writes.push_back(w);
}
void dkw_QueueSubmit(WGPUQueue queue, size_t count, const WGPUCommandBuffer* buffers) {
    // a mod's own command encoder recorded into Dawn's Null device: nothing of it would reach the GPU
    if (aurora_dk_active() && count) fail_once("a mod's own command buffer (work outside the gfx service's callbacks)");
    wgpuQueueSubmit(queue, count, buffers);
}

// -- render passes
void dkw_RenderPassEncoderSetPipeline(WGPURenderPassEncoder p, WGPURenderPipeline pipeline) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderSetPipeline(p, pipeline);
    m->st.pipe = find_in(g_pipes, pipeline);
}
void dkw_RenderPassEncoderSetBindGroup(WGPURenderPassEncoder p, uint32_t index, WGPUBindGroup group, size_t count,
                                       const uint32_t* offsets) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderSetBindGroup(p, index, group, count, offsets);
    set_group(m->st, index, group, count, offsets);
}
void dkw_RenderPassEncoderSetVertexBuffer(WGPURenderPassEncoder p, uint32_t slot, WGPUBuffer buffer, uint64_t offset,
                                          uint64_t size) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderSetVertexBuffer(p, slot, buffer, offset, size);
    if (slot < 16) m->st.vertex[slot] = {buffer, offset, size};
}
void dkw_RenderPassEncoderSetIndexBuffer(WGPURenderPassEncoder p, WGPUBuffer buffer, WGPUIndexFormat format,
                                         uint64_t offset, uint64_t size) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderSetIndexBuffer(p, buffer, format, offset, size);
    m->st.index = {buffer, offset, size};
    m->st.indexFormat = format == WGPUIndexFormat_Uint32 ? DkIdxFormat_Uint32 : DkIdxFormat_Uint16;
}
void dkw_RenderPassEncoderSetViewport(WGPURenderPassEncoder p, float x, float y, float w, float h, float minDepth,
                                      float maxDepth) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderSetViewport(p, x, y, w, h, minDepth, maxDepth);
    if (!(w > 0.f) || !(h > 0.f) || m->dead) return;
    const DkViewport v = {x, y, w, h, minDepth, maxDepth};
    dkCmdBufSetViewports(R.cmd, 0, &v, 1);
}
void dkw_RenderPassEncoderSetScissorRect(WGPURenderPassEncoder p, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderSetScissorRect(p, x, y, w, h);
    if (m->dead) return;
    x = std::min(x, m->width);
    y = std::min(y, m->height);
    const DkScissor s = {x, y, std::min(w, m->width - x), std::min(h, m->height - y)};
    dkCmdBufSetScissors(R.cmd, 0, &s, 1);
}
void dkw_RenderPassEncoderSetBlendConstant(WGPURenderPassEncoder p, const WGPUColor* c) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderSetBlendConstant(p, c);
    dkCmdBufSetBlendConst(R.cmd, float(c->r), float(c->g), float(c->b), float(c->a));
}
void dkw_RenderPassEncoderSetStencilReference(WGPURenderPassEncoder p, uint32_t reference) {
    // (mods' pipelines get no stencil test here, so the reference has nothing to feed)
    if (!as_pass(p)) wgpuRenderPassEncoderSetStencilReference(p, reference);
}
void dkw_RenderPassEncoderDraw(WGPURenderPassEncoder p, uint32_t vertices, uint32_t instances, uint32_t firstVertex,
                               uint32_t firstInstance) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderDraw(p, vertices, instances, firstVertex, firstInstance);
    if (!prepare_draw(*m)) return;
    dkCmdBufDraw(R.cmd, m->st.pipe->primitive, vertices, instances, firstVertex, firstInstance);
    encode_stats().draws++;
}
void dkw_RenderPassEncoderDrawIndexed(WGPURenderPassEncoder p, uint32_t indices, uint32_t instances,
                                      uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderDrawIndexed(p, indices, instances, firstIndex, baseVertex, firstInstance);
    DkGpuAddr addr = 0;
    uint64_t size = 0;
    if (!m->st.index.buffer || !buffer_range(m->st.index.buffer, m->st.index.offset, m->st.index.size, addr, size)) {
        count_failure();
        return;
    }
    if (!prepare_draw(*m)) return;
    dkCmdBufBindIdxBuffer(R.cmd, m->st.indexFormat, addr);
    dkCmdBufDrawIndexed(R.cmd, m->st.pipe->primitive, indices, instances, firstIndex, baseVertex, firstInstance);
    encode_stats().draws++;
}
void dkw_RenderPassEncoderEnd(WGPURenderPassEncoder p) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderEnd(p);
    if (m->scene || m->ended) return;  // (the scene pass is Aurora's)
    m->ended = true;
    release_groups(m->st);
    if (!m->dead) dkCmdBufBarrier(R.cmd, DkBarrier_Fragments, kGpuWritten);
}
void dkw_RenderPassEncoderAddRef(WGPURenderPassEncoder p) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderAddRef(p);
    m->refs.fetch_add(1, std::memory_order_relaxed);
}
void dkw_RenderPassEncoderRelease(WGPURenderPassEncoder p) {
    ModPass* m = as_pass(p);
    if (!m) return wgpuRenderPassEncoderRelease(p);
    if (m->scene || m->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    release_groups(m->st);
    delete m;
}

// -- compute passes
void dkw_ComputePassEncoderSetPipeline(WGPUComputePassEncoder p, WGPUComputePipeline pipeline) {
    ModCompute* m = as_compute(p);
    if (!m) return wgpuComputePassEncoderSetPipeline(p, pipeline);
    m->st.pipe = find_in(g_pipes, pipeline);
}
void dkw_ComputePassEncoderSetBindGroup(WGPUComputePassEncoder p, uint32_t index, WGPUBindGroup group, size_t count,
                                        const uint32_t* offsets) {
    ModCompute* m = as_compute(p);
    if (!m) return wgpuComputePassEncoderSetBindGroup(p, index, group, count, offsets);
    set_group(m->st, index, group, count, offsets);
}
void dkw_ComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder p, uint32_t x, uint32_t y, uint32_t z) {
    ModCompute* m = as_compute(p);
    if (!m) return wgpuComputePassEncoderDispatchWorkgroups(p, x, y, z);
    ModPipe* pipe = m->st.pipe;
    if (!runnable(pipe, true) || !usable(pipe, pipe->cs.entry)) return;
    if (x == 0 || y == 0 || z == 0) return;
    const DkShader* sh[] = {&pipe->cs.entry->shader};
    bind_shaders(sh, 1, DkStageFlag_Compute);
    bind_stage(DkStage_Compute, pipe->cs, m->st);
    dkCmdBufDispatchCompute(R.cmd, x, y, z);
    // WebGPU: each dispatch sees the writes of the ones before it
    dkCmdBufBarrier(R.cmd, DkBarrier_Full, kGpuWritten);
}
void dkw_ComputePassEncoderEnd(WGPUComputePassEncoder p) {
    ModCompute* m = as_compute(p);
    if (!m) return wgpuComputePassEncoderEnd(p);
    if (m->ended) return;
    m->ended = true;
    release_groups(m->st);
    barrier_after_writes();
}
void dkw_ComputePassEncoderAddRef(WGPUComputePassEncoder p) {
    ModCompute* m = as_compute(p);
    if (!m) return wgpuComputePassEncoderAddRef(p);
    m->refs.fetch_add(1, std::memory_order_relaxed);
}
void dkw_ComputePassEncoderRelease(WGPUComputePassEncoder p) {
    ModCompute* m = as_compute(p);
    if (!m) return wgpuComputePassEncoderRelease(p);
    if (m->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    release_groups(m->st);
    delete m;
}

// -- the task's command encoder
WGPUComputePassEncoder dkw_CommandEncoderBeginComputePass(WGPUCommandEncoder e, const WGPUComputePassDescriptor* d) {
    if (!as_encoder(e)) return wgpuCommandEncoderBeginComputePass(e, d);
    return reinterpret_cast<WGPUComputePassEncoder>(new ModCompute);
}
WGPURenderPassEncoder dkw_CommandEncoderBeginRenderPass(WGPUCommandEncoder e, const WGPURenderPassDescriptor* d) {
    if (!as_encoder(e)) return wgpuCommandEncoderBeginRenderPass(e, d);
    auto* m = new ModPass;
    DkImageView colors[8];
    const DkImageView* colorPtrs[8];
    uint32_t count = 0;
    for (size_t i = 0; i < d->colorAttachmentCount && i < 8; i++) {
        const WGPURenderPassColorAttachment& a = d->colorAttachments[i];
        View* v = find_view(a.view);
        if (!v || !v->tex->valid) {
            fail_once("a mod's render pass whose colour target has no deko3d image");
            m->dead = true;  // (nothing is bound: its draws must not land in the previous targets)
            return reinterpret_cast<WGPURenderPassEncoder>(m);
        }
        if (a.resolveTarget) note_once("a mod's render pass resolve target (ignored: passes are single-sampled)");
        colors[count] = target_of(v);
        colorPtrs[count] = &colors[count];
        if (count == 0) {
            m->width = std::max(v->tex->width >> v->baseMip, 1u);
            m->height = std::max(v->tex->height >> v->baseMip, 1u);
        }
        count++;
    }
    DkImageView depth;
    const DkImageView* depthPtr = nullptr;
    if (d->depthStencilAttachment) {
        if (View* v = find_view(d->depthStencilAttachment->view); v && v->tex->valid) {
            depth = target_of(v);
            depthPtr = &depth;
            if (count == 0) {
                m->width = std::max(v->tex->width >> v->baseMip, 1u);
                m->height = std::max(v->tex->height >> v->baseMip, 1u);
            }
        }
    }
    if (count == 0 && !depthPtr) {
        fail_once("a mod's render pass with no target deko3d knows");
        m->dead = true;
        return reinterpret_cast<WGPURenderPassEncoder>(m);
    }
    dkCmdBufBindRenderTargets(R.cmd, colorPtrs, count, depthPtr);
    const DkViewportSwizzle sw = {DkSwizzle_PositiveX, swdk::C.flipY ? DkSwizzle_NegativeY : DkSwizzle_PositiveY,
                                  DkSwizzle_PositiveZ, DkSwizzle_PositiveW};
    dkCmdBufSetViewportSwizzles(R.cmd, 0, &sw, 1);
    const DkViewport v = {0.f, 0.f, float(m->width), float(m->height), 0.f, 1.f};
    dkCmdBufSetViewports(R.cmd, 0, &v, 1);
    const DkScissor s = {0, 0, m->width, m->height};
    dkCmdBufSetScissors(R.cmd, 0, &s, 1);
    dkCmdBufSetBlendConst(R.cmd, 0.f, 0.f, 0.f, 0.f);
    for (uint32_t i = 0; i < count; i++) {
        const WGPURenderPassColorAttachment& a = d->colorAttachments[i];
        if (a.loadOp == WGPULoadOp_Clear) {
            dkCmdBufClearColorFloat(R.cmd, i, DkColorMask_RGBA, float(a.clearValue.r), float(a.clearValue.g),
                                    float(a.clearValue.b), float(a.clearValue.a));
        }
    }
    if (depthPtr) {
        const auto& a = *d->depthStencilAttachment;
        const bool clearDepth = a.depthLoadOp == WGPULoadOp_Clear;
        const bool clearStencil = a.stencilLoadOp == WGPULoadOp_Clear;
        if (clearDepth || clearStencil) {
            dkCmdBufClearDepthStencil(R.cmd, clearDepth, a.depthClearValue, clearStencil ? 0xFF : 0,
                                      uint8_t(a.stencilClearValue));
        }
    }
    return reinterpret_cast<WGPURenderPassEncoder>(m);
}
void dkw_CommandEncoderCopyBufferToBuffer(WGPUCommandEncoder e, WGPUBuffer src, uint64_t srcOffset, WGPUBuffer dst,
                                          uint64_t dstOffset, uint64_t size) {
    if (!as_encoder(e)) return wgpuCommandEncoderCopyBufferToBuffer(e, src, srcOffset, dst, dstOffset, size);
    DkGpuAddr from = 0, to = 0;
    uint64_t fromSize = 0, toSize = 0;
    if (!buffer_range(src, srcOffset, size, from, fromSize) || !buffer_range(dst, dstOffset, size, to, toSize)) {
        fail_once("a buffer copy between buffers it does not know");
        return;
    }
    dkCmdBufBarrier(R.cmd, DkBarrier_Full, 0);
    dkCmdBufCopyBuffer(R.cmd, from, to, uint32_t(std::min({fromSize, toSize, uint64_t(0xFFFFFFFFu)})));
    barrier_after_writes();
    encode_stats().copies++;
}
void dkw_CommandEncoderCopyTextureToTexture(WGPUCommandEncoder e, const WGPUTexelCopyTextureInfo* src,
                                            const WGPUTexelCopyTextureInfo* dst, const WGPUExtent3D* size) {
    if (!as_encoder(e)) return wgpuCommandEncoderCopyTextureToTexture(e, src, dst, size);
    Tex* s = find_texture(src->texture);
    Tex* d = find_texture(dst->texture);
    if (!s || !d || !s->valid || !d->valid || format_info(s->wformat).blockBytes != format_info(d->wformat).blockBytes ||
        src->mipLevel >= s->mips || dst->mipLevel >= d->mips) {
        fail_once("a texture copy deko3d cannot do (unknown textures, levels or different texel sizes)");
        return;
    }
    uint32_t w = size->width, h = size->height;
    uint32_t sx = src->origin.x, sy = src->origin.y, dx = dst->origin.x, dy = dst->origin.y;
    if (s->compressed) {
        // in blocks (SwitchWakerHD surfaces.cpp, as dk_encode's copy_image)
        w = (w + s->blockW - 1) / s->blockW;
        h = (h + s->blockH - 1) / s->blockH;
        sx /= s->blockW;
        sy /= s->blockH;
        dx /= d->blockW;
        dy /= d->blockH;
    }
    const uint32_t layers = std::max<uint32_t>(size->depthOrArrayLayers, 1);
    const DkImageView sv = level_view(s, src->mipLevel), dv = level_view(d, dst->mipLevel);
    const DkImageRect sr = {sx, sy, src->origin.z, w, h, layers}, dr = {dx, dy, dst->origin.z, w, h, layers};
    dkCmdBufBarrier(R.cmd, DkBarrier_Full, 0);
    dkCmdBufCopyImage(R.cmd, &sv, &sr, &dv, &dr, 0);
    barrier_after_writes();
    encode_stats().copies++;
}
void dkw_CommandEncoderAddRef(WGPUCommandEncoder e) {
    if (!as_encoder(e)) wgpuCommandEncoderAddRef(e);
}
void dkw_CommandEncoderRelease(WGPUCommandEncoder e) {
    if (!as_encoder(e)) wgpuCommandEncoderRelease(e);
}
WGPUCommandBuffer dkw_CommandEncoderFinish(WGPUCommandEncoder e, const WGPUCommandBufferDescriptor* d) {
    if (!as_encoder(e)) return wgpuCommandEncoderFinish(e, d);
    fail_once("wgpuCommandEncoderFinish on the frame's encoder (the gfx service says never)");
    return nullptr;
}

// The rest of the pass and encoder functions, so that none of Dawn's ever gets our handles: debug groups,
// markers and labels do nothing on ours; the others have no deko3d version here and count as failures.
// Real Dawn handles go to Dawn.
#define DKW_IGNORED(name, check, params, args) \
    void dkw_##name params {                   \
        if (check) return;                     \
        wgpu##name args;                       \
    }
#define DKW_FAILED(name, check, params, args)                        \
    void dkw_##name params {                                         \
        if (check) return fail_once("wgpu" #name " with deko3d");    \
        wgpu##name args;                                             \
    }
DKW_IGNORED(RenderPassEncoderInsertDebugMarker, as_pass(p), (WGPURenderPassEncoder p, WGPUStringView s), (p, s))
DKW_IGNORED(RenderPassEncoderPushDebugGroup, as_pass(p), (WGPURenderPassEncoder p, WGPUStringView s), (p, s))
DKW_IGNORED(RenderPassEncoderPopDebugGroup, as_pass(p), (WGPURenderPassEncoder p), (p))
DKW_IGNORED(RenderPassEncoderSetLabel, as_pass(p), (WGPURenderPassEncoder p, WGPUStringView s), (p, s))
DKW_IGNORED(ComputePassEncoderInsertDebugMarker, as_compute(p), (WGPUComputePassEncoder p, WGPUStringView s), (p, s))
DKW_IGNORED(ComputePassEncoderPushDebugGroup, as_compute(p), (WGPUComputePassEncoder p, WGPUStringView s), (p, s))
DKW_IGNORED(ComputePassEncoderPopDebugGroup, as_compute(p), (WGPUComputePassEncoder p), (p))
DKW_IGNORED(ComputePassEncoderSetLabel, as_compute(p), (WGPUComputePassEncoder p, WGPUStringView s), (p, s))
DKW_IGNORED(CommandEncoderInsertDebugMarker, as_encoder(e), (WGPUCommandEncoder e, WGPUStringView s), (e, s))
DKW_IGNORED(CommandEncoderPushDebugGroup, as_encoder(e), (WGPUCommandEncoder e, WGPUStringView s), (e, s))
DKW_IGNORED(CommandEncoderPopDebugGroup, as_encoder(e), (WGPUCommandEncoder e), (e))
DKW_IGNORED(CommandEncoderSetLabel, as_encoder(e), (WGPUCommandEncoder e, WGPUStringView s), (e, s))
DKW_IGNORED(CommandEncoderInjectValidationError, as_encoder(e), (WGPUCommandEncoder e, WGPUStringView s), (e, s))
DKW_FAILED(RenderPassEncoderBeginOcclusionQuery, as_pass(p), (WGPURenderPassEncoder p, uint32_t i), (p, i))
DKW_FAILED(RenderPassEncoderEndOcclusionQuery, as_pass(p), (WGPURenderPassEncoder p), (p))
DKW_FAILED(RenderPassEncoderDrawIndirect, as_pass(p), (WGPURenderPassEncoder p, WGPUBuffer b, uint64_t o), (p, b, o))
DKW_FAILED(RenderPassEncoderDrawIndexedIndirect, as_pass(p), (WGPURenderPassEncoder p, WGPUBuffer b, uint64_t o),
           (p, b, o))
DKW_FAILED(RenderPassEncoderExecuteBundles, as_pass(p), (WGPURenderPassEncoder p, size_t n, const WGPURenderBundle* b),
           (p, n, b))
DKW_FAILED(RenderPassEncoderMultiDrawIndirect, as_pass(p),
           (WGPURenderPassEncoder p, WGPUBuffer b, uint64_t o, uint32_t n, WGPUBuffer c, uint64_t co), (p, b, o, n, c, co))
DKW_FAILED(RenderPassEncoderMultiDrawIndexedIndirect, as_pass(p),
           (WGPURenderPassEncoder p, WGPUBuffer b, uint64_t o, uint32_t n, WGPUBuffer c, uint64_t co), (p, b, o, n, c, co))
DKW_FAILED(RenderPassEncoderPixelLocalStorageBarrier, as_pass(p), (WGPURenderPassEncoder p), (p))
DKW_FAILED(RenderPassEncoderSetImmediates, as_pass(p), (WGPURenderPassEncoder p, uint32_t o, const void* d, size_t n),
           (p, o, d, n))
DKW_FAILED(RenderPassEncoderSetResourceTable, as_pass(p), (WGPURenderPassEncoder p, WGPUResourceTable t), (p, t))
DKW_FAILED(RenderPassEncoderWriteTimestamp, as_pass(p), (WGPURenderPassEncoder p, WGPUQuerySet q, uint32_t i), (p, q, i))
DKW_FAILED(ComputePassEncoderDispatchWorkgroupsIndirect, as_compute(p),
           (WGPUComputePassEncoder p, WGPUBuffer b, uint64_t o), (p, b, o))
DKW_FAILED(ComputePassEncoderSetImmediates, as_compute(p),
           (WGPUComputePassEncoder p, uint32_t o, const void* d, size_t n), (p, o, d, n))
DKW_FAILED(ComputePassEncoderSetResourceTable, as_compute(p), (WGPUComputePassEncoder p, WGPUResourceTable t), (p, t))
DKW_FAILED(ComputePassEncoderWriteTimestamp, as_compute(p), (WGPUComputePassEncoder p, WGPUQuerySet q, uint32_t i),
           (p, q, i))
DKW_FAILED(CommandEncoderClearBuffer, as_encoder(e), (WGPUCommandEncoder e, WGPUBuffer b, uint64_t o, uint64_t n),
           (e, b, o, n))
DKW_FAILED(CommandEncoderCopyBufferToTexture, as_encoder(e),
           (WGPUCommandEncoder e, const WGPUTexelCopyBufferInfo* s, const WGPUTexelCopyTextureInfo* d, const WGPUExtent3D* n),
           (e, s, d, n))
DKW_FAILED(CommandEncoderCopyTextureToBuffer, as_encoder(e),
           (WGPUCommandEncoder e, const WGPUTexelCopyTextureInfo* s, const WGPUTexelCopyBufferInfo* d, const WGPUExtent3D* n),
           (e, s, d, n))
DKW_FAILED(CommandEncoderResolveQuerySet, as_encoder(e),
           (WGPUCommandEncoder e, WGPUQuerySet q, uint32_t f, uint32_t n, WGPUBuffer b, uint64_t o), (e, q, f, n, b, o))
DKW_FAILED(CommandEncoderWriteBuffer, as_encoder(e),
           (WGPUCommandEncoder e, WGPUBuffer b, uint64_t o, const void* d, size_t n), (e, b, o, d, n))
DKW_FAILED(CommandEncoderWriteTimestamp, as_encoder(e), (WGPUCommandEncoder e, WGPUQuerySet q, uint32_t i), (e, q, i))
#undef DKW_IGNORED
#undef DKW_FAILED

struct Proc {
    const char* name;
    void* fn;
};
#define DKW(name) {"wgpu" #name, reinterpret_cast<void*>(&dkw_##name)}
const Proc kProcs[] = {
    // objects
    DKW(DeviceCreateTexture), DKW(TextureAddRef), DKW(TextureRelease), DKW(TextureCreateView),
    DKW(TextureViewAddRef), DKW(TextureViewRelease), DKW(DeviceCreateSampler), DKW(SamplerAddRef),
    DKW(SamplerRelease), DKW(DeviceCreateBindGroupLayout), DKW(DeviceCreateBindGroup), DKW(BindGroupAddRef),
    DKW(BindGroupRelease), DKW(DeviceCreateShaderModule), DKW(ShaderModuleAddRef), DKW(ShaderModuleRelease),
    DKW(DeviceCreateRenderPipeline), DKW(DeviceCreateComputePipeline), DKW(DeviceCreateBuffer), DKW(BufferAddRef),
    DKW(BufferRelease), DKW(BufferGetMappedRange), DKW(BufferGetConstMappedRange), DKW(BufferUnmap),
    DKW(BufferMapAsync), DKW(InstanceProcessEvents), DKW(QueueWriteBuffer), DKW(QueueWriteTexture), DKW(QueueSubmit),
    // render passes
    DKW(RenderPassEncoderSetPipeline), DKW(RenderPassEncoderSetBindGroup), DKW(RenderPassEncoderSetVertexBuffer),
    DKW(RenderPassEncoderSetIndexBuffer), DKW(RenderPassEncoderSetViewport), DKW(RenderPassEncoderSetScissorRect),
    DKW(RenderPassEncoderSetBlendConstant), DKW(RenderPassEncoderSetStencilReference), DKW(RenderPassEncoderDraw),
    DKW(RenderPassEncoderDrawIndexed), DKW(RenderPassEncoderEnd), DKW(RenderPassEncoderAddRef),
    DKW(RenderPassEncoderRelease), DKW(RenderPassEncoderInsertDebugMarker), DKW(RenderPassEncoderPushDebugGroup),
    DKW(RenderPassEncoderPopDebugGroup), DKW(RenderPassEncoderSetLabel), DKW(RenderPassEncoderBeginOcclusionQuery),
    DKW(RenderPassEncoderEndOcclusionQuery), DKW(RenderPassEncoderDrawIndirect),
    DKW(RenderPassEncoderDrawIndexedIndirect), DKW(RenderPassEncoderExecuteBundles),
    DKW(RenderPassEncoderMultiDrawIndirect), DKW(RenderPassEncoderMultiDrawIndexedIndirect),
    DKW(RenderPassEncoderPixelLocalStorageBarrier), DKW(RenderPassEncoderSetImmediates),
    DKW(RenderPassEncoderSetResourceTable), DKW(RenderPassEncoderWriteTimestamp),
    // compute passes
    DKW(ComputePassEncoderSetPipeline), DKW(ComputePassEncoderSetBindGroup),
    DKW(ComputePassEncoderDispatchWorkgroups), DKW(ComputePassEncoderEnd), DKW(ComputePassEncoderAddRef),
    DKW(ComputePassEncoderRelease), DKW(ComputePassEncoderInsertDebugMarker), DKW(ComputePassEncoderPushDebugGroup),
    DKW(ComputePassEncoderPopDebugGroup), DKW(ComputePassEncoderSetLabel),
    DKW(ComputePassEncoderDispatchWorkgroupsIndirect), DKW(ComputePassEncoderSetImmediates),
    DKW(ComputePassEncoderSetResourceTable), DKW(ComputePassEncoderWriteTimestamp),
    // command encoders
    DKW(CommandEncoderBeginComputePass), DKW(CommandEncoderBeginRenderPass), DKW(CommandEncoderCopyBufferToBuffer),
    DKW(CommandEncoderCopyTextureToTexture), DKW(CommandEncoderAddRef), DKW(CommandEncoderRelease),
    DKW(CommandEncoderFinish), DKW(CommandEncoderInsertDebugMarker), DKW(CommandEncoderPushDebugGroup),
    DKW(CommandEncoderPopDebugGroup), DKW(CommandEncoderSetLabel), DKW(CommandEncoderInjectValidationError),
    DKW(CommandEncoderClearBuffer), DKW(CommandEncoderCopyBufferToTexture), DKW(CommandEncoderCopyTextureToBuffer),
    DKW(CommandEncoderResolveQuerySet), DKW(CommandEncoderWriteBuffer), DKW(CommandEncoderWriteTimestamp),
};
#undef DKW

}  // namespace

extern "C" {

const char* aurora_dk_wgpu_proc_name(size_t index) {
    return index < sizeof kProcs / sizeof kProcs[0] ? kProcs[index].name : nullptr;
}

void* aurora_dk_wgpu_proc(const char* name) {
    if (!name) return nullptr;
    for (const Proc& p : kProcs)
        if (strcmp(p.name, name) == 0) return p.fn;
    return nullptr;
}

uint32_t aurora_dk_modgfx_failures(void) { return t_failures; }

bool aurora_dk_active(void) { return aurora::gfx::dk::active(); }

}  // extern "C"
