# aleks2 (deko3d P1): the experimental deko3d renderer, selected at run time (BACKEND_DEKO3D).
#
# Aurora keeps recording every frame as for WebGPU, with Dawn's Null device as its object model, and
# lib/deko turns the recorded frames into deko3d commands (the design of SwitchWaker 084291f, MIT;
# lib/deko/README.md). With the Dawn backend nothing here runs: the --wrap'ed wgpu* functions forward
# to Dawn, and no deko3d memory is reserved.
#
#   lib/deko/      MIT: Aurora's side (encoder, object shadows, pipelines, present, RmlUi), shader
#                  cache and translation (adapted from SwitchWaker switch/deko)
#   lib/deko/mpl/  MPL-2.0: the deko3d platform layer of SwitchWakerHD via SwitchWaker (device,
#                  memory, descriptors, ImGui overlay, test pattern, uam worker, DKSH files);
#                  their headers are kept and their diffs from 084291f are the change notes
#
# Third party, from AURORA_DEKO3D_SW_DIR (a SwitchWaker checkout at 084291f): deko3d 0.5.0 and
# uam 1.1.0 with their PATCHES.md (zlib; uam's mesa-imported is MIT), and the two Tint options of
# switch/dawn/patches (applied to this build's Dawn tree below).
if (NOT AURORA_GFX_DEKO3D)
    return()
endif ()
if (NOT EXISTS "${AURORA_DEKO3D_SW_DIR}/switch/deko3d/CMakeLists.txt")
    message(FATAL_ERROR "AURORA_GFX_DEKO3D: AURORA_DEKO3D_SW_DIR (${AURORA_DEKO3D_SW_DIR}) is not a SwitchWaker checkout")
endif ()
if (NOT TARGET tint_lang_glsl_writer)
    message(FATAL_ERROR "AURORA_GFX_DEKO3D: Dawn was configured without Tint's GLSL writer (TINT_BUILD_GLSL_WRITER)")
endif ()

set(_dk_lib_dir "${CMAKE_CURRENT_LIST_DIR}/../lib/deko")
include(FetchContent)
FetchContent_GetProperties(dawn)
if (NOT dawn_SOURCE_DIR)
    message(FATAL_ERROR "AURORA_GFX_DEKO3D: Dawn's source tree is unknown (FetchContent dawn)")
endif ()
set(_dk_sw "${AURORA_DEKO3D_SW_DIR}/switch")

# --- Tint: gl_Position.z and .y written as WebGPU's (deko3d's device has the same conventions) ---
include("${CMAKE_SOURCE_DIR}/cmake/SwitchPatches.cmake" OPTIONAL RESULT_VARIABLE _dk_have_patcher)
if (NOT _dk_have_patcher)
    message(FATAL_ERROR "AURORA_GFX_DEKO3D: SwitchPatches.cmake (dusk_switch_apply_patch) not found")
endif ()
set(_dk_tint_opts "${dawn_SOURCE_DIR}/src/tint/lang/glsl/writer/common/options.h")
file(READ "${_dk_tint_opts}" _dk_text)
if (NOT _dk_text MATCHES "depth_zero_to_one")
    dusk_switch_apply_patch("${dawn_SOURCE_DIR}" "${_dk_sw}/dawn/patches/dawn-switch-tint-depth-zero-to-one.patch"
            src/tint/lang/glsl/writer/common/options.h "depth_zero_to_one")
endif ()
file(READ "${_dk_tint_opts}" _dk_text)
if (NOT _dk_text MATCHES "disable_position_y_negation")
    dusk_switch_apply_patch("${dawn_SOURCE_DIR}" "${_dk_sw}/dawn/patches/dawn-switch-tint-position-y-up.patch"
            src/tint/lang/glsl/writer/common/options.h "disable_position_y_negation")
endif ()

# --- deko3d 0.5.0 and uam 1.1.0 (built for the console) ---
set(NX_ROOT "${DEVKITPRO}/libnx")  # uam's CMakeLists: the console build (no host threads, no tool)
add_subdirectory("${_dk_sw}/deko3d" "${CMAKE_BINARY_DIR}/aurora_deko3d/deko3d" EXCLUDE_FROM_ALL)
add_subdirectory("${_dk_sw}/uam" "${CMAKE_BINARY_DIR}/aurora_deko3d/uam" EXCLUDE_FROM_ALL)
# (the Switch build defines SWITCH everywhere; it is a token of uam's GLSL parser)
target_compile_options(uamlib PRIVATE -USWITCH)
# uam's Mesa and the NVK driver's Mesa (libvulkan.a) share symbol names: the NRO links a copy of uamlib
# whose clashing symbols are renamed (aurora_deko3d_uam_rename.cmake), aurora_uam.
if (NOT NAGA_VULKAN_ROOT OR NOT EXISTS "${NAGA_VULKAN_ROOT}/lib/libvulkan.a")
    message(FATAL_ERROR "AURORA_GFX_DEKO3D: NAGA_VULKAN_ROOT/lib/libvulkan.a (NVK's Mesa) is needed to rename uam's")
endif ()
set(_dk_uam_renamed "${CMAKE_BINARY_DIR}/aurora_deko3d/libuam_renamed.a")
add_custom_command(OUTPUT "${_dk_uam_renamed}"
        COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:uamlib>" "${_dk_uam_renamed}"
        COMMAND "${CMAKE_COMMAND}" "-DUAM=${_dk_uam_renamed}" "-DOTHER=${NAGA_VULKAN_ROOT}/lib/libvulkan.a"
                "-DNM=${CMAKE_NM}" "-DOBJCOPY=${CMAKE_OBJCOPY}" "-DOUT=${CMAKE_BINARY_DIR}/aurora_deko3d/uam_renamed.txt"
                -P "${CMAKE_CURRENT_LIST_DIR}/aurora_deko3d_uam_rename.cmake"
        DEPENDS uamlib "$<TARGET_FILE:uamlib>" "${CMAKE_CURRENT_LIST_DIR}/aurora_deko3d_uam_rename.cmake"
        COMMENT "uam: renaming the symbols NVK's Mesa also defines"
        VERBATIM)
add_custom_target(aurora_uam_renamed DEPENDS "${_dk_uam_renamed}")
add_library(aurora_uam STATIC IMPORTED GLOBAL)
set_target_properties(aurora_uam PROPERTIES IMPORTED_LOCATION "${_dk_uam_renamed}"
        INTERFACE_INCLUDE_DIRECTORIES "${_dk_sw}/uam")
add_dependencies(aurora_uam aurora_uam_renamed)

# --- the renderer's own shaders (ImGui, the 3x5 text): GLSL compiled by devkitPro's uam, embedded ---
find_program(AURORA_DEKO3D_UAM uam HINTS "${DEVKITPRO}/tools/bin" NO_DEFAULT_PATH REQUIRED)
set(_dk_gen "${CMAKE_BINARY_DIR}/aurora_deko3d/gen")
file(MAKE_DIRECTORY "${_dk_gen}")
set(_dk_embedded)
foreach (_name imgui_vsh imgui_fsh text_vsh text_fsh)
    if (_name MATCHES "_vsh$")
        set(_stage vert)
    else ()
        set(_stage frag)
    endif ()
    set(_src "${_dk_lib_dir}/mpl/shaders/${_name}.glsl")
    add_custom_command(OUTPUT "${_dk_gen}/${_name}_dksh.h"
            COMMAND "${AURORA_DEKO3D_UAM}" -s ${_stage} -o "${_dk_gen}/${_name}.dksh" "${_src}"
            COMMAND "${CMAKE_COMMAND}" -DIN=${_dk_gen}/${_name}.dksh -DOUT=${_dk_gen}/${_name}_dksh.h
                    -DNAME=${_name}_dksh -P "${CMAKE_CURRENT_LIST_DIR}/aurora_deko3d_embed.cmake"
            DEPENDS "${_src}" "${CMAKE_CURRENT_LIST_DIR}/aurora_deko3d_embed.cmake"
            COMMENT "uam: ${_name}.glsl")
    list(APPEND _dk_embedded "${_dk_gen}/${_name}_dksh.h")
endforeach ()

# --- the platform layer (MPL) and the shader machinery: their own library ---
add_library(aurora_deko STATIC
        "${_dk_lib_dir}/mpl/device.cpp"
        "${_dk_lib_dir}/mpl/memory.cpp"
        "${_dk_lib_dir}/mpl/descriptors.cpp"
        "${_dk_lib_dir}/mpl/overlay.cpp"
        "${_dk_lib_dir}/mpl/pattern.cpp"
        "${_dk_lib_dir}/mpl/shaders.cpp"
        "${_dk_lib_dir}/mpl/dksh_file.cpp"
        "${_dk_lib_dir}/shader_cache.cpp"
        "${_dk_lib_dir}/shader_translate.cpp"
        "${_dk_lib_dir}/dk_runtime.cpp"
        ${_dk_embedded})
target_include_directories(aurora_deko PUBLIC "${_dk_lib_dir}" "${_dk_lib_dir}/mpl")
target_include_directories(aurora_deko PRIVATE "${_dk_gen}" "${dawn_SOURCE_DIR}" "${dawn_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_LIST_DIR}/../include"
        "${CMAKE_CURRENT_LIST_DIR}/../lib")
set_target_properties(aurora_deko PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
# The SD-card folders (lib/deko/README.md): the offline DKSH cache ships in romfs, what this
# console compiles goes next to Aurora's pipeline cache.
target_compile_definitions(aurora_deko PRIVATE "COS_SWITCH_NRO_DIR=\"romfs:\""
        "COS_SWITCH_DATA_DIR=\"sdmc:/switch/dusklight\"")
add_dependencies(aurora_deko aurora_uam_renamed)
target_link_libraries(aurora_deko PRIVATE imgui xxHash::xxhash aurora_uam fmt::fmt ZLIB::ZLIB
        tint_api tint_lang_glsl_writer tint_lang_wgsl_reader tint_lang_wgsl_inspector
        PUBLIC deko3d_release)

# Every unit of Aurora sees the define (the structs it adds fields to are shared).
target_compile_definitions(aurora_core PUBLIC AURORA_GFX_DEKO3D=1)
target_link_libraries(aurora_core PUBLIC aurora_deko)
# Aurora's side: inside aurora_gx, with its internal headers.
target_sources(aurora_gx PRIVATE
        "${_dk_lib_dir}/dk_objects.cpp"
        "${_dk_lib_dir}/dk_pipeline.cpp"
        "${_dk_lib_dir}/dk_encode.cpp"
        "${_dk_lib_dir}/dk_present.cpp"
        "${_dk_lib_dir}/dk_rmlui.cpp")
target_include_directories(aurora_gx PRIVATE "${_dk_lib_dir}" "${_dk_lib_dir}/mpl" "${CMAKE_CURRENT_LIST_DIR}/../lib")
target_include_directories(aurora_core PRIVATE "${_dk_lib_dir}" "${_dk_lib_dir}/mpl")

# The wgpu* functions whose objects get deko3d shadows (dk_objects.cpp); the executable adds
# -Wl,--wrap= for each (AURORA_DEKO3D_WRAP_FUNCTIONS).
set(AURORA_DEKO3D_WRAP_FUNCTIONS
        wgpuDeviceCreateTexture wgpuTextureAddRef wgpuTextureRelease wgpuTextureCreateView
        wgpuTextureViewAddRef wgpuTextureViewRelease wgpuDeviceCreateSampler wgpuSamplerAddRef wgpuSamplerRelease
        wgpuDeviceCreateBindGroup wgpuBindGroupAddRef wgpuBindGroupRelease
        CACHE INTERNAL "")
message(STATUS "aurora: deko3d renderer (run-time selectable), third party from ${AURORA_DEKO3D_SW_DIR}")
