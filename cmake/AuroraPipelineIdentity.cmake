# aleks2 (SHADERS-c2): an identity for the code that turns a pipeline recipe into a pipeline.
#
# A recipe (pipeline_cache.db, the shipped list) stays valid across builds as long as its
# config struct does (the *PipelineConfigVersion constants), but the shaders made from it
# change whenever this code or Dawn changes -- and then everything the console compiled is
# a cache miss again. The game keys its "Preparing shaders" marker on this value too, so a
# build that changes shader generation shows the screen again instead of stuttering in
# play. Only the sources that generate shaders or build pipelines are hashed (plus the
# Dawn revision), so unrelated changes leave it alone.
#
# (Idea from SwitchWakerHD's PipelineRuntimeIdentity; written from scratch for Aurora.)
function(_aurora_identity_strip_deko3d text_var)
  set(_in "${${text_var}}")
  string(FIND "${_in}" "AURORA_GFX_DEKO3D" _any)
  if(_any EQUAL -1)
    return()
  endif()
  # one list element per line: ';' and the brackets CMake's list splitting honours are masked first
  string(REPLACE ";" "@AURORA_SEMI@" _in "${_in}")
  string(REPLACE "[" "@AURORA_LB@" _in "${_in}")
  string(REPLACE "]" "@AURORA_RB@" _in "${_in}")
  string(REPLACE "\n" ";" _lines "${_in}")
  set(_out "")
  set(_first TRUE)
  set(_depth 0)
  set(_dropBlank FALSE)
  foreach(_line IN LISTS _lines)
    if(_depth GREATER 0)
      if(_line MATCHES "^[ \t]*#[ \t]*if")
        math(EXPR _depth "${_depth} + 1")
      elseif(_line MATCHES "^[ \t]*#[ \t]*endif")
        math(EXPR _depth "${_depth} - 1")
        if(_depth EQUAL 0)
          set(_dropBlank TRUE)
        endif()
      endif()
      continue()
    endif()
    if(_line MATCHES "^#if defined\\(AURORA_GFX_DEKO3D\\)")
      set(_depth 1)
      continue()
    endif()
    if(_dropBlank AND _line STREQUAL "")
      set(_dropBlank FALSE)
      continue()
    endif()
    set(_dropBlank FALSE)
    if(NOT _first)
      string(APPEND _out "\n")
    endif()
    set(_first FALSE)
    string(APPEND _out "${_line}")
  endforeach()
  string(REPLACE "@AURORA_SEMI@" ";" _out "${_out}")
  string(REPLACE "@AURORA_LB@" "[" _out "${_out}")
  string(REPLACE "@AURORA_RB@" "]" _out "${_out}")
  set(${text_var} "${_out}" PARENT_SCOPE)
endfunction()

function(aurora_pipeline_identity out_var)
  set(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
  set(_inputs
    lib/gx/pipeline.cpp lib/gx/pipeline.hpp
    lib/gx/shader.cpp lib/gx/shader_info.cpp lib/gx/shader_info.hpp
    lib/gx/gx.hpp
    lib/gfx/clear.cpp lib/gfx/clear.hpp
    lib/rmlui/pipeline.cpp lib/rmlui/pipeline.hpp)
  set(_material "aurora-pipeline-identity-1\ndawn=${AURORA_DAWN_REF}\n")
  foreach(_rel IN LISTS _inputs)
    set(_file "${_root}/${_rel}")
    if(EXISTS "${_file}")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
      file(READ "${_file}" _text)
      # Line endings must not change the identity (a CRLF checkout builds the same shaders).
      string(REPLACE "\r\n" "\n" _text "${_text}")
      # aleks2 (deko3d P1): the deko3d renderer's blocks (#if defined(AURORA_GFX_DEKO3D) ... #endif) build no
      # Dawn shader, so they are left out (with the blank line after them): Dawn/NVK keeps the identity of
      # the same Aurora without them, and switching between a release and the lab build re-prepares nothing.
      _aurora_identity_strip_deko3d(_text)
      string(SHA256 _digest "${_text}")
      string(APPEND _material "${_rel}=${_digest}\n")
    else()
      string(APPEND _material "${_rel}=missing\n")
    endif()
  endforeach()
  string(SHA256 _identity "${_material}")
  string(SUBSTRING "${_identity}" 0 16 _short)
  set(${out_var} "${_short}" PARENT_SCOPE)
endfunction()
