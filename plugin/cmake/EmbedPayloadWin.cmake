# Make a Windows build self-contained: onnxruntime.dll beside the binary that
# loads it, the exported graphs where findModels() looks for them.
#
# The reason is the same as on macOS -- a plugin that reaches outside itself
# works exactly once, on the machine it was built on -- but none of the
# mechanism carries over. Windows has no rpath and no @loader_path: a DLL is
# resolved from the directory of the module that imported it, or from PATH, so
# the only reliable answer is to put it there. There is also no signature to
# invalidate, which is why this script is a quarter the length of its Mach-O
# counterpart.
#
# Run as a script step (cmake -P) from POST_BUILD; BINDIR is only known then.

if(NOT EXISTS "${ORT_RUNTIME}")
    message(STATUS "EmbedPayloadWin: no dll at ${ORT_RUNTIME}; skipping")
    return()
endif()
if(NOT IS_DIRECTORY "${BINDIR}")
    message(STATUS "EmbedPayloadWin: no ${BINDIR}; skipping")
    return()
endif()

file(COPY "${ORT_RUNTIME}" DESTINATION "${BINDIR}")

# A VST3 on Windows is still a bundle -- Contents/x86_64-win/Plug.vst3 -- so
# the binary's parent is Contents and the graphs belong in its Resources, which
# is the same place the macOS build puts them and the same place findModels()
# checks first. A standalone .exe has no bundle around it, so they sit beside
# the executable instead.
get_filename_component(parent "${BINDIR}" DIRECTORY)
get_filename_component(parent_name "${parent}" NAME)
if(parent_name STREQUAL "Contents")
    set(models_dir "${parent}/Resources/models")
else()
    set(models_dir "${BINDIR}/models")
endif()

# Absent during a first build, which is fine: the processor falls back to the
# mock engine and says so, rather than failing to load.
if(EXISTS "${MODELS}/flow.onnx" AND EXISTS "${MODELS}/decoder.onnx")
    file(MAKE_DIRECTORY "${models_dir}")
    file(GLOB payload "${MODELS}/*.onnx" "${MODELS}/*.onnx.data" "${MODELS}/manifest.json")
    file(COPY ${payload} DESTINATION "${models_dir}")
    message(STATUS "EmbedPayloadWin: ${BINDIR} (+ graphs in ${models_dir})")
else()
    message(STATUS "EmbedPayloadWin: ${BINDIR} (dll only; no graphs exported yet)")
endif()
