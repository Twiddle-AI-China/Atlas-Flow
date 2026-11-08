# Make a plugin bundle self-contained: libonnxruntime inside Frameworks, the
# exported graphs inside Resources.
#
# Both exist for the same reason. A plugin that reaches outside its own bundle
# -- at an rpath into someone's build tree, or at a models/ folder in a
# checkout -- works exactly once, on the machine it was built on, and fails
# everywhere else with no useful message.
#
# Run as a script step (cmake -P) from a POST_BUILD command, because the bundle
# path is only known at build time. Also fixes up the copy JUCE installs into
# ~/Library/Audio/Plug-Ins, which is the one a host actually loads.

if(NOT EXISTS "${ORT_LIB}/libonnxruntime.dylib")
    message(STATUS "EmbedPayload: no dylib at ${ORT_LIB}; skipping")
    return()
endif()

# Resolve the real file behind the symlink chain so the bundle gets one copy.
file(REAL_PATH "${ORT_LIB}/libonnxruntime.dylib" real_lib)
get_filename_component(real_name "${real_lib}" NAME)

# The SONAME the plugin binary actually asks for.
set(soname "libonnxruntime.1.dylib")

function(embed bundle)
    if(NOT IS_DIRECTORY "${bundle}")
        return()
    endif()

    # The exported graphs. Absent during a first build, which is fine -- the
    # processor falls back to the mock and says so -- so this is a copy if
    # present rather than a hard requirement.
    if(EXISTS "${MODELS}/flow.onnx" AND EXISTS "${MODELS}/decoder.onnx")
        file(MAKE_DIRECTORY "${bundle}/Contents/Resources/models")
        file(GLOB payload "${MODELS}/*.onnx" "${MODELS}/*.onnx.data" "${MODELS}/manifest.json")
        file(COPY ${payload} DESTINATION "${bundle}/Contents/Resources/models")
    endif()
    set(frameworks "${bundle}/Contents/Frameworks")
    file(MAKE_DIRECTORY "${frameworks}")
    file(COPY "${real_lib}" DESTINATION "${frameworks}")
    file(RENAME "${frameworks}/${real_name}" "${frameworks}/${soname}" RESULT moved)

    execute_process(COMMAND install_name_tool -id "@rpath/${soname}"
                            "${frameworks}/${soname}" ERROR_QUIET)

    # The executable inside the bundle; its name matches the bundle's.
    get_filename_component(stem "${bundle}" NAME_WE)
    set(binary "${bundle}/Contents/MacOS/${stem}")
    if(EXISTS "${binary}")
        execute_process(COMMAND install_name_tool -add_rpath "@loader_path/../Frameworks"
                                "${binary}" ERROR_QUIET)
        # Re-sign ad hoc: editing a Mach-O invalidates whatever signature it had,
        # and an invalid signature is refused outright on Apple silicon.
        execute_process(COMMAND codesign --force --sign - --timestamp=none
                                "${frameworks}/${soname}" ERROR_QUIET)
        execute_process(COMMAND codesign --force --sign - --timestamp=none
                                "${bundle}" ERROR_QUIET)
    endif()
    message(STATUS "EmbedPayload: ${bundle}")
endfunction()

embed("${BUNDLE}")

# And the installed copy, which is the one a host loads.
get_filename_component(bundle_name "${BUNDLE}" NAME)
get_filename_component(bundle_ext "${BUNDLE}" LAST_EXT)
set(installed "")
if(bundle_ext STREQUAL ".vst3")
    set(installed "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/${bundle_name}")
elseif(bundle_ext STREQUAL ".component")
    set(installed "$ENV{HOME}/Library/Audio/Plug-Ins/Components/${bundle_name}")
endif()
if(installed)
    embed("${installed}")
endif()
