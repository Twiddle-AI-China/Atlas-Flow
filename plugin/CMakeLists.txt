cmake_minimum_required(VERSION 3.22)

project(AtlasFlowPlugin VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Universal, and not optional: a host running under Rosetta is x86_64 and will
# silently skip an arm64-only plugin -- it does not appear, it does not fail,
# and nothing is logged. That is exactly what happened on the first install
# here, against an Ableton Live that had Rosetta turned on.
#
# ONNX Runtime stopped publishing macOS x86_64 builds after 1.23.2, which is
# why the version is pinned there rather than at the newest; the two slices are
# fused by plugin/tools/fetch_onnxruntime.sh.
if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
endif()

add_subdirectory(third_party/JUCE)

# AU is a macOS format; asking for it elsewhere is a configure error rather
# than a no-op, so the list is built per platform.
if(APPLE)
    set(ATLASFLOW_FORMATS AU VST3 Standalone)
else()
    set(ATLASFLOW_FORMATS VST3 Standalone)
endif()

juce_add_plugin(AtlasFlow
    PRODUCT_NAME            "Atlas Flow"
    COMPANY_NAME            "TwiddleAI"
    BUNDLE_ID               "cn.twiddle.atlasflow"
    PLUGIN_MANUFACTURER_CODE Twdl
    PLUGIN_CODE             Atfl
    FORMATS                 ${ATLASFLOW_FORMATS}
    IS_SYNTH                TRUE
    NEEDS_MIDI_INPUT        TRUE
    NEEDS_MIDI_OUTPUT       FALSE
    IS_MIDI_EFFECT          FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE)

target_sources(AtlasFlow PRIVATE
    src/PluginProcessor.cpp
    src/PluginEditor.cpp)

target_compile_definitions(AtlasFlow PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    # Off, or the VST3 publishes 2080 dummy parameters -- one per MIDI CC per
    # channel -- and every host's parameter list and automation menu is 2080
    # entries of noise around the seven that mean something. The emulation
    # exists for hosts that cannot send CC as MIDI; this synth ignores CC
    # either way, so there is nothing for it to carry.
    JUCE_VST3_EMULATE_MIDI_CC_WITH_PARAMETERS=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)

# What to link against, and -- on Windows -- what has to travel beside the
# binary. Windows has no rpath: a DLL is found next to the module that loads it
# or on PATH, so the runtime is copied rather than referenced.
if(WIN32)
    set(ORT_LINK ${ORT_DIR}/lib/onnxruntime.lib)
    set(ORT_RUNTIME ${ORT_DIR}/lib/onnxruntime.dll)
elseif(APPLE)
    set(ORT_LINK ${ORT_DIR}/lib/libonnxruntime.dylib)
else()
    set(ORT_LINK ${ORT_DIR}/lib/libonnxruntime.so)
endif()

target_link_libraries(AtlasFlow PRIVATE
    juce::juce_audio_utils
    # The scopes need an FFT. juce_dsp is the only thing pulled in for the UI
    # and it is header-heavy rather than code-heavy, so it costs compile time
    # rather than binary size.
    juce::juce_dsp
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags)

# The real engine needs ONNX Runtime. Without it the plugin still builds and
# runs -- it falls back to the mock -- so this is a conditional rather than a
# hard requirement, and a missing SDK produces a working plugin instead of a
# configure error.
if(EXISTS ${ORT_DIR}/include/onnxruntime_cxx_api.h)
    target_include_directories(AtlasFlow PRIVATE ${ORT_DIR}/include)
    target_link_libraries(AtlasFlow PRIVATE ${ORT_LINK})
    # Where to look for the exported graphs during development; the installed
    # plugin also checks $ATLASFLOW_MODELS and ~/Library/Application Support.
    target_compile_definitions(AtlasFlow PRIVATE
        ATLASFLOW_DEFAULT_MODELS="${CMAKE_CURRENT_SOURCE_DIR}/models")
    # Embed the dylib in each bundle rather than pointing an rpath at this
    # machine. An absolute path to someone's build tree works exactly once --
    # on that machine -- and fails with a load error everywhere else, which is
    # a miserable thing to debug from the other end.
    #
    # JUCE's own COPY_PLUGIN_AFTER_BUILD step is also POST_BUILD, and ours is
    # appended after it, so the embed is done against the *installed* bundle
    # too; otherwise the copy in ~/Library would be the broken one.
    if(APPLE)
        foreach(target AtlasFlow_VST3 AtlasFlow_AU AtlasFlow_Standalone)
            if(TARGET ${target})
                set_target_properties(${target} PROPERTIES
                    BUILD_WITH_INSTALL_RPATH TRUE
                    INSTALL_RPATH "@loader_path/../Frameworks")
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND}
                            -DBUNDLE=$<TARGET_BUNDLE_DIR:${target}>
                            -DORT_LIB=${ORT_DIR}/lib
                            -DMODELS=${CMAKE_CURRENT_SOURCE_DIR}/models
                            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedPayload.cmake
                    VERBATIM)
            endif()
        endforeach()
    elseif(WIN32)
        # $<TARGET_BUNDLE_DIR:> is an Apple-only generator expression, so the
        # Windows step is driven by the binary's own directory instead -- which
        # for a VST3 is Contents/x86_64-win inside the bundle, and for the
        # standalone is simply the folder holding the .exe.
        foreach(target AtlasFlow_VST3 AtlasFlow_Standalone)
            if(TARGET ${target})
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND}
                            -DBINDIR=$<TARGET_FILE_DIR:${target}>
                            -DORT_RUNTIME=${ORT_RUNTIME}
                            -DMODELS=${CMAKE_CURRENT_SOURCE_DIR}/models
                            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedPayloadWin.cmake
                    VERBATIM)
            endif()
        endforeach()
    endif()
else()
    message(WARNING "onnxruntime not found; plugin will run the mock engine only")
endif()

# ---------------------------------------------------------------------------
# Headless render check. Builds the processor without a host so the MIDI path
# can be asserted on in CI: pitch, octave folding, voice ceiling, and whether
# moving across the map actually changes the timbre.
juce_add_console_app(AtlasFlowRenderTest PRODUCT_NAME "atlasflow-render-test")

target_sources(AtlasFlowRenderTest PRIVATE tools/render_test.cpp)

target_compile_definitions(AtlasFlowRenderTest PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_STANDALONE_APPLICATION=1)

if(EXISTS ${ORT_DIR}/include/onnxruntime_cxx_api.h)
    target_include_directories(AtlasFlowRenderTest PRIVATE ${ORT_DIR}/include)
    target_link_libraries(AtlasFlowRenderTest PRIVATE ${ORT_LINK})
    if(WIN32)
        add_custom_command(TARGET AtlasFlowRenderTest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ORT_RUNTIME} $<TARGET_FILE_DIR:AtlasFlowRenderTest>
            VERBATIM)
    else()
        set_target_properties(AtlasFlowRenderTest PROPERTIES
            BUILD_RPATH ${ORT_DIR}/lib INSTALL_RPATH ${ORT_DIR}/lib)
    endif()
endif()

target_link_libraries(AtlasFlowRenderTest PRIVATE
    AtlasFlow
    juce::juce_audio_utils
    juce::juce_dsp
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)

# ---------------------------------------------------------------------------
# Host check. Loads the *built bundle* the way a DAW does, through
# juce::AudioPluginFormat, and drives it through the generic
# AudioPluginInstance interface.
#
# It deliberately links no Atlas Flow code and no ONNX Runtime. Everything it
# touches it reaches through the wrapper, which is the half of the plugin the
# other checks construct their way around -- the bundle layout, the parameter
# metadata a host reads, the state blob a project file keeps. Linking the
# plugin in as well would also put two copies of JUCE's statics in one process,
# which is its own kind of wrong.
juce_add_console_app(AtlasFlowHostTest PRODUCT_NAME "atlasflow-host-test")

target_sources(AtlasFlowHostTest PRIVATE tools/host_test.cpp)

target_compile_definitions(AtlasFlowHostTest PUBLIC
    JUCE_PLUGINHOST_VST3=1
    JUCE_PLUGINHOST_AU=$<IF:$<BOOL:${APPLE}>,1,0>
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_STANDALONE_APPLICATION=1)

target_link_libraries(AtlasFlowHostTest PRIVATE
    juce::juce_audio_utils
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)

# ---------------------------------------------------------------------------
# Offline plan renderer. Links ONNX Runtime directly rather than through JUCE,
# so the port can be checked against the Python runtime without a host, a GUI
# or an audio device in the way.
if(EXISTS ${ORT_DIR}/include/onnxruntime_cxx_api.h)
    add_executable(atlasflow-plan tools/plan_cli.cpp)
    target_include_directories(atlasflow-plan PRIVATE ${ORT_DIR}/include)
    target_link_libraries(atlasflow-plan PRIVATE ${ORT_LINK})
    if(WIN32)
        add_custom_command(TARGET atlasflow-plan POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ORT_RUNTIME} $<TARGET_FILE_DIR:atlasflow-plan>
            VERBATIM)
    else()
        set_target_properties(atlasflow-plan PROPERTIES
            BUILD_RPATH ${ORT_DIR}/lib
            INSTALL_RPATH ${ORT_DIR}/lib)
    endif()
else()
    message(STATUS "onnxruntime not found in third_party/; skipping atlasflow-plan")
endif()
