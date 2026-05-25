# omni

find_package(Threads REQUIRED)

# CoreML / ANE support option
option(ENABLE_COREML "Enable CoreML/ANE support for vision encoder" OFF)

add_library(omni
            omni.cpp
            omni.h
            omni-log.cpp
            omni-log.h
            omni-runtime-init.cpp
            omni-runtime-init.h
            omni-encode-stage.cpp
            omni-encode-stage.h
            omni-llm-stage.cpp
            omni-llm-stage.h
            omni-session-state.cpp
            omni-session-state.h
            omni-sliding-window.cpp
            omni-sliding-window.h
            omni-turn-coordinator.cpp
            omni-turn-coordinator.h
            omni-runtime-messages.h
            omni-worker-state.h
            omni-t2w-stage.cpp
            omni-t2w-stage.h
            omni-tts-stage.cpp
            omni-tts-stage.h
            omni-token-protocol.cpp
            omni-token-protocol.h
            omni-output.cpp
            omni-output.h
            omni-python-t2w.cpp
            omni-python-t2w.h
            omni-worker-coordinator.cpp
            omni-worker-coordinator.h
            audition.cpp
            audition.h
            vision.cpp
            vision.h
            omni-impl.h
            token2wav/token2wav-impl.cpp
            token2wav/token2wav-impl.h
            token2wav/token2wav.cpp
            token2wav/token2wav.h
            )

# Add CoreML related files when enabled
if(ENABLE_COREML)
    target_sources(omni PRIVATE
        coreml/omni_coreml.h
        coreml/omni_coreml.mm
        coreml/coreml_minicpmo45_vit_all_f16.h
        coreml/coreml_minicpmo45_vit_all_f16.m
    )
    # Define compile-time macro for code guards
    target_compile_definitions(omni PRIVATE ENABLE_COREML)

    # Enable ARC for Objective-C files
    set_source_files_properties(coreml/omni_coreml.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(coreml/coreml_minicpmo45_vit_all_f16.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

target_link_libraries     (omni PUBLIC ggml llama common)
target_link_libraries     (omni PRIVATE Threads::Threads)
target_include_directories(omni PUBLIC  .)
target_include_directories(omni PRIVATE ../..)
target_include_directories(omni PRIVATE ../../vendor)
target_include_directories(omni PRIVATE ../../ggml/src)
target_include_directories(omni PRIVATE ../../ggml/src/ggml-cpu)
target_compile_features   (omni PRIVATE cxx_std_17)

# Link CoreML and Accelerate frameworks when CoreML is enabled
if(ENABLE_COREML)
    target_link_libraries(omni PRIVATE
        "-framework Foundation"
        "-framework CoreML"
        "-framework Accelerate"
        "-ObjC"
    )
endif()

# Windows-specific definitions
if (WIN32)
    target_compile_definitions(omni PRIVATE NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (omni PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(omni PRIVATE LLAMA_BUILD)
    target_compile_definitions(omni PUBLIC  LLAMA_SHARED)
    if (WIN32)
        set_target_properties(omni PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

set(OMNI_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/omni.h
    )

set_target_properties(omni
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS omni LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(omni PRIVATE -Wno-cast-qual)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(omni        BUILD_INFO)
endif()

set(TARGET llama-omni-cli)
add_executable         (${TARGET} omni-cli.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-omni-cli)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# 双工模式测试
set(TARGET_DUPLEX_TEST llama-omni-test-duplex)
add_executable         (${TARGET_DUPLEX_TEST} test/test-duplex.cpp)
set_target_properties  (${TARGET_DUPLEX_TEST} PROPERTIES OUTPUT_NAME llama-omni-test-duplex)
target_link_libraries  (${TARGET_DUPLEX_TEST} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_DUPLEX_TEST} PRIVATE cxx_std_17)

# LLM 推理性能基准测试
set(TARGET_LLM_BENCH llama-omni-test-llm-bench)
add_executable         (${TARGET_LLM_BENCH} test/test-llm-bench.cpp)
set_target_properties  (${TARGET_LLM_BENCH} PROPERTIES OUTPUT_NAME llama-omni-test-llm-bench)
target_link_libraries  (${TARGET_LLM_BENCH} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_LLM_BENCH} PRIVATE cxx_std_17)

# VIT 推理性能基准测试
set(TARGET_VIT_BENCH llama-omni-test-vit-bench)
add_executable         (${TARGET_VIT_BENCH} test/test-vit-bench.cpp)
set_target_properties  (${TARGET_VIT_BENCH} PROPERTIES OUTPUT_NAME llama-omni-test-vit-bench)
target_link_libraries  (${TARGET_VIT_BENCH} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_VIT_BENCH} PRIVATE cxx_std_17)

# LLM SM partitioning benchmark (analytical + prefill + decode + SM scaling)
set(TARGET_LLM_SM_BENCH llama-omni-bench-llm)
add_executable         (${TARGET_LLM_SM_BENCH} test/bench-llm.cpp)
set_target_properties  (${TARGET_LLM_SM_BENCH} PROPERTIES OUTPUT_NAME llama-omni-bench-llm)
target_link_libraries  (${TARGET_LLM_SM_BENCH} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_LLM_SM_BENCH} PRIVATE cxx_std_17)

# TTS memory-bound验证 + SM scaling benchmark
set(TARGET_TTS_BENCH llama-omni-bench-tts)
add_executable         (${TARGET_TTS_BENCH} test/bench-tts.cpp)
set_target_properties  (${TARGET_TTS_BENCH} PROPERTIES OUTPUT_NAME llama-omni-bench-tts)
target_link_libraries  (${TARGET_TTS_BENCH} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_TTS_BENCH} PRIVATE cxx_std_17)

# Token2Wav SM scaling benchmark
set(TARGET_T2W_BENCH llama-omni-bench-t2w)
add_executable         (${TARGET_T2W_BENCH} test/bench-t2w.cpp)
set_target_properties  (${TARGET_T2W_BENCH} PROPERTIES OUTPUT_NAME llama-omni-bench-t2w)
target_link_libraries  (${TARGET_T2W_BENCH} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_T2W_BENCH} PRIVATE cxx_std_17)

# Vision encoder SM scaling benchmark
set(TARGET_VISION_BENCH llama-omni-bench-vision)
add_executable         (${TARGET_VISION_BENCH} test/bench-vision.cpp)
set_target_properties  (${TARGET_VISION_BENCH} PROPERTIES OUTPUT_NAME llama-omni-bench-vision)
target_link_libraries  (${TARGET_VISION_BENCH} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_VISION_BENCH} PRIVATE cxx_std_17)
