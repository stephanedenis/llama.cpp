# Shared engine: used by llama-parallel-decision and, once the endpoint is wired, by
# llama-server's /decision route (tools/server/CMakeLists.txt).
add_library(llama-decision STATIC
    decision-engine.cpp
    decision-engine.h
)
target_include_directories(llama-decision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-decision PUBLIC llama-common llama)
target_compile_features(llama-decision PUBLIC cxx_std_17)
# llama-server is a shared library (libllama-server-impl), so this must stay PIC
set_target_properties(llama-decision PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TARGET llama-parallel-decision)

add_executable(${TARGET} parallel-decision.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-decision llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
