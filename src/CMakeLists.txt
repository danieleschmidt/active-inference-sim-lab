add_library(active_inference_core
    inference_engine.cpp
    planning_engine.cpp
    belief_state.cpp
    generative_model.cpp
    utils.cpp
)

# Set target properties
set_target_properties(active_inference_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "../include/active_inference/core.h"
)

# Link libraries
target_link_libraries(active_inference_core
    PUBLIC
    Eigen3::Eigen
    PRIVATE
    Threads::Threads
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(active_inference_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Include directories
target_include_directories(active_inference_core
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Compiler features
target_compile_features(active_inference_core PUBLIC cxx_std_17)

# Install library
install(TARGETS active_inference_core
    EXPORT ActiveInferenceSimLabTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/active_inference
)

# Install export
install(EXPORT ActiveInferenceSimLabTargets
    FILE ActiveInferenceSimLabTargets.cmake
    NAMESPACE ActiveInferenceSimLab::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ActiveInferenceSimLab
)