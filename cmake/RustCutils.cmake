# Experimental native PyO3 build. Keep scikit-build-core as the sole wheel backend.
if(CMAKE_CROSSCOMPILING OR EMSCRIPTEN)
  message(FATAL_ERROR "Rust cutils prototype supports native builds only; disable SHAP_BUILD_RUST_CUTILS for cross builds")
endif()
if(APPLE AND CMAKE_OSX_ARCHITECTURES AND
   NOT "${CMAKE_OSX_ARCHITECTURES}" STREQUAL "${CMAKE_HOST_SYSTEM_PROCESSOR}")
  message(FATAL_ERROR "Rust cutils prototype does not support macOS cross/universal builds")
endif()
find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
  message(FATAL_ERROR "SHAP_BUILD_RUST_CUTILS requires Rust and Cargo on PATH")
endif()
set(RUST_CUTILS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rust/cutils")
set(RUST_CUTILS_TARGET "${CMAKE_CURRENT_BINARY_DIR}/rust-target")
set(RUST_CUTILS_LIBRARY "${RUST_CUTILS_TARGET}/release/${CMAKE_SHARED_LIBRARY_PREFIX}_rust_cutils${CMAKE_SHARED_LIBRARY_SUFFIX}")

# GIL builds use abi3. Free-threaded builds require their interpreter-specific suffix.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import sys, sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') if sysconfig.get_config_var('Py_GIL_DISABLED') else ('.pyd' if sys.platform == 'win32' else '.abi3.so'))"
  OUTPUT_VARIABLE RUST_CUTILS_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE RUST_PYTHON_RESULT
)
if(NOT RUST_PYTHON_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not query the Rust extension suffix from Python")
endif()
add_custom_command(
  OUTPUT "${RUST_CUTILS_LIBRARY}"
  COMMAND ${CMAKE_COMMAND} -E env
    "PYO3_PYTHON=${Python_EXECUTABLE}"
    "PYO3_BUILD_EXTENSION_MODULE=1"
    "CARGO_TARGET_DIR=${RUST_CUTILS_TARGET}"
    "${CARGO_EXECUTABLE}" build --release --locked
      --manifest-path "${RUST_CUTILS_SOURCE}/Cargo.toml"
  WORKING_DIRECTORY "${RUST_CUTILS_SOURCE}"
  DEPENDS rust/cutils/Cargo.toml rust/cutils/Cargo.lock
          rust/cutils/build.rs rust/cutils/src/lib.rs
  COMMENT "Building experimental Rust cutils (PyO3 + rust-numpy)"
  VERBATIM
)
add_custom_target(_rust_cutils ALL DEPENDS "${RUST_CUTILS_LIBRARY}")
install(FILES "${RUST_CUTILS_LIBRARY}" DESTINATION ${SKBUILD_PROJECT_NAME}
        RENAME "_rust_cutils${RUST_CUTILS_SUFFIX}")
