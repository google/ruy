macro(RUY_TEST_COMPILE_ARM64_SME VARIABLE)
    if(NOT DEFINED ${VARIABLE})
        try_compile(${VARIABLE} "${PROJECT_BINARY_DIR}"
            "${PROJECT_SOURCE_DIR}/cmake/ruy_test_compile_ARM64_SME.cc"
            COMPILE_DEFINITIONS "-march=armv8.6-a+sve2+sme2")
        if(${VARIABLE})
            set(${VARIABLE} 1 CACHE INTERNAL "Ruy can compile ARM64 SME" FORCE)
        endif ()
    endif ()
endmacro(RUY_TEST_COMPILE_ARM64_SME)
