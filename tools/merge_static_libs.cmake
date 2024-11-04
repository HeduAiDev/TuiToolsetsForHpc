# Merge_static_libs(outlib lib1 lib2 ... libn) merges a number of static
# libs into a single static library.
function(merge_static_libs outlib)
    set(libs ${ARGV})
    list(REMOVE_AT libs 0)
    # Create a dummy file that the target will depend on
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/${outlib}_dummy.cpp)
    file(WRITE ${dummyfile} "const char* dummy = \"${dummyfile}\";")

    add_library(${outlib} STATIC ${dummyfile})
    set(COMBINED_INCLUDE_DIRS)
    # Type check
    foreach(lib ${libs})
        get_target_property(INCLUDE_DIRS ${lib} INCLUDE_DIRECTORIES)
        list(APPEND COMBINED_INCLUDE_DIRS ${INCLUDE_DIRS})
        list(REMOVE_DUPLICATES COMBINED_INCLUDE_DIRS)
        get_target_property(libtype ${lib} TYPE)
        if(NOT libtype STREQUAL "STATIC_LIBRARY")
            message(FATAL_ERROR "Merge_static_libs can only process static libraries\n\tlibraries: ${lib}\n\tlibtype ${libtype}")
        endif()
    endforeach()
    target_include_directories(${outlib} PUBLIC ${COMBINED_INCLUDE_DIRS})
    if(MSVC)
        if(CMAKE_LIBTOOL)
            set(BUNDLE_TOOL ${CMAKE_LIBTOOL})
            else()
            find_program(BUNDLE_TOOL lib HINTS "${CMAKE_C_COMPILER}/..")

            if(NOT BUNDLE_TOOL)
                message(FATAL_ERROR "Cannot locate lib.exe to bundle libraries")
            endif()
        endif()
        # Use add_custom_command to merge libraries with lib.exe
        set(flags "")
        foreach(lib ${libs})
            list(APPEND flags $<TARGET_FILE:${lib}>)
        endforeach()
        add_custom_command(TARGET ${outlib} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Merging libraries into ${outlib}"
            COMMAND ${BUNDLE_TOOL} /OUT:$<TARGET_FILE:${outlib}> ${flags} VERBATIM)
    elseif(APPLE)
        # Use OSX's libtool to merge archives
        set(flags "")
        foreach(lib ${libs})
            list(APPEND flags $<TARGET_FILE:${lib}>)
        endforeach()
        add_custom_command(TARGET ${outlib} POST_BUILD
            COMMAND rm $<TARGET_FILE:${outlib}>
            COMMAND xcrun libtool -static -o $<TARGET_FILE:${outlib}>
            ${flags}>
        )
    else() # general UNIX use ar to merge archives
        add_custom_command(
            TARGET ${outlib}
            POST_BUILD
            COMMENT "Merge static libraries with ar"
            COMMAND ${CMAKE_COMMAND} -E echo CREATE $<TARGET_FILE:${outlib}> >script.ar
        )

        foreach(lib ${libs})
            add_custom_command(TARGET ${outlib} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo ADDLIB $<TARGET_FILE:${lib}> >>script.ar
            )
        endforeach()

        add_custom_command(
            TARGET ${outlib}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo SAVE >>script.ar
            COMMAND ${CMAKE_COMMAND} -E echo END >>script.ar
            COMMAND ${CMAKE_AR} -M <script.ar
            COMMAND ${CMAKE_COMMAND} -E remove script.ar
        )
    endif()

    file(WRITE ${dummyfile}.base "const char* ${outlib}_sublibs=\"${libs}\";")
    add_custom_command(
        OUTPUT  ${dummyfile}
        COMMAND ${CMAKE_COMMAND} -E copy ${dummyfile}.base ${dummyfile}
        DEPENDS ${libs} ${extrafiles})
endfunction()
