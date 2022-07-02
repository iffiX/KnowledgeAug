# Install script for directory: /home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/xtl" TYPE FILE FILES
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xany.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xbasic_fixed_string.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xbase64.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xclosure.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xcomplex.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xcomplex_sequence.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xspan.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xspan_impl.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xdynamic_bitset.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xfunctional.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xhalf_float.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xhalf_float_impl.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xhash.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xhierarchy_generator.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xiterator_base.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xjson.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xmasked_value_meta.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xmasked_value.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xmeta_utils.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xmultimethods.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xoptional_meta.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xoptional.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xoptional_sequence.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xplatform.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xproxy_wrapper.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xsequence.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xsystem.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xtl_config.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xtype_traits.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xvariant.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xvariant_impl.hpp"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/xtl/include/xtl/xvisitor.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/xtl" TYPE FILE FILES
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/cmake-build-debug/xtl/xtlConfig.cmake"
    "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/cmake-build-debug/xtl/xtlConfigVersion.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtl/xtlTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtl/xtlTargets.cmake"
         "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/cmake-build-debug/xtl/CMakeFiles/Export/share/cmake/xtl/xtlTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtl/xtlTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtl/xtlTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/xtl" TYPE FILE FILES "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/cmake-build-debug/xtl/CMakeFiles/Export/share/cmake/xtl/xtlTargets.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pkgconfig" TYPE FILE FILES "/home/Administrator/iffi/Projects/NWU_CS396_StatiticalLanguageModels/kb_encoder/encoder/dataset/concept_net/matcher_src/cmake-build-debug/xtl/xtl.pc")
endif()

