# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /data/software/clion-2020.3.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /data/software/clion-2020.3.4/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug

# Include any dependencies generated for this target.
include backward-cpp/CMakeFiles/backward_object.dir/depend.make

# Include the progress variables for this target.
include backward-cpp/CMakeFiles/backward_object.dir/progress.make

# Include the compile flags for this target's objects.
include backward-cpp/CMakeFiles/backward_object.dir/flags.make

backward-cpp/CMakeFiles/backward_object.dir/backward.cpp.o: backward-cpp/CMakeFiles/backward_object.dir/flags.make
backward-cpp/CMakeFiles/backward_object.dir/backward.cpp.o: ../backward-cpp/backward.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object backward-cpp/CMakeFiles/backward_object.dir/backward.cpp.o"
	cd /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug/backward-cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/backward_object.dir/backward.cpp.o -c /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/backward-cpp/backward.cpp

backward-cpp/CMakeFiles/backward_object.dir/backward.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/backward_object.dir/backward.cpp.i"
	cd /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug/backward-cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/backward-cpp/backward.cpp > CMakeFiles/backward_object.dir/backward.cpp.i

backward-cpp/CMakeFiles/backward_object.dir/backward.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/backward_object.dir/backward.cpp.s"
	cd /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug/backward-cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/backward-cpp/backward.cpp -o CMakeFiles/backward_object.dir/backward.cpp.s

backward_object: backward-cpp/CMakeFiles/backward_object.dir/backward.cpp.o
backward_object: backward-cpp/CMakeFiles/backward_object.dir/build.make

.PHONY : backward_object

# Rule to build all files generated by this target.
backward-cpp/CMakeFiles/backward_object.dir/build: backward_object

.PHONY : backward-cpp/CMakeFiles/backward_object.dir/build

backward-cpp/CMakeFiles/backward_object.dir/clean:
	cd /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug/backward-cpp && $(CMAKE_COMMAND) -P CMakeFiles/backward_object.dir/cmake_clean.cmake
.PHONY : backward-cpp/CMakeFiles/backward_object.dir/clean

backward-cpp/CMakeFiles/backward_object.dir/depend:
	cd /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/backward-cpp /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug/backward-cpp /home/iffi/Projects/KnowledgeAug/encoder/dataset/matcher/matcher_src/cmake-build-debug/backward-cpp/CMakeFiles/backward_object.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : backward-cpp/CMakeFiles/backward_object.dir/depend
