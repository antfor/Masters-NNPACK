set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv)

#set(CMAKE_SYSROOT /home/victor/Documents/llvm-EPI-0.7-release-toolchain-cross/riscv64-unknown-elf)
#set(CMAKE_STAGING_PREFIX /home/devel/stage)

#set(tools /home/victor/Documents/llvm-EPI-0.7-release-toolchain-cross/riscv64-unknown-elf)
set(CMAKE_C_COMPILER /home/victor/Documents/llvm-EPI-development-toolchain-cross/bin/clang)
set(CMAKE_CXX_COMPILER /home/victor/Documents/llvm-EPI-development-toolchain-cross/bin/clang++)

set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} --target=riscv64-unknown-linux-gnu -mepi" CACHE STRING "")


set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)