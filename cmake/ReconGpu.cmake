# ReconGpu.cmake - optional GPU offload of KISS-ICP's data association (Jetson Nano, CUDA).
#
#   cmake -B build -DRECON_GPU=ON            # needs nvcc (default /usr/local/cuda/bin/nvcc)
#   KISS_USE_GPU=1 ./build/recon ...         # run with the GPU; without it, the CPU path runs
#
# The CUDA code is built with the system nvcc into a static library with a plain C interface
# (gpu/), and linked into KISS-ICP's core. KISS-ICP itself keeps using the project compiler.
option(RECON_GPU "Offload KISS-ICP data association to the GPU (CUDA, Jetson Nano)" OFF)
set(RECON_NVCC "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "nvcc used to build the GPU library")
set(RECON_CUDA_ARCH "sm_53" CACHE STRING "GPU architecture for nvcc (sm_53 = Jetson Nano)")
set(RECON_CUDA_LIBDIR "/usr/local/cuda/lib64" CACHE PATH "Directory that contains libcudart")
# nvcc 10.2 only accepts GCC <= 8 as host compiler. Force the system one, so that a newer GCC
# first in PATH (e.g. the conda GCC 12 used for the rest of the project) cannot be picked.
set(RECON_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE FILEPATH "Host compiler used by nvcc")

get_filename_component(_RECON_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(RECON_GPU AND NOT EXISTS "${RECON_NVCC}")
  message(FATAL_ERROR "RECON_GPU=ON but nvcc was not found at ${RECON_NVCC} (use -DRECON_NVCC=<path>)")
endif()
message(STATUS "GPU offload: RECON_GPU=${RECON_GPU}")

# Call after add_subdirectory(third_party/kiss_icp).
function(recon_gpu_attach)
  if(NOT RECON_GPU)
    return()
  endif()
  set(_gdir "${_RECON_ROOT}/gpu")
  set(_out  "${CMAKE_BINARY_DIR}/gpu")
  add_custom_command(
    OUTPUT ${_out}/libgpu_assoc.a
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_out}
    COMMAND ${RECON_NVCC} -ccbin ${RECON_CUDA_HOST_COMPILER} -std=c++14 --fmad=false -O2
            -arch=${RECON_CUDA_ARCH} -Xcompiler -fPIC -I${_gdir} -c ${_gdir}/gpu_associate.cu -o ${_out}/gpu_associate.o
    COMMAND ${CMAKE_AR} rcs ${_out}/libgpu_assoc.a ${_out}/gpu_associate.o
    DEPENDS ${_gdir}/gpu_associate.cu ${_gdir}/gpu_associate.h ${_gdir}/voxel_search.h
            ${_gdir}/sort_voxels.h ${_gdir}/associate_one.h
    COMMENT "nvcc: building the GPU data-association library")
  add_custom_target(gpu_assoc_build DEPENDS ${_out}/libgpu_assoc.a)

  add_library(gpu_assoc STATIC IMPORTED GLOBAL)
  set_target_properties(gpu_assoc PROPERTIES
    IMPORTED_LOCATION ${_out}/libgpu_assoc.a
    INTERFACE_INCLUDE_DIRECTORIES ${_gdir}
    INTERFACE_LINK_LIBRARIES "-L${RECON_CUDA_LIBDIR};-lcudart;-ldl;-lrt;-lpthread;-Wl,-rpath,${RECON_CUDA_LIBDIR}")
  add_dependencies(gpu_assoc gpu_assoc_build)

  target_compile_definitions(kiss_icp_core PRIVATE KISS_ICP_USE_GPU)
  target_link_libraries(kiss_icp_core PUBLIC gpu_assoc)
endfunction()
