#=============================================================================
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#=============================================================================
# users should provide locations for QNN_INCLUDE and HEXAGON_SDK_ROOT
# export HEXAGON_SDK_ROOT = /path/to/hexagon-sdk

# check all setup prerequisites if the command goal is not clean
ifneq ($(MAKECMDGOALS),clean)
ifndef QNN_INCLUDE
$(info "INFO: Qnn include not explicitly defined, attempting to use QNN_SDK_ROOT if it is valid")
QNN_INCLUDE := $(QNN_SDK_ROOT)/include/QNN
endif
ifndef QNN_TARGET_LIB
$(info "INFO: Qnn target not explicitly defined, attempting to use QNN_SDK_ROOT if it is valid")
QNN_TARGET_LIB := $(QNN_SDK_ROOT)/lib/aarch64-android
endif
ifeq ($(wildcard $(QNN_INCLUDE)),)
$(error "ERROR: QNN_INCLUDE path is not set. QNN include paths must be set to obtain BE headers necessary to compile the package")
endif
ifeq ($(wildcard $(QNN_TARGET_LIB)),)
ifeq ($(MAKECMDGOALS),htp_aarch64)
$(error "ERROR: QNN_TARGET_LIB is needed to compile package for aarch64")
else ifeq ($(MAKECMDGOALS),all)
$(info "WARNING:QNN_TARGET_LIB may need to be defined to compile packages")
endif
endif

ifndef HEXAGON_SDK_ROOT
$(error "ERROR: HEXAGON_SDK_ROOT is not set. Hexagon-SDK path must be set to the latest hexagon-sdk-x.y.z")
endif

ifeq ($(wildcard $(HEXAGON_SDK_ROOT)),)
$(error "ERROR: HEXAGON_SDK_ROOT is not set correctly. Please provide hexagon-sdk-X.Y.Z path")
endif

HEXAGON_SDK_BASE := $(dir $(HEXAGON_SDK_ROOT))

$(info "The parent directory for all hexagon-sdk [${HEXAGON_SDK_BASE}]")

# Users should note that the tools version may change between hexagon sdk versions
# Following combination of SDK and Tool version is supported
HEXAGON_SDK_ROOT_V68 := $(HEXAGON_SDK_BASE)/hexagon-sdk-4.2.0
HEXAGON_SDK_ROOT_V69 := $(HEXAGON_SDK_BASE)/hexagon-sdk-4.3.0
HEXAGON_SDK_ROOT_V73 := $(HEXAGON_SDK_BASE)/hexagon-sdk-5.4.0
HEXAGON_SDK_ROOT_V75 := $(HEXAGON_SDK_BASE)/hexagon-sdk-5.4.0

#Updated to point to latest sdk to match with libQnnHtp.so
HEXAGON_SDK_ROOT_X86 := $(HEXAGON_SDK_BASE)/hexagon-sdk-5.4.0
HEXAGON_TOOLS_VERSION_V68 := 8.4.09
HEXAGON_TOOLS_VERSION_V69 := 8.5.03
HEXAGON_TOOLS_VERSION_V73 := 8.6.02
HEXAGON_TOOLS_VERSION_V75 := 8.7.03

#Updated to point to latest sdk to match with libQnnHtp.so
HEXAGON_TOOLS_VERSION_X86 := 8.7.03

ifndef ANDROID_NDK_ROOT
ifeq ($(MAKECMDGOALS),htp_aarch64)
$(error "ERROR: ANDROID_NDK_ROOT is not set. Android NDK path must be set to compile package for aarch64")
else ifeq ($(MAKECMDGOALS),all)
$(info "WARNING: ANDROID_NDK_ROOT is not set. Android NDK path must be set to compile package for aarch64")
endif
endif

SUPPORTED_TARGETS = x86_64-linux-clang hexagon-v68 hexagon-v69 hexagon-v73 hexagon-v75 aarch64-android

COMMON_CXX_FLAGS = -std=c++17 -I$(QNN_INCLUDE) -fPIC -Wall -Wreorder -Wno-missing-braces -Wno-unused-function
COMMON_CXX_FLAGS += -Werror -Wno-format -Wno-unused-command-line-argument -fvisibility=default -stdlib=libc++
COMMON_CXX_FLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"

X86_LIBNATIVE_RELEASE_DIR := $(HEXAGON_SDK_ROOT_X86)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_X86)/Tools

# Ensure hexagon sdk tool version can be retrieved
ifeq ($(wildcard $(X86_LIBNATIVE_RELEASE_DIR)/.),)
$(error "Cannot retrieve hexagon tools from: $(X86_LIBNATIVE_RELEASE_DIR).  \
         \
         Please check that hexagon tools version is correct. Expected: $(HEXAGON_TOOLS_VERSION_X86)")
endif

#Check tools for hexagon_v68 are present.
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V68)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V68 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V68)")
endif

ifeq ($(MAKECMDGOALS),htp_v69)
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V69)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V69 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V69)")
endif
endif

ifeq ($(MAKECMDGOALS),htp_v73)
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V73)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V73 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V73)")
endif
endif

ifeq ($(MAKECMDGOALS),htp_v75)
ifeq ($(wildcard $(HEXAGON_SDK_ROOT_V75)),)
$(error "ERROR: HEXAGON_SDK_ROOT_V75 is set incorrectly. Cannot retrieve $(HEXAGON_SDK_ROOT_V75)")
endif
endif



endif

PACKAGE_NAME = examples.OpPackage
WORK := build
OP_SOURCES = $(wildcard *.cpp)
HFILES = $(wildcard $(QNN_INCLUDE)/*.h)
HFILES += $(wildcard $(QNN_INCLUDE)/HTP/*.h)
HFILES += $(wildcard $(QNN_INCLUDE)/HTP/core/*.h)
OP_OBJS = $(patsubst %.cpp,%.o,$(OP_SOURCES))
LIBRARY_NAME := libQnnHtpOpPackageExample.so

all: htp_v68 htp_v69 htp_v73 htp_v75 htp_x86 htp_aarch64

#============================================================================================================
# Setup compiler, compiler instructions and linker for x86
X86_CXX ?= clang++-9
# Checking if clang++-9 is present. If not switch to clang++
ifeq ($(shell $(X86_CXX) -v 2>&1 | grep -c "clang version"), 0)
  X86_CXX := clang++
endif
X86_LDFLAGS:= -Wl,--whole-archive -L$(X86_LIBNATIVE_RELEASE_DIR)/libnative/lib -lnative -Wl,--no-whole-archive -lpthread
X86_C_FLAGS := -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include -ffast-math -DUSE_OS_LINUX
X86_CXX_FLAGS = $(COMMON_CXX_FLAGS) $(X86_C_FLAGS) -fomit-frame-pointer -Wno-invalid-offsetof
linux_objs =
#============================================================================================================
# Setup compiler, compiler instructions and linker for hexagon
HEXAGON_CXX_FLAGS := $(COMMON_CXX_FLAGS) -mhvx -mhvx-length=128B -mhmx -DUSE_OS_QURT -O2 -Wno-reorder -DPREPARE_DISABLED

HEXAGON_CXX_FLAGS_V68 := $(HEXAGON_CXX_FLAGS) -mv68 -I$(HEXAGON_SDK_ROOT_V68)/rtos/qurt/computev68/include/qurt -I$(HEXAGON_SDK_ROOT_V68)/rtos/qurt/computev68/include/posix -I$(HEXAGON_SDK_ROOT_V68)/incs -I$(HEXAGON_SDK_ROOT_V68)/incs/stddef
HEXAGON_CXX_FLAGS_V69 := $(HEXAGON_CXX_FLAGS) -mv69 -I$(HEXAGON_SDK_ROOT_V69)/rtos/qurt/computev69/include/qurt -I$(HEXAGON_SDK_ROOT_V69)/rtos/qurt/computev69/include/posix -I$(HEXAGON_SDK_ROOT_V69)/incs -I$(HEXAGON_SDK_ROOT_V69)/incs/stddef
HEXAGON_CXX_FLAGS_V73 := $(HEXAGON_CXX_FLAGS) -mv73 -I$(HEXAGON_SDK_ROOT_V73)/rtos/qurt/computev73/include/qurt -I$(HEXAGON_SDK_ROOT_V73)/rtos/qurt/computev73/include/posix -I$(HEXAGON_SDK_ROOT_V73)/incs -I$(HEXAGON_SDK_ROOT_V73)/incs/stddef
HEXAGON_CXX_FLAGS_V75 := $(HEXAGON_CXX_FLAGS) -mv75 -I$(HEXAGON_SDK_ROOT_V75)/rtos/qurt/computev75/include/qurt -I$(HEXAGON_SDK_ROOT_V75)/rtos/qurt/computev75/include/posix -I$(HEXAGON_SDK_ROOT_V75)/incs -I$(HEXAGON_SDK_ROOT_V75)/incs/stddef


HEXAGON_CXX_V68 := $(HEXAGON_SDK_ROOT_V68)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V68)/Tools/bin/hexagon-clang++
HEXAGON_CXX_V69 := $(HEXAGON_SDK_ROOT_V69)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V69)/Tools/bin/hexagon-clang++
HEXAGON_CXX_V73 := $(HEXAGON_SDK_ROOT_V73)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V73)/Tools/bin/hexagon-clang++
HEXAGON_CXX_V75 := $(HEXAGON_SDK_ROOT_V75)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION_V75)/Tools/bin/hexagon-clang++


HEX_LDFLAGS =
hexagon_objs =
#============================================================================================================
# Setup compiler, compiler instructions and linker for aarch64
AARCH64_C__FLAGS = -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include -ffast-math -DUSE_OS_LINUX -DANDROID
AARCH64_CXX_FLAGS = $(COMMON_CXX_FLAGS) $(AARCH64_C__FLAGS) -fomit-frame-pointer -Wno-invalid-offsetof  -Wno-unused-variable -Wno-unused-parameter -Wno-missing-braces -Wno-sign-compare -Wno-unused-private-field -Wno-unused-variable -Wno-ignored-qualifiers -Wno-missing-field-initializers
ARM_CLANG_OPTS =--target=aarch64-none-linux-android21 --sysroot=$(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/sysroot -stdlib=libc++ -static-libstdc++
AARCH64_CXX = $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ $(ARM_CLANG_OPTS)
AARCH64_LDFLAGS = -L$(QNN_TARGET_LIB) -lQnnHtp -lQnnHtpPrepare
aarch64_objs =
#============================================================================================================
# Setup targets and goals

htp_x86: X86_BUILD

htp_v68: HEXAGON_BUILD_V68

htp_v69: HEXAGON_BUILD_V69

htp_v73: HEXAGON_BUILD_V73

htp_v75: HEXAGON_BUILD_V75



htp_aarch64: AARCH64_BUILD

AARCH64_BUILD: $(WORK)/aarch64-android/$(LIBRARY_NAME)

HEXAGON_BUILD_V68:  $(WORK)/hexagon-v68/$(LIBRARY_NAME)

HEXAGON_BUILD_V69:  $(WORK)/hexagon-v69/$(LIBRARY_NAME)

HEXAGON_BUILD_V73:  $(WORK)/hexagon-v73/$(LIBRARY_NAME)

HEXAGON_BUILD_V75:  $(WORK)/hexagon-v75/$(LIBRARY_NAME)



X86_BUILD: $(WORK)/x86_64-linux-clang/$(LIBRARY_NAME)


$(WORK)/x86_64-linux-clang $(WORK)/hexagon-v68 $(WORK)/hexagon-v69 $(WORK)/hexagon-v73 $(WORK)/hexagon-v75 $(WORK)/aarch64-android:
	@mkdir -p $@/ops

# x86
$(WORK)/x86_64-linux-clang/%.o: %.cpp | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/x86_64-linux-clang/$(LIBRARY_NAME): $(patsubst %,$(WORK)/x86_64-linux-clang/%,$(OP_OBJS)) | $(HFILES)
	$(X86_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(X86_LDFLAGS)

# v68
$(WORK)/hexagon-v68/%.o: %.cpp | $(WORK)/hexagon-v68
	$(HEXAGON_CXX_V68) $(HEXAGON_CXX_FLAGS_V68) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v68/$(LIBRARY_NAME): $(patsubst %,$(WORK)/hexagon-v68/%,$(OP_OBJS)) | $(HFILES)
	$(HEXAGON_CXX_V68) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)

# v69
$(WORK)/hexagon-v69/%.o: %.cpp | $(WORK)/hexagon-v69
	$(HEXAGON_CXX_V69) $(HEXAGON_CXX_FLAGS_V69) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v69/$(LIBRARY_NAME): $(patsubst %,$(WORK)/hexagon-v69/%,$(OP_OBJS)) | $(HFILES)
	$(HEXAGON_CXX_V69) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)

# v73
$(WORK)/hexagon-v73/%.o: %.cpp | $(WORK)/hexagon-v73
	$(HEXAGON_CXX_V73) $(HEXAGON_CXX_FLAGS_V73) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v73/$(LIBRARY_NAME): $(patsubst %,$(WORK)/hexagon-v73/%,$(OP_OBJS)) | $(HFILES)
	$(HEXAGON_CXX_V73) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)

# v75
$(WORK)/hexagon-v75/%.o: %.cpp | $(WORK)/hexagon-v75
	$(HEXAGON_CXX_V75) $(HEXAGON_CXX_FLAGS_V75) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/hexagon-v75/$(LIBRARY_NAME): $(patsubst %,$(WORK)/hexagon-v75/%,$(OP_OBJS)) | $(HFILES)
	$(HEXAGON_CXX_V75) -fPIC -std=c++17 -g -shared -o $@ $^ $(HEX_LDFLAGS)



# aarch64
$(WORK)/aarch64-android/%.o: %.cpp | $(WORK)/aarch64-android
	$(AARCH64_CXX) $(AARCH64_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -c $< -o $@

$(WORK)/aarch64-android/$(LIBRARY_NAME): $(patsubst %,$(WORK)/aarch64-android/%,$(OP_OBJS)) | $(HFILES)
	$(AARCH64_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(AARCH64_LDFLAGS)

clean:
	-rm -rf $(WORK)
