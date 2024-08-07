//=============================================================================
//
//  Copyright (c) 2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN LPAI Context components
 */

#ifndef QNN_LPAI_CONTEXT_H
#define QNN_LPAI_CONTEXT_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t option;
  void* config;
} QnnLpaiContext_CustomConfig_t;

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif
