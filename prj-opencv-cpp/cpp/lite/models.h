//
// Created by DefTruth on 2021/8/8.
//

#ifndef LITE_AI_MODELS_H
#define LITE_AI_MODELS_H

#include "backend.h"

// ENABLE_ONNXRUNTIME
#ifdef ENABLE_ONNXRUNTIME

#include "lite/ort/core/ort_core.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/ort/cv/glint_arcface.h"

#endif

// Default Engine ONNXRuntime
namespace lite
{
  // mediapipe
  namespace mediapipe
  {
#ifdef BACKEND_ONNXRUNTIME
#endif
  }

  namespace cv
  {
#ifdef BACKEND_ONNXRUNTIME
    typedef ortcv::GlintArcFace _GlintArcFace;
#endif

    // 4. face recognition
    namespace faceid
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _GlintArcFace GlintArcFace; //
#endif

    }
  }

  namespace asr
  {
#ifdef BACKEND_ONNXRUNTIME
#endif
  }

  namespace nlp
  {
#ifdef BACKEND_ONNXRUNTIME
#endif
  }
}

// ONNXRuntime version
namespace lite
{
#ifdef ENABLE_ONNXRUNTIME
  namespace onnxruntime
  {
    // mediapipe
    namespace mediapipe
    {
    }

    namespace cv
    {
      typedef ortcv::GlintArcFace _ONNXGlintArcFace;

      // 4. face recognition
      namespace faceid
      {
        typedef _ONNXGlintArcFace GlintArcFace; //

      }

    }

  }
#endif
}

#endif //LITE_AI_MODELS_H
