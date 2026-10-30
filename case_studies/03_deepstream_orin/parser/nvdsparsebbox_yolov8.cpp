// Bounding-box parser for an Ultralytics YOLOv8 ONNX export inside DeepStream's nvinfer.
//
// The export's single output is [84, 8400] per image: four box values
// (centre x, centre y, width, height in network input pixels) followed by 80
// class scores, one column per candidate box. There is no objectness term.
// For every column the parser takes the best class; if its score clears that
// class's pre-cluster threshold the box is converted to a top-left corner
// rectangle, clipped to the network input, and handed to nvinfer, which runs
// its own NMS (cluster-mode 2) and maps the boxes back to the source frame
// (maintain-aspect-ratio and symmetric-padding are honoured by nvinfer, not
// here). Written for bench2field case study 03; no third-party code.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYoloV8(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                     NvDsInferNetworkInfo const& networkInfo,
                                     NvDsInferParseDetectionParams const& detectionParams,
                                     std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    if (layer.inferDims.numDims < 2) return false;
    const int channels = layer.inferDims.d[0];      // 84
    const int anchors = layer.inferDims.d[1];       // 8400
    const int numClasses = channels - 4;
    if (numClasses <= 0 || layer.dataType != FLOAT) return false;
    const float* data = static_cast<const float*>(layer.buffer);
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const int classesWithThreshold = static_cast<int>(detectionParams.perClassPreclusterThreshold.size());

    for (int i = 0; i < anchors; ++i) {
        int best = 0;
        float bestScore = data[(4 + 0) * anchors + i];
        for (int c = 1; c < numClasses; ++c) {
            const float s = data[(4 + c) * anchors + i];
            if (s > bestScore) { bestScore = s; best = c; }
        }
        const float threshold = best < classesWithThreshold ? detectionParams.perClassPreclusterThreshold[best] : 0.25f;
        if (bestScore < threshold) continue;
        const float cx = data[0 * anchors + i], cy = data[1 * anchors + i];
        const float w = data[2 * anchors + i], h = data[3 * anchors + i];
        float left = std::max(0.0f, cx - w * 0.5f);
        float top = std::max(0.0f, cy - h * 0.5f);
        float right = std::min(netW, cx + w * 0.5f);
        float bottom = std::min(netH, cy + h * 0.5f);
        if (right - left <= 1.0f || bottom - top <= 1.0f) continue;
        NvDsInferParseObjectInfo obj{};
        obj.classId = static_cast<unsigned int>(best);
        obj.detectionConfidence = bestScore;
        obj.left = left; obj.top = top; obj.width = right - left; obj.height = bottom - top;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
