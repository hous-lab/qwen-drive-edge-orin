/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused RMSNorm + silu gate for the Qwen3.5 GDN output (replaces the
 * RMSNorm + silu(z) + mul elementwise chain Myelin would otherwise compile).
 *
 * API targets the ORIN lineage (TRT 10.4: IPluginV3OneBuildV2 /
 * IPluginCreatorV3One, setPluginNamespace not virtual-overridden), matching
 * gatedDeltaNetPlugin.h in the same tree.
 */
#pragma once

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

// Inputs:  o (FP16 [b, s, hv, v] — GDN plugin output, last dim = head dim),
//          z (FP16 [b, s, hv, v] — per-head gate from in_proj),
//          weight (FP16 [v]).
// Output:  y (FP16, same shape as o).
constexpr int32_t kIN_O_IDX{0};
constexpr int32_t kIN_Z_IDX{1};
constexpr int32_t kIN_WEIGHT_IDX{2};
constexpr int32_t kOUT_Y_IDX{0};

constexpr int32_t kNUM_INPUTS{3};
constexpr int32_t kNUM_OUTPUTS{1};

class GatedRmsNormPlugin : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuildV2,
                           public nvinfer1::IPluginV3OneRuntime
{
public:
    GatedRmsNormPlugin(std::string const& name, float eps);
    GatedRmsNormPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);
    GatedRmsNormPlugin() = delete;
    ~GatedRmsNormPlugin() override;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t getAliasedInput(int32_t outputIndex) noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mLayerName;
    std::string mNamespace;
    float mEps{1.e-6f};

    // Serialization fields (used by getFieldsToSerialize)
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
    float mEpsField{1.e-6f};
};

class GatedRmsNormPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    GatedRmsNormPluginCreator();
    ~GatedRmsNormPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    std::string mNamespace;
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace plugins
} // namespace trt_edgellm
