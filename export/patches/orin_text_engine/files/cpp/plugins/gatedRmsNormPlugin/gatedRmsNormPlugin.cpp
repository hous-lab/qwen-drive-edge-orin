/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GatedRmsNorm plugin: y = w * bf16_round-ish(RMSNorm(o)) * silu(z), all FP16 I/O.
 *
 * Replaces the Myelin-compiled RMSNorm + silu(z) + mul chain that follows the
 * gated_delta_net plugin on every Qwen3.5 GDN layer. One warp per row
 * (row = one (token, v-head) pair); math is the FP16 twin of the SCAN-P
 * epilogue kernel (deltanet_persistent/epilogue_gated.cu), which was verified
 * 99.9992% bitwise against the HF Qwen3_5RMSNormGated reference on x86:
 *   fp32 sum of squares -> rsqrt(mean + eps) -> normalize in fp32
 *   -> round to input dtype -> * weight (fp32 math) -> * silu(z in fp32)
 *   -> single round to input dtype.
 */
#include "gatedRmsNormPlugin.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/gatedRmsNorm/gatedRmsNormKernel.h"
#include "plugins/utils/pluginUtils.h"

#include <mutex>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kGRN_PLUGIN_VERSION{"1"};
constexpr char const* kGRN_PLUGIN_NAME{"gated_rmsnorm"};
} // namespace

PluginFieldCollection GatedRmsNormPluginCreator::mFieldCollection{};
std::vector<PluginField> GatedRmsNormPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GatedRmsNormPluginCreator);

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

GatedRmsNormPlugin::GatedRmsNormPlugin(std::string const& name, float eps)
    : mLayerName(name), mEps(eps)
{
}

GatedRmsNormPlugin::GatedRmsNormPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    mEps = parsePluginScalarField<float>("eps", fc).value_or(1.e-6f);
}

GatedRmsNormPlugin::~GatedRmsNormPlugin() = default;

// ---------------------------------------------------------------------------
// IPluginV3
// ---------------------------------------------------------------------------

IPluginCapability* GatedRmsNormPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuildV2*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* GatedRmsNormPlugin::clone() noexcept
{
    try
    {
        auto* p = new GatedRmsNormPlugin(mLayerName, mEps);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (...)
    {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// IPluginV3OneCore
// ---------------------------------------------------------------------------

char const* GatedRmsNormPlugin::getPluginName() const noexcept
{
    return kGRN_PLUGIN_NAME;
}

char const* GatedRmsNormPlugin::getPluginVersion() const noexcept
{
    return kGRN_PLUGIN_VERSION;
}

char const* GatedRmsNormPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedRmsNormPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuild
// ---------------------------------------------------------------------------

int32_t GatedRmsNormPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t GatedRmsNormPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        outputTypes[kOUT_Y_IDX] = inputTypes[kIN_O_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t GatedRmsNormPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    try
    {
        outputs[kOUT_Y_IDX] = inputs[kIN_O_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool GatedRmsNormPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    auto checkFP16 = [](PluginTensorDesc const& desc)
    {
        return desc.type == DataType::kHALF && desc.format == TensorFormat::kLINEAR;
    };
    if (pos < nbInputs)
    {
        return checkFP16(inOut[pos].desc);
    }
    int32_t outPos = pos - nbInputs;
    return outPos == kOUT_Y_IDX && checkFP16(inOut[pos].desc);
}

int32_t GatedRmsNormPlugin::configurePlugin(DynamicPluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

size_t GatedRmsNormPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t GatedRmsNormPlugin::getAliasedInput(int32_t /* outputIndex */) noexcept
{
    // No in-place aliasing: the kernel writes a fresh output tensor.
    return -1;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime
// ---------------------------------------------------------------------------

int32_t GatedRmsNormPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs,
    void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        auto const& oDesc = inputDesc[kIN_O_IDX];
        int dv = oDesc.dims.d[oDesc.dims.nbDims - 1];
        int64_t rows = 1;
        for (int i = 0; i < oDesc.dims.nbDims - 1; ++i)
        {
            rows *= oDesc.dims.d[i];
        }
        ELLM_CHECK(dv > 0 && rows > 0, "GatedRmsNorm enqueue: non-positive extent");
        // z must match o's shape; weight must be [dv]
        auto const& zDesc = inputDesc[kIN_Z_IDX];
        ELLM_CHECK(zDesc.dims.nbDims == oDesc.dims.nbDims, "GatedRmsNorm: o/z rank mismatch");

        kernel::gatedRmsNormForward(inputs[kIN_O_IDX], inputs[kIN_Z_IDX], inputs[kIN_WEIGHT_IDX],
            outputs[kOUT_Y_IDX], rows, dv, mEps, stream);
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("GatedRmsNorm enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t GatedRmsNormPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

IPluginV3* GatedRmsNormPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* GatedRmsNormPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mEpsField = mEps;
    mDataToSerialize.emplace_back("eps", &mEpsField, PluginFieldType::kFLOAT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

GatedRmsNormPluginCreator::GatedRmsNormPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* GatedRmsNormPluginCreator::getPluginName() const noexcept
{
    return kGRN_PLUGIN_NAME;
}

char const* GatedRmsNormPluginCreator::getPluginVersion() const noexcept
{
    return kGRN_PLUGIN_VERSION;
}

PluginFieldCollection const* GatedRmsNormPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* GatedRmsNormPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedRmsNormPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

IPluginV3* GatedRmsNormPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new GatedRmsNormPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("GatedRmsNormPluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
