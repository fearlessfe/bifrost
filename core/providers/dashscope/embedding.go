package dashscope

import (
	"github.com/maximhq/bifrost/core/providers/openai"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// compatibleModeEmbeddingsPath is the OpenAI-compatible embeddings endpoint
// (text-embedding-v3/v4, qwen3.7-text-embedding*).
const compatibleModeEmbeddingsPath = "/compatible-mode/v1/embeddings"

// Embedding generates embeddings via DashScope's OpenAI-compatible surface
// (/compatible-mode), reusing the shared openai handler. DashScope-only knobs
// (dimension, output_dtype, ...) ride the request's ExtraParams and are
// flattened onto the body upstream.
func (provider *DashScopeProvider) Embedding(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostEmbeddingRequest) (*schemas.BifrostEmbeddingResponse, *schemas.BifrostError) {
	// DashScope-only knobs ride ExtraParams; opt into flattening provider-side
	// (same as deepseek/sgl/vllm) so callers don't need the passthrough header.
	ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
	return openai.HandleOpenAIEmbeddingRequest(
		ctx,
		provider.client,
		provider.networkConfig.BaseURL+providerUtils.GetPathFromContext(ctx, compatibleModeEmbeddingsPath),
		request,
		openai.BearerAuthHeader(key),
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		nil,
		nil,
		provider.logger,
	)
}
