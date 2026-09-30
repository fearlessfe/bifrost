package dashscope

import (
	"fmt"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// sanitizeDashScopeInputImages normalizes InputImages entries to what
// DashScope accepts: http(s) URLs or data URLs. Bare base64 (allowed by the
// Bifrost contract) is wrapped as a data URL; empty entries, unsupported
// schemes, and malformed data URLs are conversion errors.
func sanitizeDashScopeInputImages(images []string) ([]string, error) {
	if len(images) == 0 {
		return nil, nil
	}
	out := make([]string, 0, len(images))
	for i, img := range images {
		sanitized, err := schemas.SanitizeImageURL(img)
		if err != nil {
			// Caller-supplied input, not an internal fault: promote to a 400
			// with this message intact instead of a generic 500.
			return nil, providerUtils.InvalidRequestErrorf("input_images[%d]: %s", i, err.Error())
		}
		out = append(out, sanitized)
	}
	return out, nil
}

// imageParameterKnobs are the ExtraParams keys this provider promotes into
// named DashScope parameters fields; anything else rides the merge in
// DashScopeImageParameters.MarshalJSON.
func applyImageParameterKnobs(params *DashScopeImageParameters, extra map[string]interface{}) {
	if params.Extra == nil {
		params.Extra = copyExtraParams(extra)
	}
	if extra == nil {
		return
	}
	for key, value := range extra {
		switch key {
		case "n":
			if v, ok := schemas.SafeExtractInt(value); ok {
				delete(params.Extra, key)
				params.N = &v
			}
		case "size":
			if v, ok := schemas.SafeExtractString(value); ok {
				delete(params.Extra, key)
				params.Size = dashScopeImageSize(v)
			}
		case "seed":
			if v, ok := schemas.SafeExtractInt(value); ok {
				delete(params.Extra, key)
				params.Seed = &v
			}
		case "negative_prompt":
			if v, ok := schemas.SafeExtractString(value); ok {
				delete(params.Extra, key)
				params.NegativePrompt = v
			}
		case "prompt_extend":
			if v, ok := schemas.SafeExtractBool(value); ok {
				delete(params.Extra, key)
				params.PromptExtend = &v
			}
		case "watermark":
			if v, ok := schemas.SafeExtractBool(value); ok {
				delete(params.Extra, key)
				params.Watermark = &v
			}
		}
	}
}

// imageParamsFromBifrost builds the DashScope parameters block from a Bifrost
// image generation parameters struct.
func imageParamsFromBifrost(params *schemas.ImageGenerationParameters) *DashScopeImageParameters {
	if params == nil {
		return nil
	}
	out := &DashScopeImageParameters{}
	if params.N != nil {
		out.N = params.N
	}
	if params.Size != nil {
		out.Size = dashScopeImageSize(*params.Size)
	}
	if params.Seed != nil {
		out.Seed = params.Seed
	}
	if params.NegativePrompt != nil {
		out.NegativePrompt = *params.NegativePrompt
	}
	applyImageParameterKnobs(out, params.ExtraParams)
	return out
}

// ToDashScopeSyncImageRequest builds the multimodal-generation body for the
// synchronous image models (qwen-image*/z-image*): a single user message whose
// content parts carry reference images (if any) followed by exactly one text
// part, per the upstream contract. Reference images are normalized via
// SanitizeImageURL (bare base64 becomes a data URL).
func ToDashScopeSyncImageRequest(bifrostReq *schemas.BifrostImageGenerationRequest) (*DashScopeSyncImageRequest, error) {
	if bifrostReq == nil || bifrostReq.Input == nil {
		return nil, nil
	}

	var content []DashScopeMultimodalContentPart
	if bifrostReq.Params != nil {
		images, err := sanitizeDashScopeInputImages(bifrostReq.Params.InputImages)
		if err != nil {
			return nil, err
		}
		for _, img := range images {
			content = append(content, DashScopeMultimodalContentPart{Image: img})
		}
	}
	content = append(content, DashScopeMultimodalContentPart{Text: bifrostReq.Input.Prompt})

	req := &DashScopeSyncImageRequest{
		Model: bifrostReq.Model,
		Input: DashScopeImageInput{
			Messages: []DashScopeMultimodalMessage{{Role: "user", Content: content}},
		},
	}
	if params := imageParamsFromBifrost(bifrostReq.Params); params != nil {
		req.Parameters = params
	}
	return req, nil
}

// ToDashScopeImageEditRequest builds the multimodal-generation body for
// qwen-image-edit*: the input images (bytes are inlined as data URLs, URLs
// pass through) become image parts ahead of the edit instruction text.
func ToDashScopeImageEditRequest(bifrostReq *schemas.BifrostImageEditRequest) *DashScopeSyncImageRequest {
	if bifrostReq == nil || bifrostReq.Input == nil || len(bifrostReq.Input.Images) == 0 {
		return nil
	}

	content := make([]DashScopeMultimodalContentPart, 0, len(bifrostReq.Input.Images)+1)
	for _, img := range bifrostReq.Input.Images {
		if img.URL != "" {
			content = append(content, DashScopeMultimodalContentPart{Image: img.URL})
			continue
		}
		content = append(content, DashScopeMultimodalContentPart{Image: imageDataURL(img.Image)})
	}
	content = append(content, DashScopeMultimodalContentPart{Text: bifrostReq.Input.Prompt})

	req := &DashScopeSyncImageRequest{
		Model: bifrostReq.Model,
		Input: DashScopeImageInput{
			Messages: []DashScopeMultimodalMessage{{Role: "user", Content: content}},
		},
	}
	if bifrostReq.Params == nil {
		return req
	}
	params := &DashScopeImageParameters{}
	if bifrostReq.Params.N != nil {
		params.N = bifrostReq.Params.N
	}
	if bifrostReq.Params.Size != nil {
		params.Size = dashScopeImageSize(*bifrostReq.Params.Size)
	}
	if bifrostReq.Params.Seed != nil {
		params.Seed = bifrostReq.Params.Seed
	}
	if bifrostReq.Params.NegativePrompt != nil {
		params.NegativePrompt = *bifrostReq.Params.NegativePrompt
	}
	applyImageParameterKnobs(params, bifrostReq.Params.ExtraParams)
	req.Parameters = params
	return req
}

// ToDashScopeTaskImageRequest builds the submit body for the legacy
// asynchronous task API (wan*/wanx*). negative_prompt lives in input on this
// surface, not in parameters, and reference images for the i2i models
// (wan*-i2i*) ride input.images after SanitizeImageURL normalization (bare
// base64 becomes a data URL).
func ToDashScopeTaskImageRequest(bifrostReq *schemas.BifrostImageGenerationRequest) (*DashScopeTaskRequest, error) {
	if bifrostReq == nil || bifrostReq.Input == nil {
		return nil, nil
	}

	req := &DashScopeTaskRequest{
		Model: bifrostReq.Model,
		Input: DashScopeTaskInput{Prompt: bifrostReq.Input.Prompt},
	}
	if bifrostReq.Params != nil {
		if len(bifrostReq.Params.InputImages) > 0 {
			images, err := sanitizeDashScopeInputImages(bifrostReq.Params.InputImages)
			if err != nil {
				return nil, err
			}
			req.Input.Images = images
		}
		if bifrostReq.Params.NegativePrompt != nil {
			req.Input.NegativePrompt = *bifrostReq.Params.NegativePrompt
		}
		req.Parameters = imageParamsFromBifrost(bifrostReq.Params)
		// The task API reads negative_prompt from input only; parameters must
		// not carry a duplicate or the upstream rejects the call. Backfill
		// input first so a negative_prompt promoted out of ExtraParams is not
		// dropped with the clearing.
		if req.Parameters != nil {
			if req.Input.NegativePrompt == "" {
				req.Input.NegativePrompt = req.Parameters.NegativePrompt
			}
			req.Parameters.NegativePrompt = ""
		}
	}
	return req, nil
}

// ToBifrostSyncImageResponse converts a synchronous multimodal image response.
// Image parts become Data entries in order; the first text part (the
// prompt-extended rewrite, when prompt_extend is on) becomes RevisedPrompt.
func ToBifrostSyncImageResponse(resp *DashScopeMultimodalResponse, model string) *schemas.BifrostImageGenerationResponse {
	if resp == nil {
		return nil
	}
	data := make([]schemas.ImageData, 0)
	revisedPrompt := ""
	for _, choice := range resp.Output.Choices {
		for _, part := range choice.Message.Content {
			switch {
			case part.Image != "":
				data = append(data, schemas.ImageData{URL: part.Image, Index: len(data)})
			case part.Text != "" && revisedPrompt == "":
				revisedPrompt = part.Text
			}
		}
	}
	bifrostResp := &schemas.BifrostImageGenerationResponse{
		Model: model,
		Data:  data,
	}
	if revisedPrompt != "" && len(bifrostResp.Data) > 0 {
		bifrostResp.Data[0].RevisedPrompt = revisedPrompt
	}
	bifrostResp.Usage = dashScopeUsageToBifrost(resp.Usage)
	return bifrostResp
}

// ToBifrostTaskImageResponse converts a completed async task poll answer.
// Per-result entries carrying a code (e.g. DataInspectionFailed) are
// moderation failures, not images, and are skipped. When every result failed,
// the response's Data comes back empty and callers should prefer
// FirstTaskImageFailureError over a generic "no images" error so the
// moderation reason reaches the caller.
func ToBifrostTaskImageResponse(resp *DashScopeTaskResponse, model string) *schemas.BifrostImageGenerationResponse {
	if resp == nil {
		return nil
	}
	data := make([]schemas.ImageData, 0, len(resp.Output.Results))
	for _, result := range resp.Output.Results {
		if result.URL == "" || result.Code != "" {
			continue
		}
		data = append(data, schemas.ImageData{URL: result.URL, Index: len(data)})
	}
	bifrostResp := &schemas.BifrostImageGenerationResponse{
		Model: model,
		Data:  data,
	}
	bifrostResp.Usage = dashScopeUsageToBifrost(resp.Usage)
	return bifrostResp
}

// FirstTaskImageFailureError reports the first per-result failure on a
// completed async task (e.g. a DataInspectionFailed moderation rejection) as a
// BifrostError carrying the upstream code and message, so a fully-moderated
// task is recognizable as a content-moderation failure rather than an empty
// success. It returns nil when at least one result produced an image (partial
// failures are simply skipped by ToBifrostTaskImageResponse) or when no result
// carries a failure code.
func FirstTaskImageFailureError(resp *DashScopeTaskResponse) *schemas.BifrostError {
	if resp == nil {
		return nil
	}
	firstCode, firstMessage := "", ""
	for _, result := range resp.Output.Results {
		if result.URL != "" && result.Code == "" {
			return nil
		}
		if firstCode == "" && result.Code != "" {
			firstCode, firstMessage = result.Code, result.Message
		}
	}
	if firstCode == "" {
		return nil
	}
	message := fmt.Sprintf("dashscope image task results failed with %s", firstCode)
	if firstMessage != "" {
		message = fmt.Sprintf("%s: %s", message, firstMessage)
	}
	bifrostErr := providerUtils.NewBifrostOperationError(message, nil)
	bifrostErr.Error.Code = schemas.Ptr(firstCode)
	return bifrostErr
}

// dashScopeUsageToBifrost maps the token-shaped part of DashScope usage onto
// Bifrost's image usage. The image_count fields have no Bifrost destination;
// cost calculation runs off the backfilled request params instead.
func dashScopeUsageToBifrost(usage *DashScopeUsage) *schemas.ImageUsage {
	if usage == nil {
		return nil
	}
	return &schemas.ImageUsage{
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
		TotalTokens:  usage.TotalTokens,
	}
}
