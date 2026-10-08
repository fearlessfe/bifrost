package dashscope

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// Transcription performs a speech-to-text request, dispatched by model family.
//
// qwen3-asr-flash rides DashScope's OpenAI-compatible chat completions endpoint
// (DashScope has no Whisper-style /audio/transcriptions surface; the
// compatible-mode path 404s). The audio travels as an input_audio content block
// on a single user message — a bare base64 payload is rejected upstream ("The
// provided URL does not appear to be valid"), so it is wrapped as a data URL —
// and the assistant's reply text is the transcript.
//
// The qwen-audio-3.x ASR flash models are not hosted on the compatible-mode
// surface (3.1 404s with "Unsupported model ... for OpenAI compatibility mode",
// 3.0 rejects the request with UNSUPPORTED_FORMAT); they run on the native
// multimodal-generation endpoint instead (nativeASRTranscription). The
// realtime/streaming and filetrans variants of that family are not served at
// all: they live on the WebSocket realtime and asynchronous task APIs.
func (provider *DashScopeProvider) Transcription(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostTranscriptionRequest) (*schemas.BifrostTranscriptionResponse, *schemas.BifrostError) {
	if request == nil || request.Input == nil || len(request.Input.File) == 0 {
		// Input == nil is how large-payload passthrough arrives (core skips
		// materialization by design); the ASR chat path needs the audio bytes,
		// so that mode cannot be served here.
		return nil, providerUtils.NewBifrostOperationError("audio input is required for DashScope transcription (large-payload passthrough is not supported on this path)", nil)
	}

	if isNativeASRModel(request.Model) {
		return provider.nativeASRTranscription(ctx, key, request)
	}
	if isUnsupportedASRVariant(request.Model) {
		return nil, providerUtils.NewBifrostOperationError(fmt.Sprintf("dashscope model %q is not supported for transcription: the realtime/streaming and filetrans variants of the qwen-audio-3.x ASR family run on the WebSocket realtime and asynchronous task APIs, which Bifrost does not serve", request.Model), nil)
	}

	format := dashScopeAudioFormat(request)

	chatRequest := &schemas.BifrostChatRequest{
		Provider: request.Provider,
		Model:    request.Model,
		Input: []schemas.ChatMessage{{
			Role: schemas.ChatMessageRoleUser,
			Content: &schemas.ChatMessageContent{
				ContentBlocks: []schemas.ChatContentBlock{{
					Type: schemas.ChatContentBlockTypeInputAudio,
					InputAudio: &schemas.ChatInputAudio{
						Data:   dashScopeAudioDataURL(request.Input.File, format),
						Format: format,
					},
				}},
			},
		}},
	}

	// DashScope's non-standard ASR knobs (asr_options.language, enable_itn,
	// ...) ride ExtraParams and are flattened onto the chat body upstream when
	// the caller opts into extra-params passthrough, same as the chat
	// delegation. The standard `language` parameter is folded into
	// asr_options.language here; an explicit extra_params.asr_options block
	// wins on conflicts.
	if request.Params != nil {
		extraParams := request.Params.ExtraParams
		if request.Params.Language != nil && *request.Params.Language != "" {
			extraParams = dashScopeASRLanguageParams(extraParams, *request.Params.Language)
		}
		if len(extraParams) > 0 {
			chatRequest.Params = &schemas.ChatParameters{ExtraParams: extraParams}
		}
	}

	chatResponse, err := provider.ChatCompletion(ctx, key, chatRequest)
	if err != nil {
		return nil, err
	}

	text, ok := dashScopeTranscriptText(chatResponse)
	if !ok {
		return nil, providerUtils.NewBifrostOperationError("dashscope returned no transcription text", nil)
	}

	response := &schemas.BifrostTranscriptionResponse{
		Text:        text,
		Task:        schemas.Ptr("transcribe"),
		ExtraFields: chatResponse.ExtraFields,
	}
	if usage := chatResponse.Usage; usage != nil {
		response.Usage = &schemas.TranscriptionUsage{
			Type:         "tokens",
			InputTokens:  schemas.Ptr(usage.PromptTokens),
			OutputTokens: schemas.Ptr(usage.CompletionTokens),
			TotalTokens:  schemas.Ptr(usage.TotalTokens),
		}
		if details := usage.PromptTokensDetails; details != nil {
			response.Usage.InputTokenDetails = &schemas.TranscriptionUsageInputTokenDetails{
				TextTokens:  details.TextTokens,
				AudioTokens: details.AudioTokens,
			}
		}
	}
	return response, nil
}

// dashScopeASRLanguageParams returns a copy of extra with
// asr_options.language set from the standard transcription language
// parameter. Caller maps are never mutated, and an explicit
// extra_params.asr_options.language takes precedence.
func dashScopeASRLanguageParams(extra map[string]interface{}, language string) map[string]interface{} {
	out := make(map[string]interface{}, len(extra)+1)
	for k, v := range extra {
		out[k] = v
	}
	asrOptions := map[string]interface{}{}
	if existing, ok := out["asr_options"].(map[string]interface{}); ok {
		for k, v := range existing {
			asrOptions[k] = v
		}
	}
	if _, ok := asrOptions["language"]; !ok {
		asrOptions["language"] = language
	}
	out["asr_options"] = asrOptions
	return out
}

// dashScopeAudioFormat resolves the audio container format for the
// input_audio block: an explicit file_format parameter wins, otherwise the
// filename extension is used. Returns nil when neither is available so the
// upstream can auto-detect.
func dashScopeAudioFormat(request *schemas.BifrostTranscriptionRequest) *string {
	if request.Params != nil && request.Params.Format != nil && *request.Params.Format != "" {
		return schemas.Ptr(strings.ToLower(strings.TrimPrefix(*request.Params.Format, ".")))
	}
	if request.Input != nil {
		if ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(request.Input.Filename)), "."); ext != "" {
			return schemas.Ptr(ext)
		}
	}
	return nil
}

// dashScopeAudioDataURL wraps raw audio bytes as a data URL, which is the only
// base64 form the upstream accepts. The MIME type comes from the resolved
// format; when no format is known, the bytes are content-sniffed instead (same
// pattern as imageDataURL in utils.go).
func dashScopeAudioDataURL(data []byte, format *string) string {
	mime := ""
	if format != nil && *format != "" {
		mime = dashScopeAudioMIME(*format)
	} else {
		mime = http.DetectContentType(data)
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
}

// dashScopeAudioMIME maps an audio container format to the MIME type used in
// the data URL. Unknown formats fall back to audio/<format>.
func dashScopeAudioMIME(format string) string {
	switch strings.ToLower(format) {
	case "mp3":
		return "audio/mpeg"
	case "wav", "wave":
		return "audio/wav"
	case "flac":
		return "audio/flac"
	case "m4a", "mp4":
		return "audio/mp4"
	case "aac":
		return "audio/aac"
	case "ogg", "oga":
		return "audio/ogg"
	case "opus":
		return "audio/opus"
	case "amr":
		return "audio/amr"
	case "wma":
		return "audio/x-ms-wma"
	default:
		return "audio/" + format
	}
}

// dashScopeTranscriptText extracts the transcript from the chat response: the
// first choice's assistant content is the recognized text. ok is false when
// the response carries no assistant message at all.
func dashScopeTranscriptText(response *schemas.BifrostChatResponse) (string, bool) {
	if response == nil || len(response.Choices) == 0 {
		return "", false
	}
	choice := response.Choices[0]
	if choice.ChatNonStreamResponseChoice == nil || choice.Message == nil || choice.Message.Content == nil {
		return "", false
	}
	if choice.Message.Content.ContentStr != nil {
		return *choice.Message.Content.ContentStr, true
	}
	var sb strings.Builder
	for _, block := range choice.Message.Content.ContentBlocks {
		if block.Text != nil {
			sb.WriteString(*block.Text)
		}
	}
	return sb.String(), true
}

// nativeASRTranscription runs the qwen-audio-3.x ASR models on the native
// multimodal-generation endpoint: one JSON POST with X-DashScope-SSE disabled,
// the audio as an input_audio part inside input.messages, and the (upstream-
// mandatory) audio format in parameters.format.
func (provider *DashScopeProvider) nativeASRTranscription(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostTranscriptionRequest) (*schemas.BifrostTranscriptionResponse, *schemas.BifrostError) {
	format := dashScopeAudioFormat(request)
	if format == nil {
		// parameters.format is required upstream; fail locally with a clear
		// error instead of shipping a body that 400s with UNSUPPORTED_FORMAT.
		return nil, providerUtils.NewBifrostOperationError("audio format is required for DashScope qwen-audio-3.x transcription: pass file_format or use a filename with an extension", nil)
	}

	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			return ToDashScopeASRRequest(request), nil
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

	req.SetRequestURI(provider.networkConfig.BaseURL + nativeMultimodalGenerationPath)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	req.Header.Set(headerXDashScopeSSE, headerValueDisable)
	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, schemas.DashScope) {
		req.SetBody(jsonData)
	}

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, providerUtils.EnrichError(ctx, parseDashScopeError(resp), jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	body, bifrostErr := decodeResponseBody(ctx, resp)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	var asrResp DashScopeASRResponse
	if err := sonic.Unmarshal(body, &asrResp); err != nil {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("failed to parse DashScope transcription response", err), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	bifrostResponse, ok := ToBifrostASRResponse(&asrResp)
	if !ok {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("dashscope returned no transcription text", nil), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}
	bifrostResponse.ExtraFields = schemas.BifrostResponseExtraFields{
		Latency:                 latency.Milliseconds(),
		ProviderResponseHeaders: providerUtils.ExtractProviderResponseHeaders(resp),
	}

	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		providerUtils.ParseAndSetRawRequest(&bifrostResponse.ExtraFields, jsonData)
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		var rawResponse interface{}
		if err := sonic.Unmarshal(body, &rawResponse); err != nil {
			rawResponse = string(body)
		}
		bifrostResponse.ExtraFields.RawResponse = rawResponse
	}

	return bifrostResponse, nil
}

// ToDashScopeASRRequest builds the native multimodal-generation body for the
// qwen-audio-3.x ASR models: a single user message whose content is one
// input_audio part carrying a data URL, plus a parameters block holding the
// audio format and the ASR knobs. The standard language parameter folds into
// parameters.language_hints (an explicit extra_params.language_hints wins);
// remaining extra params merge into parameters. asr_options is a
// compatible-mode-only knob bag and never crosses over.
func ToDashScopeASRRequest(bifrostReq *schemas.BifrostTranscriptionRequest) *DashScopeASRRequest {
	if bifrostReq == nil || bifrostReq.Input == nil {
		return nil
	}

	format := dashScopeAudioFormat(bifrostReq)

	req := &DashScopeASRRequest{
		Model: bifrostReq.Model,
		Input: DashScopeASRInput{
			Messages: []DashScopeASRMessage{{
				Role: "user",
				Content: []DashScopeASRContentPart{{
					Type:       "input_audio",
					InputAudio: &DashScopeASRInputAudio{Data: dashScopeAudioDataURL(bifrostReq.Input.File, format)},
				}},
			}},
		},
	}

	parameters := &DashScopeASRParameters{}
	if format != nil {
		parameters.Format = *format
	}

	if bifrostReq.Params != nil {
		extra := copyExtraParams(bifrostReq.Params.ExtraParams)
		delete(extra, "asr_options")
		if bifrostReq.Params.Language != nil && *bifrostReq.Params.Language != "" {
			if _, ok := extra["language_hints"]; !ok {
				extra["language_hints"] = []string{*bifrostReq.Params.Language}
			}
		}
		parameters.Extra = extra
	}

	req.Parameters = parameters
	return req
}

// ToBifrostASRResponse maps the native ASR response onto the Bifrost
// transcription shape. The observed wire nests the result twice
// (output.output.text); output.text is the defensive fallback. ok is false
// when neither level carries text. Token usage maps to the "tokens" usage
// type; the native duration (seconds) maps to both Usage.Seconds and the
// response-level Duration, and carries the "duration" usage type when it is
// the only accounting the upstream returned.
func ToBifrostASRResponse(resp *DashScopeASRResponse) (*schemas.BifrostTranscriptionResponse, bool) {
	if resp == nil {
		return nil, false
	}

	text := ""
	if resp.Output.Output != nil {
		text = resp.Output.Output.Text
	}
	if text == "" {
		text = resp.Output.Text
	}
	if text == "" {
		return nil, false
	}

	bifrostResp := &schemas.BifrostTranscriptionResponse{
		Text: text,
		Task: schemas.Ptr("transcribe"),
	}

	if usage := resp.Usage; usage != nil {
		u := &schemas.TranscriptionUsage{}
		if usage.InputTokens != nil || usage.OutputTokens != nil || usage.TotalTokens != nil {
			u.Type = "tokens"
			u.InputTokens = usage.InputTokens
			u.OutputTokens = usage.OutputTokens
			u.TotalTokens = usage.TotalTokens
		} else if usage.Duration != nil {
			u.Type = "duration"
		}
		if usage.Duration != nil {
			u.Seconds = usage.Duration
			bifrostResp.Duration = usage.Duration
		}
		if u.Type != "" {
			bifrostResp.Usage = u
		}
	}

	return bifrostResp, true
}
