package dashscope

import (
	"encoding/base64"
	"net/http"
	"path/filepath"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// Transcription performs a speech-to-text request via DashScope's
// OpenAI-compatible chat completions endpoint. DashScope has no Whisper-style
// /audio/transcriptions surface (the compatible-mode path 404s); qwen3-asr-flash
// is reached through /compatible-mode/v1/chat/completions with the audio riding
// a single user message as an input_audio content block. The upstream rejects
// a bare base64 payload with a 400 ("The provided URL does not appear to be
// valid"), so the data is wrapped as a data URL (data:audio/<mime>;base64,...)
// alongside the format field. The assistant's reply text is the transcript.
func (provider *DashScopeProvider) Transcription(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostTranscriptionRequest) (*schemas.BifrostTranscriptionResponse, *schemas.BifrostError) {
	if request == nil || request.Input == nil || len(request.Input.File) == 0 {
		// Input == nil is how large-payload passthrough arrives (core skips
		// materialization by design); the ASR chat path needs the audio bytes,
		// so that mode cannot be served here.
		return nil, providerUtils.NewBifrostOperationError("audio input is required for DashScope transcription (large-payload passthrough is not supported on this path)", nil)
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
