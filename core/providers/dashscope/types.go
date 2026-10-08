package dashscope

import (
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// DashScopeMultimodalContentPart is one part of a multimodal message content
// array. Exactly one field carries a value, mirroring the upstream
// {"text": ...} / {"image": ...} / {"audio": ...} part objects.
type DashScopeMultimodalContentPart struct {
	Text  string `json:"text,omitempty"`
	Image string `json:"image,omitempty"`
	Audio string `json:"audio,omitempty"`
}

// DashScopeMultimodalMessage is one message of a multimodal-generation input.
type DashScopeMultimodalMessage struct {
	Role    string                           `json:"role"`
	Content []DashScopeMultimodalContentPart `json:"content"`
}

// DashScopeTTSInput is the input block for the qwen-tts family served by
// multimodal-generation. ExtraParams not promoted to named fields are merged
// into the marshaled object so qwen-specific knobs (e.g. rate, volume) ride
// along without polluting the top level.
type DashScopeTTSInput struct {
	Text         string `json:"text"`
	Voice        string `json:"voice,omitempty"`
	Format       string `json:"format,omitempty"`
	SampleRate   int    `json:"sample_rate,omitempty"`
	Instructions string `json:"instructions,omitempty"`

	Extra map[string]interface{} `json:"-"`
}

// MarshalJSON merges Extra into the input object after the named fields.
func (i *DashScopeTTSInput) MarshalJSON() ([]byte, error) {
	type inputAlias DashScopeTTSInput
	base, err := schemas.MarshalSorted((*inputAlias)(i))
	if err != nil {
		return nil, err
	}
	return mergeExtra(base, i.Extra)
}

// DashScopeTTSRequest is the request body for TTS on multimodal-generation.
type DashScopeTTSRequest struct {
	Model string            `json:"model"`
	Input DashScopeTTSInput `json:"input"`
}

// GetExtraParams satisfies providerUtils.RequestBodyWithExtraParams. It
// returns nil because MarshalJSON already merges extras into the nested input
// object; handing them to the passthrough layer as well would duplicate them
// at the JSON root.
func (r *DashScopeTTSRequest) GetExtraParams() map[string]interface{} {
	return nil
}

// DashScopeImageInput is the input block for the synchronous image models
// (qwen-image*/z-image*): a single user message whose content parts carry the
// prompt text plus any reference images.
type DashScopeImageInput struct {
	Messages []DashScopeMultimodalMessage `json:"messages"`
}

// DashScopeImageParameters carries the generation knobs for image calls.
// ExtraParams not promoted to named fields are merged into the marshaled
// object (watermark, prompt_extend, ...).
type DashScopeImageParameters struct {
	N              *int   `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	Seed           *int   `json:"seed,omitempty"`
	NegativePrompt string `json:"negative_prompt,omitempty"`
	PromptExtend   *bool  `json:"prompt_extend,omitempty"`
	Watermark      *bool  `json:"watermark,omitempty"`

	Extra map[string]interface{} `json:"-"`
}

// MarshalJSON merges Extra into the parameters object after the named fields.
func (p *DashScopeImageParameters) MarshalJSON() ([]byte, error) {
	type parametersAlias DashScopeImageParameters
	base, err := schemas.MarshalSorted((*parametersAlias)(p))
	if err != nil {
		return nil, err
	}
	return mergeExtra(base, p.Extra)
}

// DashScopeSyncImageRequest is the request body for the synchronous image
// models on multimodal-generation.
type DashScopeSyncImageRequest struct {
	Model      string                    `json:"model"`
	Input      DashScopeImageInput       `json:"input"`
	Parameters *DashScopeImageParameters `json:"parameters,omitempty"`
}

// GetExtraParams satisfies providerUtils.RequestBodyWithExtraParams. It
// returns nil because MarshalJSON already merges extras into the nested
// parameters object; handing them to the passthrough layer as well would
// duplicate them at the JSON root.
func (r *DashScopeSyncImageRequest) GetExtraParams() map[string]interface{} {
	return nil
}

// DashScopeTaskInput is the input block for the legacy asynchronous image task
// API. Unlike the synchronous models, negative_prompt lives in input, and the
// i2i models (wan*-i2i*) take their reference images here as an images array.
type DashScopeTaskInput struct {
	Prompt         string   `json:"prompt"`
	Images         []string `json:"images,omitempty"`
	NegativePrompt string   `json:"negative_prompt,omitempty"`
}

// DashScopeTaskRequest is the submit body for wanx*/wan2.x image tasks.
type DashScopeTaskRequest struct {
	Model      string                    `json:"model"`
	Input      DashScopeTaskInput        `json:"input"`
	Parameters *DashScopeImageParameters `json:"parameters,omitempty"`
}

// GetExtraParams satisfies providerUtils.RequestBodyWithExtraParams. It
// returns nil because MarshalJSON already merges extras into the nested
// parameters object; handing them to the passthrough layer as well would
// duplicate them at the JSON root.
func (r *DashScopeTaskRequest) GetExtraParams() map[string]interface{} {
	return nil
}

// DashScopeASRInputAudio carries the audio payload of an input_audio content
// part on the native ASR path: a data URL (data:audio/<mime>;base64,...).
type DashScopeASRInputAudio struct {
	Data string `json:"data"`
}

// DashScopeASRContentPart is one part of a native ASR message's content
// array. The qwen-audio-3.x transcription path sends exactly one input_audio
// part per request.
type DashScopeASRContentPart struct {
	Type       string                  `json:"type"`
	InputAudio *DashScopeASRInputAudio `json:"input_audio,omitempty"`
}

// DashScopeASRMessage is one message of a native ASR input.
type DashScopeASRMessage struct {
	Role    string                    `json:"role"`
	Content []DashScopeASRContentPart `json:"content"`
}

// DashScopeASRInput is the input block for qwen-audio-3.x ASR on
// multimodal-generation: a single user message carrying the audio part.
type DashScopeASRInput struct {
	Messages []DashScopeASRMessage `json:"messages"`
}

// DashScopeASRParameters is the parameters block for the native ASR call. The
// audio container format is mandatory upstream; the remaining ASR knobs
// (language_hints, sample_rate, vocabulary, vocabulary_id,
// speaker_diarization_enabled, keep_dialect, ...) ride Extra and are merged
// into the marshaled object.
type DashScopeASRParameters struct {
	Format string `json:"format"`

	Extra map[string]interface{} `json:"-"`
}

// MarshalJSON merges Extra into the parameters object after the named fields.
func (p *DashScopeASRParameters) MarshalJSON() ([]byte, error) {
	type parametersAlias DashScopeASRParameters
	base, err := schemas.MarshalSorted((*parametersAlias)(p))
	if err != nil {
		return nil, err
	}
	return mergeExtra(base, p.Extra)
}

// DashScopeASRRequest is the request body for the qwen-audio-3.x ASR models on
// multimodal-generation.
type DashScopeASRRequest struct {
	Model      string                  `json:"model"`
	Input      DashScopeASRInput       `json:"input"`
	Parameters *DashScopeASRParameters `json:"parameters,omitempty"`
}

// GetExtraParams satisfies providerUtils.RequestBodyWithExtraParams. It
// returns nil because MarshalJSON already merges extras into the nested
// parameters object; handing them to the passthrough layer as well would
// duplicate them at the JSON root.
func (r *DashScopeASRRequest) GetExtraParams() map[string]interface{} {
	return nil
}

// DashScopeASRUsage carries the native ASR usage: token counts plus the audio
// duration in seconds.
type DashScopeASRUsage struct {
	InputTokens  *int     `json:"input_tokens,omitempty"`
	OutputTokens *int     `json:"output_tokens,omitempty"`
	TotalTokens  *int     `json:"total_tokens,omitempty"`
	Duration     *float64 `json:"duration,omitempty"` // seconds
}

// DashScopeASRResponse is the multimodal-generation response for the
// qwen-audio-3.x ASR models. The observed wire shape nests the result twice
// (output.output.text); output.text is kept as a defensive fallback for
// sibling shapes that answer flat.
type DashScopeASRResponse struct {
	Output struct {
		Output *struct {
			Text string `json:"text,omitempty"`
		} `json:"output,omitempty"`
		Text string `json:"text,omitempty"`
	} `json:"output"`
	Usage     *DashScopeASRUsage `json:"usage,omitempty"`
	RequestID string             `json:"request_id,omitempty"`
	Code      string             `json:"code,omitempty"`
	Message   string             `json:"message,omitempty"`
}

// DashScopeUsage carries both the token usage (TTS) and image-count usage
// (image APIs); the wire only fills whichever applies.
type DashScopeUsage struct {
	InputTokens      int `json:"input_tokens,omitempty"`
	OutputTokens     int `json:"output_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
	ImageCount       int `json:"image_count,omitempty"`
	OutputImageCount int `json:"output_image_count,omitempty"`
	InputImageCount  int `json:"input_image_count,omitempty"`
	OutputWidth      int `json:"output_width,omitempty"`
	OutputHeight     int `json:"output_height,omitempty"`
}

// DashScopeErrorBody is the native error envelope: flat
// {code, message, request_id} on HTTP failures.
type DashScopeErrorBody struct {
	Code      string `json:"code"`
	Type      string `json:"type"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

// DashScopeTTSResponse is the multimodal-generation response for TTS. It is
// also the per-event shape of the SSE stream: non-streaming returns the
// synthesized audio as a download URL, streaming fills audio.data with base64
// chunks and terminates with finish_reason set plus usage.
type DashScopeTTSResponse struct {
	Output struct {
		Audio struct {
			URL       string `json:"url,omitempty"`
			Data      string `json:"data,omitempty"`
			ExpiresAt int64  `json:"expires_at,omitempty"`
		} `json:"audio"`
		FinishReason *string `json:"finish_reason,omitempty"`
	} `json:"output"`
	Usage     *DashScopeUsage `json:"usage,omitempty"`
	RequestID string          `json:"request_id,omitempty"`
	Code      string          `json:"code,omitempty"`
	Message   string          `json:"message,omitempty"`
}

// DashScopeMultimodalResponse is the multimodal-generation response for the
// synchronous image models: image results come back as content parts.
type DashScopeMultimodalResponse struct {
	Output struct {
		Choices []struct {
			FinishReason string `json:"finish_reason,omitempty"`
			Message      struct {
				Role    string                           `json:"role,omitempty"`
				Content []DashScopeMultimodalContentPart `json:"content,omitempty"`
			} `json:"message"`
		} `json:"choices,omitempty"`
	} `json:"output"`
	Usage     *DashScopeUsage `json:"usage,omitempty"`
	RequestID string          `json:"request_id,omitempty"`
}

// DashScopeTaskStatus is the lifecycle of an asynchronous DashScope task.
type DashScopeTaskStatus string

const (
	TaskStatusPending   DashScopeTaskStatus = "PENDING"
	TaskStatusRunning   DashScopeTaskStatus = "RUNNING"
	TaskStatusSucceeded DashScopeTaskStatus = "SUCCEEDED"
	TaskStatusFailed    DashScopeTaskStatus = "FAILED"
	TaskStatusCanceled  DashScopeTaskStatus = "CANCELED"
	TaskStatusUnknown   DashScopeTaskStatus = "UNKNOWN"
)

// DashScopeTaskSubmitResponse is the synchronous answer to an async task
// submit: a task_id to poll.
type DashScopeTaskSubmitResponse struct {
	Output struct {
		TaskID     string `json:"task_id,omitempty"`
		TaskStatus string `json:"task_status,omitempty"`
	} `json:"output"`
	RequestID string `json:"request_id,omitempty"`
	Code      string `json:"code,omitempty"`
	Message   string `json:"message,omitempty"`
}

// DashScopeTaskResponse is one task-status poll answer.
type DashScopeTaskResponse struct {
	Output struct {
		TaskID     string `json:"task_id,omitempty"`
		TaskStatus string `json:"task_status,omitempty"`
		Results    []struct {
			URL     string `json:"url,omitempty"`
			Code    string `json:"code,omitempty"`    // per-result moderation failure, e.g. DataInspectionFailed
			Message string `json:"message,omitempty"` // detail for a per-result failure
		} `json:"results,omitempty"`
		Code    string `json:"code,omitempty"`
		Message string `json:"message,omitempty"`
	} `json:"output"`
	Usage     *DashScopeUsage `json:"usage,omitempty"`
	RequestID string          `json:"request_id,omitempty"`
	Code      string          `json:"code,omitempty"`
	Message   string          `json:"message,omitempty"`
}
