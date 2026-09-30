package dashscope

import (
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// ToDashScopeTTSRequest maps a Bifrost speech request onto the qwen-tts
// multimodal-generation body. Voice, format and sample rate come from the
// standard speech params; qwen-specific knobs ride ExtraParams and are merged
// into the input object by DashScopeTTSInput.MarshalJSON.
func ToDashScopeTTSRequest(bifrostReq *schemas.BifrostSpeechRequest) *DashScopeTTSRequest {
	if bifrostReq == nil || bifrostReq.Input == nil {
		return nil
	}

	req := &DashScopeTTSRequest{
		Model: bifrostReq.Model,
		Input: DashScopeTTSInput{Text: bifrostReq.Input.Input},
	}

	if bifrostReq.Params == nil {
		return req
	}

	if bifrostReq.Params.VoiceConfig != nil && bifrostReq.Params.VoiceConfig.Voice != nil {
		req.Input.Voice = *bifrostReq.Params.VoiceConfig.Voice
	}
	if bifrostReq.Params.ResponseFormat != "" {
		req.Input.Format = bifrostReq.Params.ResponseFormat
	}
	if bifrostReq.Params.Instructions != "" {
		req.Input.Instructions = bifrostReq.Params.Instructions
	}

	req.Input.Extra = copyExtraParams(bifrostReq.Params.ExtraParams)
	for key, value := range bifrostReq.Params.ExtraParams {
		switch key {
		case "voice":
			if v, ok := schemas.SafeExtractString(value); ok {
				delete(req.Input.Extra, key)
				req.Input.Voice = v
			}
		case "format":
			if v, ok := schemas.SafeExtractString(value); ok {
				delete(req.Input.Extra, key)
				req.Input.Format = v
			}
		case "sample_rate":
			if v, ok := schemas.SafeExtractInt(value); ok {
				delete(req.Input.Extra, key)
				req.Input.SampleRate = v
			}
		case "instructions":
			if v, ok := schemas.SafeExtractString(value); ok {
				delete(req.Input.Extra, key)
				req.Input.Instructions = v
			}
		}
	}

	return req
}
