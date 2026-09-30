package dashscope

import (
	"encoding/base64"
	"net/http"
	"strings"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// DefaultDashScopeBaseURL is the China (Beijing) region endpoint. The
// Singapore region (dashscope-intl.aliyuncs.com) is selectable via
// network_config.base_url.
const DefaultDashScopeBaseURL = "https://dashscope.aliyuncs.com"

// Native and compatible-mode endpoint paths. Chat and transcription ride the
// OpenAI-compatible surface (transcription as chat with input_audio); TTS and
// images speak the native DashScope protocol, which compatible-mode does not
// cover.
const (
	nativeMultimodalGenerationPath    = "/api/v1/services/aigc/multimodal-generation/generation"
	nativeTextToImagePath             = "/api/v1/services/aigc/text2image/image-synthesis"
	nativeImageToImagePath            = "/api/v1/services/aigc/image2image/image-synthesis"
	nativeTasksPath                   = "/api/v1/tasks/"
	compatibleModeChatCompletionsPath = "/compatible-mode/v1/chat/completions"
)

// DashScope native protocol headers.
const (
	headerXDashScopeSSE   = "X-DashScope-SSE"   // switches a native endpoint into SSE streaming
	headerXDashScopeAsync = "X-DashScope-Async" // marks a native submit as an asynchronous task
	headerValueEnable     = "enable"            // both native protocol switches take this value
)

// dashScopeTaskPollingInterval is the interval between task status polls for
// the asynchronous image APIs.
const dashScopeTaskPollingInterval = 3 * time.Second

// isAsyncImageModel reports whether the model runs on the legacy asynchronous
// task API (wan*/wanx* families) rather than the synchronous
// multimodal-generation endpoint (qwen-image*/z-image*).
func isAsyncImageModel(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(m, "wan")
}

// isImageToImageModel reports whether the async model is an image-editing one
// (wan*-i2i*), whose tasks submit to the image2image endpoint instead of the
// text2image one.
func isImageToImageModel(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	return isAsyncImageModel(m) && strings.Contains(m, "-i2i-")
}

// dashScopeImageSize converts an OpenAI-style "1024x1024" size into
// DashScope's "1024*1024" notation. Already-native sizes pass through.
func dashScopeImageSize(size string) string {
	if size == "" {
		return ""
	}
	return strings.NewReplacer("x", "*", "X", "*").Replace(size)
}

// copyExtraParams copies a Bifrost ExtraParams map so promoting known keys and
// deleting them never mutates the caller's request.
func copyExtraParams(extra map[string]interface{}) map[string]interface{} {
	if extra == nil {
		return nil
	}
	copied := make(map[string]interface{}, len(extra))
	for k, v := range extra {
		copied[k] = v
	}
	return copied
}

// mergeExtra merges unpromoted ExtraParams into an already-marshaled object,
// never overwriting named fields. Output is key-sorted for byte-stable
// requests (prompt-cache-friendly).
func mergeExtra(base []byte, extra map[string]interface{}) ([]byte, error) {
	if len(extra) == 0 {
		return base, nil
	}
	var merged map[string]interface{}
	if err := schemas.Unmarshal(base, &merged); err != nil {
		return nil, err
	}
	for k, v := range extra {
		if _, exists := merged[k]; !exists {
			merged[k] = v
		}
	}
	return schemas.MarshalSorted(merged)
}

// imageDataURL inlines raw image bytes as a base64 data URL so they can ride a
// multimodal content part alongside URL-referenced images.
func imageDataURL(data []byte) string {
	if len(data) == 0 {
		return ""
	}
	return "data:" + http.DetectContentType(data) + ";base64," + base64.StdEncoding.EncodeToString(data)
}
