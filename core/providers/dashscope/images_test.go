package dashscope

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

func TestIsAsyncImageModel(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"wan2.2-t2i-flash", true},
		{"wanx2.1-t2i-turbo", true},
		{"wan2.5-i2i-preview", true},
		{"WAN2.2-T2I-FLASH", true}, // case-insensitive
		{"qwen-image", false},
		{"qwen-image-edit-plus", false},
		{"z-image-turbo", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := isAsyncImageModel(tc.model); got != tc.want {
			t.Errorf("isAsyncImageModel(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestDashScopeImageSize(t *testing.T) {
	cases := []struct{ in, want string }{
		{"1024x1024", "1024*1024"},
		{"1664x928", "1664*928"},
		{"1024X1024", "1024*1024"}, // uppercase separator is accepted
		{"1024*1024", "1024*1024"}, // native notation passes through
		{"", ""},
	}
	for _, tc := range cases {
		if got := dashScopeImageSize(tc.in); got != tc.want {
			t.Errorf("dashScopeImageSize(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestIsImageToImageModel(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"wan2.5-i2i-preview", true},
		{"WAN2.5-I2I-PREVIEW", true}, // case-insensitive
		{"wan2.2-t2i-flash", false},
		{"wanx2.1-t2i-turbo", false},
		{"qwen-image-edit", false}, // sync family is never i2i-task
		{"", false},
	}
	for _, tc := range cases {
		if got := isImageToImageModel(tc.model); got != tc.want {
			t.Errorf("isImageToImageModel(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestToDashScopeSyncImageRequest(t *testing.T) {
	size := "1024x1024"
	n := 2
	req, err := ToDashScopeSyncImageRequest(&schemas.BifrostImageGenerationRequest{
		Model: "qwen-image",
		Input: &schemas.ImageGenerationInput{Prompt: "画一只猫"},
		Params: &schemas.ImageGenerationParameters{
			Size:           &size,
			N:              &n,
			NegativePrompt: schemas.Ptr("模糊"),
			ExtraParams:    map[string]interface{}{"watermark": false, "prompt_extend": true},
		},
	})
	if err != nil {
		t.Fatalf("unexpected conversion error: %v", err)
	}

	if req == nil {
		t.Fatal("expected request, got nil")
	}
	if len(req.Input.Messages) != 1 || req.Input.Messages[0].Role != "user" {
		t.Fatalf("expected one user message, got %+v", req.Input.Messages)
	}
	content := req.Input.Messages[0].Content
	if len(content) != 1 || content[0].Text != "画一只猫" {
		t.Fatalf("expected single text part, got %+v", content)
	}
	if req.Parameters == nil {
		t.Fatal("expected parameters block")
	}
	if req.Parameters.Size != "1024*1024" {
		t.Errorf("size = %q, want %q", req.Parameters.Size, "1024*1024")
	}
	if req.Parameters.N == nil || *req.Parameters.N != 2 {
		t.Errorf("n = %v, want 2", req.Parameters.N)
	}
	if req.Parameters.NegativePrompt != "模糊" {
		t.Errorf("negative_prompt = %q, want %q", req.Parameters.NegativePrompt, "模糊")
	}
	if req.Parameters.Watermark == nil || *req.Parameters.Watermark {
		t.Errorf("watermark should be promoted as false, got %v", req.Parameters.Watermark)
	}
	if req.Parameters.PromptExtend == nil || !*req.Parameters.PromptExtend {
		t.Errorf("prompt_extend should be promoted as true, got %v", req.Parameters.PromptExtend)
	}
	if len(req.Parameters.Extra) != 0 {
		t.Errorf("promoted extras should be consumed, got %v", req.Parameters.Extra)
	}
}

func TestToDashScopeSyncImageRequestReferenceImagesPrecedeText(t *testing.T) {
	req, err := ToDashScopeSyncImageRequest(&schemas.BifrostImageGenerationRequest{
		Model: "z-image-turbo",
		Input: &schemas.ImageGenerationInput{Prompt: "make it night"},
		Params: &schemas.ImageGenerationParameters{
			InputImages: []string{"https://example.com/ref.png"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected conversion error: %v", err)
	}

	content := req.Input.Messages[0].Content
	if len(content) != 2 {
		t.Fatalf("expected image part + text part, got %+v", content)
	}
	if content[0].Image != "https://example.com/ref.png" {
		t.Errorf("reference image must come before the text part, got %+v", content[0])
	}
	if content[1].Text != "make it night" {
		t.Errorf("text part = %+v", content[1])
	}
}

func TestToDashScopeTaskImageRequestNegativePromptInInput(t *testing.T) {
	n := 4
	req, err := ToDashScopeTaskImageRequest(&schemas.BifrostImageGenerationRequest{
		Model: "wan2.2-t2i-flash",
		Input: &schemas.ImageGenerationInput{Prompt: "a cat"},
		Params: &schemas.ImageGenerationParameters{
			N:              &n,
			NegativePrompt: schemas.Ptr("blurry"),
			Size:           schemas.Ptr("1024x1024"),
		},
	})
	if err != nil {
		t.Fatalf("unexpected conversion error: %v", err)
	}

	if req.Input.Prompt != "a cat" {
		t.Errorf("prompt = %q", req.Input.Prompt)
	}
	// Legacy task API carries negative_prompt in input, not parameters.
	if req.Input.NegativePrompt != "blurry" {
		t.Errorf("input.negative_prompt = %q, want %q", req.Input.NegativePrompt, "blurry")
	}
	if req.Parameters == nil || req.Parameters.NegativePrompt != "" {
		t.Errorf("parameters.negative_prompt must stay empty on the task API, got %+v", req.Parameters)
	}
	if req.Parameters == nil || req.Parameters.N == nil || *req.Parameters.N != 4 {
		t.Errorf("parameters.n = %v, want 4", req.Parameters)
	}
	if req.Parameters == nil || req.Parameters.Size != "1024*1024" {
		t.Errorf("parameters.size = %+v, want %q", req.Parameters, "1024*1024")
	}
}

func TestToDashScopeImageEditRequestInlineDataURL(t *testing.T) {
	// Full 8-byte PNG signature so http.DetectContentType resolves image/png.
	png := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	req := ToDashScopeImageEditRequest(&schemas.BifrostImageEditRequest{
		Model: "qwen-image-edit",
		Input: &schemas.ImageEditInput{
			Images: []schemas.ImageInput{
				{URL: "https://example.com/in.png"},
				{Image: png},
			},
			Prompt: "remove the background",
		},
	})

	content := req.Input.Messages[0].Content
	if len(content) != 3 {
		t.Fatalf("expected 2 image parts + 1 text part, got %+v", content)
	}
	if content[0].Image != "https://example.com/in.png" {
		t.Errorf("URL input should pass through, got %q", content[0].Image)
	}
	if !strings.HasPrefix(content[1].Image, "data:image/png;base64,") {
		t.Errorf("byte input should become a data URL, got %q", content[1].Image)
	}
	if _, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(content[1].Image, "data:image/png;base64,")); err != nil {
		t.Errorf("data URL payload must be valid base64: %v", err)
	}
	if content[2].Text != "remove the background" {
		t.Errorf("text part = %+v", content[2])
	}
}

func TestToBifrostSyncImageResponse(t *testing.T) {
	raw := []byte(`{
		"output": {
			"choices": [{
				"finish_reason": "stop",
				"message": {
					"role": "assistant",
					"content": [
						{"text": "a better prompt"},
						{"image": "https://example.com/1.png"},
						{"image": "https://example.com/2.png"}
					]
				}
			}]
		},
		"usage": {"image_count": 2, "input_tokens": 10, "output_tokens": 20},
		"request_id": "req-1"
	}`)

	var mmResp DashScopeMultimodalResponse
	if err := schemas.Unmarshal(raw, &mmResp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	resp := ToBifrostSyncImageResponse(&mmResp, "qwen-image")
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 images, got %+v", resp.Data)
	}
	if resp.Data[0].URL != "https://example.com/1.png" || resp.Data[0].Index != 0 {
		t.Errorf("data[0] = %+v", resp.Data[0])
	}
	if resp.Data[1].Index != 1 {
		t.Errorf("data[1].Index = %d, want 1", resp.Data[1].Index)
	}
	if resp.Data[0].RevisedPrompt != "a better prompt" {
		t.Errorf("revised prompt = %q, want %q", resp.Data[0].RevisedPrompt, "a better prompt")
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 20 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Model != "qwen-image" {
		t.Errorf("model = %q", resp.Model)
	}
}

func TestToBifrostTaskImageResponseSkipsModerationEntries(t *testing.T) {
	raw := []byte(`{
		"output": {
			"task_id": "task-1",
			"task_status": "SUCCEEDED",
			"results": [
				{"url": "https://example.com/ok.png"},
				{"url": "https://example.com/blocked.png", "code": "DataInspectionFailed"}
			]
		},
		"usage": {"image_count": 2}
	}`)

	var taskResp DashScopeTaskResponse
	if err := schemas.Unmarshal(raw, &taskResp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	resp := ToBifrostTaskImageResponse(&taskResp, "wan2.2-t2i-flash")
	if len(resp.Data) != 1 {
		t.Fatalf("moderation-failed entries must be skipped, got %+v", resp.Data)
	}
	if resp.Data[0].URL != "https://example.com/ok.png" {
		t.Errorf("data[0].URL = %q", resp.Data[0].URL)
	}
}

func TestTaskImageRequestExtraParamsPreservedForMerge(t *testing.T) {
	req, err := ToDashScopeTaskImageRequest(&schemas.BifrostImageGenerationRequest{
		Model: "wanx2.1-t2i-turbo",
		Input: &schemas.ImageGenerationInput{Prompt: "a cat"},
		Params: &schemas.ImageGenerationParameters{
			ExtraParams: map[string]interface{}{"watermark": false, "prompt_extend": true, "stylize": float64(3)},
		},
	})
	if err != nil {
		t.Fatalf("unexpected conversion error: %v", err)
	}

	if req.Parameters.Extra["stylize"] != float64(3) {
		t.Errorf("unpromoted extra must ride the merge, got %v", req.Parameters.Extra)
	}
	if req.Parameters.Watermark == nil || req.Parameters.PromptExtend == nil {
		t.Errorf("promoted knobs missing: %+v", req.Parameters)
	}
}

func TestToDashScopeTaskImageRequestMapsInputImages(t *testing.T) {
	req, err := ToDashScopeTaskImageRequest(&schemas.BifrostImageGenerationRequest{
		Model: "wan2.5-i2i-preview",
		Input: &schemas.ImageGenerationInput{Prompt: "blend these"},
		Params: &schemas.ImageGenerationParameters{
			InputImages: []string{"https://example.com/a.png", "https://example.com/b.png"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected conversion error: %v", err)
	}

	if len(req.Input.Images) != 2 || req.Input.Images[0] != "https://example.com/a.png" || req.Input.Images[1] != "https://example.com/b.png" {
		t.Fatalf("reference images must ride input.images on the task API, got %+v", req.Input.Images)
	}
	// The i2i family submits to the image2image endpoint, not text2image.
	if !isImageToImageModel(req.Model) {
		t.Errorf("wan2.5-i2i-preview must route to the image2image endpoint")
	}
}

func TestToDashScopeTaskImageRequestNegativePromptFromExtraParams(t *testing.T) {
	req, err := ToDashScopeTaskImageRequest(&schemas.BifrostImageGenerationRequest{
		Model: "wan2.2-t2i-flash",
		Input: &schemas.ImageGenerationInput{Prompt: "a cat"},
		Params: &schemas.ImageGenerationParameters{
			ExtraParams: map[string]interface{}{"negative_prompt": "blurry"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected conversion error: %v", err)
	}

	// negative_prompt promoted out of ExtraParams must land in input, since the
	// task API ignores parameters.negative_prompt.
	if req.Input.NegativePrompt != "blurry" {
		t.Errorf("input.negative_prompt = %q, want %q", req.Input.NegativePrompt, "blurry")
	}
	if req.Parameters == nil || req.Parameters.NegativePrompt != "" {
		t.Errorf("parameters.negative_prompt must stay empty on the task API, got %+v", req.Parameters)
	}
}

func TestDashScopeImageRequestsGetExtraParamsNil(t *testing.T) {
	syncReq, err := ToDashScopeSyncImageRequest(&schemas.BifrostImageGenerationRequest{
		Model: "qwen-image",
		Input: &schemas.ImageGenerationInput{Prompt: "a cat"},
		Params: &schemas.ImageGenerationParameters{
			ExtraParams: map[string]interface{}{"stylize": float64(3)},
		},
	})
	if err != nil {
		t.Fatalf("unexpected conversion error: %v", err)
	}
	if extra := syncReq.GetExtraParams(); extra != nil {
		t.Errorf("sync request GetExtraParams must return nil, got %v", extra)
	}

	taskReq, err := ToDashScopeTaskImageRequest(&schemas.BifrostImageGenerationRequest{
		Model: "wan2.2-t2i-flash",
		Input: &schemas.ImageGenerationInput{Prompt: "a cat"},
		Params: &schemas.ImageGenerationParameters{
			ExtraParams: map[string]interface{}{"stylize": float64(3)},
		},
	})
	if err != nil {
		t.Fatalf("unexpected conversion error: %v", err)
	}
	if extra := taskReq.GetExtraParams(); extra != nil {
		t.Errorf("task request GetExtraParams must return nil, got %v", extra)
	}

	// The final body must carry extras only inside parameters, never at the
	// JSON root (passthrough must not double-merge them there).
	data, err := schemas.MarshalSorted(taskReq)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var body map[string]interface{}
	if err := schemas.Unmarshal(data, &body); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	for key := range body {
		if key != "model" && key != "input" && key != "parameters" {
			t.Errorf("unexpected root-level key %q; extras must live inside parameters only", key)
		}
	}
	params, ok := body["parameters"].(map[string]interface{})
	if !ok || params["stylize"] != float64(3) {
		t.Errorf("stylize should ride inside parameters, got %s", data)
	}
}

func TestFirstTaskImageFailureErrorAllResultsBlocked(t *testing.T) {
	raw := []byte(`{
		"output": {
			"task_id": "task-1",
			"task_status": "SUCCEEDED",
			"results": [
				{"code": "DataInspectionFailed", "message": "Output data may contain inappropriate content"},
				{"code": "DataInspectionFailed", "message": "Output data may contain inappropriate content"}
			]
		},
		"usage": {"image_count": 0}
	}`)

	var taskResp DashScopeTaskResponse
	if err := schemas.Unmarshal(raw, &taskResp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	bifrostErr := FirstTaskImageFailureError(&taskResp)
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error when every result failed moderation")
	}
	if !strings.Contains(bifrostErr.Error.Message, "DataInspectionFailed") {
		t.Errorf("error message must carry the moderation code, got %q", bifrostErr.Error.Message)
	}
	if !strings.Contains(bifrostErr.Error.Message, "inappropriate content") {
		t.Errorf("error message must carry the upstream detail, got %q", bifrostErr.Error.Message)
	}
	if bifrostErr.Error.Code == nil || *bifrostErr.Error.Code != "DataInspectionFailed" {
		t.Errorf("error code = %v, want DataInspectionFailed", bifrostErr.Error.Code)
	}
}

func TestFirstTaskImageFailureErrorCodeOnlyFallback(t *testing.T) {
	raw := []byte(`{
		"output": {
			"task_id": "task-1",
			"task_status": "SUCCEEDED",
			"results": [{"code": "DataInspectionFailed"}]
		}
	}`)

	var taskResp DashScopeTaskResponse
	if err := schemas.Unmarshal(raw, &taskResp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	bifrostErr := FirstTaskImageFailureError(&taskResp)
	if bifrostErr == nil || !strings.Contains(bifrostErr.Error.Message, "DataInspectionFailed") {
		t.Fatalf("code-only failure must still surface the code, got %+v", bifrostErr)
	}
}

func TestFirstTaskImageFailureErrorNilCases(t *testing.T) {
	if got := FirstTaskImageFailureError(nil); got != nil {
		t.Errorf("nil response should give nil error, got %+v", got)
	}

	// A surviving image means partial success: no error, the failed entries are
	// just skipped by ToBifrostTaskImageResponse.
	partial := []byte(`{
		"output": {
			"task_status": "SUCCEEDED",
			"results": [
				{"code": "DataInspectionFailed", "message": "blocked"},
				{"url": "https://example.com/ok.png"}
			]
		}
	}`)
	var partialResp DashScopeTaskResponse
	if err := schemas.Unmarshal(partial, &partialResp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if got := FirstTaskImageFailureError(&partialResp); got != nil {
		t.Errorf("partial success should give nil error, got %+v", got)
	}

	// No failure codes at all (e.g. results empty or malformed): nothing to
	// report, the caller's generic "no images" error stands.
	var emptyResp DashScopeTaskResponse
	if got := FirstTaskImageFailureError(&emptyResp); got != nil {
		t.Errorf("no failure codes should give nil error, got %+v", got)
	}
}

// pollTestLogger is a minimal no-op logger for provider construction in tests.
type pollTestLogger struct{}

func (l *pollTestLogger) Debug(msg string, args ...any)                     {}
func (l *pollTestLogger) Info(msg string, args ...any)                      {}
func (l *pollTestLogger) Warn(msg string, args ...any)                      {}
func (l *pollTestLogger) Error(msg string, args ...any)                     {}
func (l *pollTestLogger) Fatal(msg string, args ...any)                     {}
func (l *pollTestLogger) SetLevel(level schemas.LogLevel)                   {}
func (l *pollTestLogger) SetOutputType(outputType schemas.LoggerOutputType) {}
func (l *pollTestLogger) LogHTTPRequest(level schemas.LogLevel, msg string) schemas.LogEventBuilder {
	return schemas.NoopLogEvent
}

// newImageTestProvider builds a provider pointed at the given mock server.
func newImageTestProvider(t *testing.T, baseURL string) *DashScopeProvider {
	t.Helper()
	provider, err := NewDashScopeProvider(&schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        baseURL,
			DefaultRequestTimeoutInSeconds: 30,
		},
	}, &pollTestLogger{})
	if err != nil {
		t.Fatalf("NewDashScopeProvider: %v", err)
	}
	return provider
}

func imageTestKey() schemas.Key {
	return schemas.Key{Value: *schemas.NewSecretVar("test-key")}
}

func TestPollImageTaskSucceededFirstPoll(t *testing.T) {
	var polls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&polls, 1)
		if r.Method != http.MethodGet || !strings.HasPrefix(r.URL.Path, nativeTasksPath) {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"output": {
				"task_id": "task-1",
				"task_status": "SUCCEEDED",
				"results": [{"url": "https://example.com/img.png"}]
			},
			"usage": {"image_count": 1},
			"request_id": "req-1"
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	taskResp, _, bifrostErr := provider.pollImageTask(ctx, imageTestKey(), "task-1", false)
	if bifrostErr != nil {
		t.Fatalf("pollImageTask failed: %v", bifrostErr.Error.Message)
	}
	if got := atomic.LoadInt32(&polls); got != 1 {
		t.Errorf("polls = %d, want 1 (SUCCEEDED on first poll)", got)
	}
	if DashScopeTaskStatus(taskResp.Output.TaskStatus) != TaskStatusSucceeded {
		t.Errorf("task_status = %q, want SUCCEEDED", taskResp.Output.TaskStatus)
	}
	if len(taskResp.Output.Results) != 1 || taskResp.Output.Results[0].URL != "https://example.com/img.png" {
		t.Errorf("results = %+v", taskResp.Output.Results)
	}
}

func TestPollImageTaskPendingThenSucceeded(t *testing.T) {
	var polls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := atomic.AddInt32(&polls, 1)
		w.Header().Set("Content-Type", "application/json")
		if attempt == 1 {
			_, _ = w.Write([]byte(`{"output": {"task_id": "task-1", "task_status": "PENDING"}}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"output": {
				"task_id": "task-1",
				"task_status": "SUCCEEDED",
				"results": [{"url": "https://example.com/img.png"}]
			}
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	taskResp, _, bifrostErr := provider.pollImageTask(ctx, imageTestKey(), "task-1", false)
	if bifrostErr != nil {
		t.Fatalf("pollImageTask failed: %v", bifrostErr.Error.Message)
	}
	if got := atomic.LoadInt32(&polls); got != 2 {
		t.Errorf("polls = %d, want 2 (PENDING then SUCCEEDED)", got)
	}
	if DashScopeTaskStatus(taskResp.Output.TaskStatus) != TaskStatusSucceeded {
		t.Errorf("task_status = %q, want SUCCEEDED", taskResp.Output.TaskStatus)
	}
}

func TestPollImageTaskFailedCarriesMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"output": {
				"task_id": "task-1",
				"task_status": "FAILED",
				"code": "ImageGeneratingFailed",
				"message": "allocation failed"
			}
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	taskResp, _, bifrostErr := provider.pollImageTask(ctx, imageTestKey(), "task-1", false)
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error for a FAILED task")
	}
	if taskResp != nil {
		t.Errorf("failed task must not return a response, got %+v", taskResp)
	}
	if !strings.Contains(bifrostErr.Error.Message, "allocation failed") {
		t.Errorf("error message = %q, want the upstream failure detail", bifrostErr.Error.Message)
	}
	if bifrostErr.Error.Code == nil || *bifrostErr.Error.Code != "ImageGeneratingFailed" {
		t.Errorf("error code = %v, want ImageGeneratingFailed", bifrostErr.Error.Code)
	}
}

func TestRetrieveImageTaskEmptyTaskID(t *testing.T) {
	var polls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&polls, 1)
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, _, bifrostErr := provider.retrieveImageTask(ctx, imageTestKey(), "", false)
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error for an empty task id")
	}
	if !strings.Contains(bifrostErr.Error.Message, "task_id") {
		t.Errorf("error message = %q, want it to name the task_id field", bifrostErr.Error.Message)
	}
	if got := atomic.LoadInt32(&polls); got != 0 {
		t.Errorf("empty task id must be rejected before any HTTP call, server saw %d requests", got)
	}
}

func TestPollImageTaskUnrecognizedStatus(t *testing.T) {
	var polls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&polls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"output": {"task_id": "task-1", "task_status": "PARTIAL_SUCCESS"}}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, _, bifrostErr := provider.pollImageTask(ctx, imageTestKey(), "task-1", false)
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error for an unrecognized task status")
	}
	if !strings.Contains(bifrostErr.Error.Message, "PARTIAL_SUCCESS") {
		t.Errorf("error message = %q, want it to name the unrecognized status", bifrostErr.Error.Message)
	}
	if got := atomic.LoadInt32(&polls); got != 1 {
		t.Errorf("polls = %d, want 1 (unrecognized status must fail fast, not poll until timeout)", got)
	}
}

func TestPollImageTaskHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"code": "InternalError", "message": "upstream exploded", "request_id": "req-1"}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, _, bifrostErr := provider.pollImageTask(ctx, imageTestKey(), "task-1", false)
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error for an HTTP 500 poll response")
	}
	if !strings.Contains(bifrostErr.Error.Message, "upstream exploded") {
		t.Errorf("error message = %q, want the upstream error message", bifrostErr.Error.Message)
	}
	if bifrostErr.Error.Code == nil || *bifrostErr.Error.Code != "InternalError" {
		t.Errorf("error code = %v, want InternalError", bifrostErr.Error.Code)
	}
	if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusInternalServerError {
		t.Errorf("status code = %v, want 500", bifrostErr.StatusCode)
	}
}

// TestImageGenerationSanitizesInputImages pins the InputImages normalization:
// the Bifrost contract allows URLs or bare base64, but DashScope accepts only
// URLs/data URLs, so every entry must pass through schemas.SanitizeImageURL on
// both the sync (multimodal-generation) and async task paths, and unsupported
// schemes must fail the conversion instead of reaching the wire.
func TestImageGenerationSanitizesInputImages(t *testing.T) {
	// 1x1 PNG.
	pngB64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

	newRequest := func(model string, images []string) *schemas.BifrostImageGenerationRequest {
		return &schemas.BifrostImageGenerationRequest{
			Provider: schemas.DashScope,
			Model:    model,
			Input:    &schemas.ImageGenerationInput{Prompt: "给这张图上色"},
			Params:   &schemas.ImageGenerationParameters{InputImages: images},
		}
	}

	t.Run("sync path wraps bare base64 as a data URL", func(t *testing.T) {
		var capturedBody map[string]interface{}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost || r.URL.Path != nativeMultimodalGenerationPath {
				t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			}
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, &capturedBody)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"output": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": [{"image": "https://example.com/out.png", "type": "image"}]}}]},
				"usage": {"image_count": 1}
			}`))
		}))
		defer server.Close()

		provider := newImageTestProvider(t, server.URL)
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

		if _, bifrostErr := provider.ImageGeneration(ctx, imageTestKey(), newRequest("qwen-image", []string{pngB64})); bifrostErr != nil {
			t.Fatalf("ImageGeneration failed: %v", bifrostErr.Error.Message)
		}
		output, _ := capturedBody["input"].(map[string]interface{})
		messages, _ := output["messages"].([]interface{})
		if len(messages) == 0 {
			t.Fatalf("no messages in outgoing body: %v", capturedBody)
		}
		content, _ := messages[0].(map[string]interface{})["content"].([]interface{})
		if len(content) == 0 {
			t.Fatalf("no content parts in outgoing body: %v", capturedBody)
		}
		image, _ := content[0].(map[string]interface{})["image"].(string)
		if !strings.HasPrefix(image, "data:image/png;base64,") {
			t.Errorf("image part = %q, want bare base64 wrapped as a data URL", image)
		}
	})

	t.Run("task path wraps bare base64 as a data URL", func(t *testing.T) {
		var capturedBody map[string]interface{}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			switch {
			case r.Method == http.MethodPost && r.URL.Path == nativeImageToImagePath:
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &capturedBody)
				_, _ = w.Write([]byte(`{"output": {"task_id": "task-1", "task_status": "SUCCEEDED", "results": [{"url": "https://example.com/out.png"}]}, "request_id": "r1"}`))
			case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, nativeTasksPath):
				_, _ = w.Write([]byte(`{"output": {"task_id": "task-1", "task_status": "SUCCEEDED", "results": [{"url": "https://example.com/out.png"}]}}`))
			default:
				t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			}
		}))
		defer server.Close()

		provider := newImageTestProvider(t, server.URL)
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

		if _, bifrostErr := provider.ImageGeneration(ctx, imageTestKey(), newRequest("wan2.5-i2i-preview", []string{pngB64})); bifrostErr != nil {
			t.Fatalf("ImageGeneration failed: %v", bifrostErr.Error.Message)
		}
		input, _ := capturedBody["input"].(map[string]interface{})
		images, _ := input["images"].([]interface{})
		if len(images) != 1 {
			t.Fatalf("expected one input image in the task submit body, got %v", capturedBody)
		}
		image, _ := images[0].(string)
		if !strings.HasPrefix(image, "data:image/png;base64,") {
			t.Errorf("input.images[0] = %q, want bare base64 wrapped as a data URL", image)
		}
	})

	t.Run("unsupported scheme fails the conversion before any HTTP call", func(t *testing.T) {
		var hits int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&hits, 1)
		}))
		defer server.Close()

		provider := newImageTestProvider(t, server.URL)
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

		_, bifrostErr := provider.ImageGeneration(ctx, imageTestKey(), newRequest("qwen-image", []string{"ftp://example.com/ref.png"}))
		if bifrostErr == nil || bifrostErr.Error == nil {
			t.Fatal("expected a conversion error for an unsupported URL scheme")
		}
		if !strings.Contains(bifrostErr.Error.Message, "input_images") {
			t.Errorf("error message = %q, want it to name the input_images entry", bifrostErr.Error.Message)
		}
		if got := atomic.LoadInt32(&hits); got != 0 {
			t.Errorf("unsupported scheme must be rejected before any HTTP call, server saw %d requests", got)
		}
	})
}
