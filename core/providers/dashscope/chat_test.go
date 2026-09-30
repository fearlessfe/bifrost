package dashscope

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// TestChatCompletionExtraParamsFlattened verifies that qwen-specific knobs in
// ExtraParams (enable_thinking, top_k, ...) reach the wire without the caller
// opting into extra-params passthrough — the provider sets
// BifrostContextKeyPassthroughExtraParams itself, same as deepseek/sgl/vllm.
func TestChatCompletionExtraParamsFlattened(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read error", http.StatusInternalServerError)
			return
		}
		if err := json.Unmarshal(body, &capturedBody); err != nil {
			http.Error(w, "json error", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-1",
			"object": "chat.completion",
			"created": 1767683986,
			"model": "qwen-plus",
			"choices": [{
				"index": 0,
				"finish_reason": "stop",
				"message": {"role": "assistant", "content": "Hello!"}
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	// Intentionally do NOT set BifrostContextKeyPassthroughExtraParams — the
	// provider should set it automatically.
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	hello := "Hello"
	_, bifrostErr := provider.ChatCompletion(ctx, imageTestKey(), &schemas.BifrostChatRequest{
		Provider: schemas.DashScope,
		Model:    "qwen-plus",
		Input: []schemas.ChatMessage{{
			Role:    schemas.ChatMessageRoleUser,
			Content: &schemas.ChatMessageContent{ContentStr: &hello},
		}},
		Params: &schemas.ChatParameters{
			ExtraParams: map[string]interface{}{
				"enable_thinking": false,
			},
		},
	})
	if bifrostErr != nil {
		t.Fatalf("ChatCompletion failed: %v", bifrostErr.Error.Message)
	}
	if capturedBody == nil {
		t.Fatal("mock server did not receive a request body")
	}
	got, ok := capturedBody["enable_thinking"]
	if !ok {
		t.Fatalf("enable_thinking missing from outgoing request body; got keys: %v", capturedBody)
	}
	if got != false {
		t.Errorf("enable_thinking = %v, want false", got)
	}
}
