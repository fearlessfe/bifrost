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

// TestEmbeddingExtraParamsFlattened verifies that DashScope-only embedding
// knobs in ExtraParams (dimension, output_dtype, ...) reach the wire without
// the caller opting into extra-params passthrough — the provider sets
// BifrostContextKeyPassthroughExtraParams itself.
func TestEmbeddingExtraParamsFlattened(t *testing.T) {
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
			"object": "list",
			"data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
			"model": "text-embedding-v3",
			"usage": {"prompt_tokens": 2, "total_tokens": 2}
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	// Intentionally do NOT set BifrostContextKeyPassthroughExtraParams — the
	// provider should set it automatically.
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	text := "hello"
	_, bifrostErr := provider.Embedding(ctx, imageTestKey(), &schemas.BifrostEmbeddingRequest{
		Provider: schemas.DashScope,
		Model:    "text-embedding-v3",
		Input:    &schemas.EmbeddingInput{Text: &text},
		Params: &schemas.EmbeddingParameters{
			ExtraParams: map[string]interface{}{
				"dimension": 1024,
			},
		},
	})
	if bifrostErr != nil {
		t.Fatalf("Embedding failed: %v", bifrostErr.Error.Message)
	}
	if capturedBody == nil {
		t.Fatal("mock server did not receive a request body")
	}
	if got, ok := capturedBody["dimension"]; !ok {
		t.Fatalf("dimension missing from outgoing request body; got keys: %v", capturedBody)
	} else if got != float64(1024) {
		t.Errorf("dimension = %v, want 1024", got)
	}
}
