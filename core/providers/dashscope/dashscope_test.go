package dashscope_test

import (
	"os"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/internal/llmtests"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestDashscope(t *testing.T) {
	t.Parallel()
	if strings.TrimSpace(os.Getenv("DASHSCOPE_API_KEY")) == "" {
		t.Skip("Skipping DashScope tests because DASHSCOPE_API_KEY is not set")
	}

	client, ctx, cancel, err := llmtests.SetupTest()
	if err != nil {
		t.Fatalf("Error initializing test setup: %v", err)
	}
	defer cancel()
	defer client.Shutdown()

	testConfig := llmtests.ComprehensiveTestConfig{
		Provider:             schemas.DashScope,
		ChatModel:            "qwen-plus", // compatible-mode chat
		SpeechSynthesisModel: "qwen3-tts-flash",
		TranscriptionModel:   "qwen3-asr-flash",
		ImageGenerationModel: "qwen-image",
		ImageEditModel:       "qwen-image-edit",   // sync multimodal-generation family; the wan* async task API rejects image edit
		EmbeddingModel:       "text-embedding-v3", // compatible-mode embeddings
		Scenarios: llmtests.TestScenarios{
			TextCompletion:        false,
			TextCompletionStream:  false,
			SimpleChat:            true, // dual-API: also covers the non-streaming Responses API (chat fallback)
			CompletionStream:      true, // dual-API: also gates the streaming ResponsesStream scenario
			MultiTurnConversation: true,
			ToolCalls:             true,
			MultipleToolCalls:     false,
			End2EndToolCalling:    false,
			AutomaticFunctionCall: false,
			ImageURL:              false,
			ImageBase64:           false,
			MultipleImages:        false,
			CompleteEnd2End:       false,
			SpeechSynthesis:       true,
			SpeechSynthesisStream: true,
			Transcription:         true,
			TranscriptionStream:   false,
			Embedding:             true, // text-embedding-v3 via compatible-mode delegation
			Reasoning:             false,
			ListModels:            false,
			Realtime:              false,
			ImageGeneration:       true,
			ImageEdit:             true,  // qwen-image-edit on the synchronous multimodal-generation endpoint
			ImageEditStream:       false, // streaming image edit is not supported
		},
	}

	t.Run("DashScopeTests", func(t *testing.T) {
		llmtests.RunAllComprehensiveTests(t, client, ctx, testConfig)
	})
}
