package dashscope

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// TestTranscriptionDelegatesToChatCompletions pins the wire shape: DashScope
// has no /audio/transcriptions surface, so a transcription request must go out
// as a chat completion carrying one user message with a single input_audio
// block, and the assistant's reply text must come back as the transcript.
func TestTranscriptionDelegatesToChatCompletions(t *testing.T) {
	audio := []byte("fake-mp3-audio-bytes")
	wantData := "data:audio/mpeg;base64," + base64.StdEncoding.EncodeToString(audio)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != compatibleModeChatCompletionsPath {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", got)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var payload map[string]interface{}
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("request body is not JSON: %v", err)
			return
		}
		if got := payload["model"]; got != "qwen3-asr-flash" {
			t.Errorf("model = %v, want qwen3-asr-flash", got)
		}
		messages, _ := payload["messages"].([]interface{})
		if len(messages) != 1 {
			t.Fatalf("expected exactly one message, got %v", payload["messages"])
		}
		message, _ := messages[0].(map[string]interface{})
		if got := message["role"]; got != "user" {
			t.Errorf("message role = %v, want user", got)
		}
		content, _ := message["content"].([]interface{})
		if len(content) != 1 {
			t.Fatalf("expected exactly one content block, got %v", message["content"])
		}
		block, _ := content[0].(map[string]interface{})
		if got := block["type"]; got != "input_audio" {
			t.Errorf("content block type = %v, want input_audio", got)
		}
		inputAudio, _ := block["input_audio"].(map[string]interface{})
		if got := inputAudio["data"]; got != wantData {
			t.Errorf("input_audio.data = %v, want data URL wrapping the base64 payload", got)
		}
		if got := inputAudio["format"]; got != "mp3" {
			t.Errorf("input_audio.format = %v, want mp3 (from filename)", got)
		}
		// asr_options must ride ExtraParams and be flattened at the top level.
		asrOptions, ok := payload["asr_options"].(map[string]interface{})
		if !ok || asrOptions["enable_itn"] != true {
			t.Errorf("asr_options not flattened onto body: %v", payload["asr_options"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-1",
			"object": "chat.completion",
			"created": 1767683986,
			"model": "qwen3-asr-flash",
			"choices": [{
				"index": 0,
				"finish_reason": "stop",
				"message": {"role": "assistant", "content": "Welcome to Alibaba Cloud."}
			}],
			"usage": {
				"prompt_tokens": 42,
				"completion_tokens": 12,
				"total_tokens": 54,
				"prompt_tokens_details": {"audio_tokens": 42, "text_tokens": 0}
			}
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	// Intentionally do NOT set BifrostContextKeyPassthroughExtraParams — the
	// provider sets it on the chat delegation itself.
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	resp, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
		Provider: schemas.DashScope,
		Model:    "qwen3-asr-flash",
		Input:    &schemas.TranscriptionInput{File: audio, Filename: "sample.mp3"},
		Params: &schemas.TranscriptionParameters{
			ExtraParams: map[string]interface{}{
				"asr_options": map[string]interface{}{"enable_itn": true},
			},
		},
	})
	if bifrostErr != nil {
		t.Fatalf("Transcription failed: %v", bifrostErr.Error.Message)
	}
	if resp.Text != "Welcome to Alibaba Cloud." {
		t.Errorf("Text = %q, want the assistant reply", resp.Text)
	}
	if resp.Task == nil || *resp.Task != "transcribe" {
		t.Errorf("Task = %v, want transcribe", resp.Task)
	}
	if resp.Usage == nil {
		t.Fatal("expected usage to be mapped from the chat response")
	}
	if resp.Usage.InputTokens == nil || *resp.Usage.InputTokens != 42 {
		t.Errorf("InputTokens = %v, want 42", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens == nil || *resp.Usage.OutputTokens != 12 {
		t.Errorf("OutputTokens = %v, want 12", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens == nil || *resp.Usage.TotalTokens != 54 {
		t.Errorf("TotalTokens = %v, want 54", resp.Usage.TotalTokens)
	}
	if resp.Usage.InputTokenDetails == nil || resp.Usage.InputTokenDetails.AudioTokens != 42 {
		t.Errorf("InputTokenDetails.AudioTokens = %v, want 42", resp.Usage.InputTokenDetails)
	}
}

func TestDashScopeAudioFormat(t *testing.T) {
	cases := []struct {
		name    string
		request *schemas.BifrostTranscriptionRequest
		want    *string
	}{
		{
			name: "explicit format wins over filename",
			request: &schemas.BifrostTranscriptionRequest{
				Input:  &schemas.TranscriptionInput{Filename: "sample.mp3"},
				Params: &schemas.TranscriptionParameters{Format: schemas.Ptr("wav")},
			},
			want: schemas.Ptr("wav"),
		},
		{
			name: "explicit format is normalized",
			request: &schemas.BifrostTranscriptionRequest{
				Params: &schemas.TranscriptionParameters{Format: schemas.Ptr(".FLAC")},
			},
			want: schemas.Ptr("flac"),
		},
		{
			name: "filename extension fallback",
			request: &schemas.BifrostTranscriptionRequest{
				Input: &schemas.TranscriptionInput{Filename: "Voice Note.M4A"},
			},
			want: schemas.Ptr("m4a"),
		},
		{
			name: "no format anywhere",
			request: &schemas.BifrostTranscriptionRequest{
				Input: &schemas.TranscriptionInput{Filename: "no-extension"},
			},
			want: nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := dashScopeAudioFormat(tc.request)
			if tc.want == nil {
				if got != nil {
					t.Errorf("dashScopeAudioFormat() = %q, want nil", *got)
				}
				return
			}
			if got == nil || *got != *tc.want {
				t.Errorf("dashScopeAudioFormat() = %v, want %q", got, *tc.want)
			}
		})
	}
}

func TestDashScopeAudioDataURL(t *testing.T) {
	audio := []byte("fake-audio")
	encoded := base64.StdEncoding.EncodeToString(audio)

	cases := []struct {
		name   string
		format *string
		want   string
	}{
		{"mp3 maps to audio/mpeg", schemas.Ptr("mp3"), "data:audio/mpeg;base64," + encoded},
		{"wav maps to audio/wav", schemas.Ptr("wav"), "data:audio/wav;base64," + encoded},
		{"format case is normalized", schemas.Ptr("FLAC"), "data:audio/flac;base64," + encoded},
		{"unknown format falls back to audio/<format>", schemas.Ptr("spx"), "data:audio/spx;base64," + encoded},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := dashScopeAudioDataURL(audio, tc.format); got != tc.want {
				t.Errorf("dashScopeAudioDataURL() = %q, want %q", got, tc.want)
			}
		})
	}

	t.Run("no format sniffs content", func(t *testing.T) {
		wav := append([]byte("RIFFxxxxWAVE"), 0, 0, 0, 0)
		got := dashScopeAudioDataURL(wav, nil)
		if !strings.HasPrefix(got, "data:audio/wav") ||
			!strings.HasSuffix(got, ";base64,"+base64.StdEncoding.EncodeToString(wav)) {
			t.Errorf("dashScopeAudioDataURL() = %q, want a sniffed audio data URL", got)
		}
	})
}

func TestTranscriptionRequiresAudioInput(t *testing.T) {
	provider := newImageTestProvider(t, "http://127.0.0.1:1")
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	if _, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
		Provider: schemas.DashScope,
		Model:    "qwen3-asr-flash",
	}); bifrostErr == nil {
		t.Fatal("expected an error when no audio input is provided")
	}
	if _, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
		Provider: schemas.DashScope,
		Model:    "qwen3-asr-flash",
		Input:    &schemas.TranscriptionInput{},
	}); bifrostErr == nil {
		t.Fatal("expected an error when the audio payload is empty")
	}
}

func TestTranscriptionNilRequest(t *testing.T) {
	provider := newImageTestProvider(t, "http://127.0.0.1:1")
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	bifrostErr := func() (err *schemas.BifrostError) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Transcription panicked on a nil request: %v", r)
			}
		}()
		_, err = provider.Transcription(ctx, imageTestKey(), nil)
		return err
	}()
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error for a nil request")
	}
}

func TestTranscriptionRejectsLargePayloadPassthrough(t *testing.T) {
	provider := newImageTestProvider(t, "http://127.0.0.1:1")
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	// Large-payload mode deliberately skips input materialization, so the
	// provider sees Input == nil even though the caller uploaded a file.
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMode, true)
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadReader, strings.NewReader("raw-multipart-body"))

	_, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
		Provider: schemas.DashScope,
		Model:    "qwen3-asr-flash",
	})
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error when large-payload passthrough leaves Input nil")
	}
	if !strings.Contains(bifrostErr.Error.Message, "passthrough") {
		t.Errorf("error message = %q, want it to say large-payload passthrough is unsupported, not a generic missing-input error", bifrostErr.Error.Message)
	}
}

// TestTranscriptionLanguageMappedToAsrOptions pins the standard transcription
// `language` parameter onto DashScope's wire shape: the compatible-mode ASR
// API reads the language from a top-level asr_options object, so the provider
// must fold Params.Language into ExtraParams as asr_options.language. An
// explicit extra_params.asr_options block wins over Params.Language, and its
// other keys (enable_itn, ...) survive the merge.
func TestTranscriptionLanguageMappedToAsrOptions(t *testing.T) {
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
			"model": "qwen3-asr-flash",
			"choices": [{
				"index": 0,
				"finish_reason": "stop",
				"message": {"role": "assistant", "content": "Welcome to Alibaba Cloud."}
			}]
		}`))
	}))
	defer server.Close()

	newRequest := func() *schemas.BifrostTranscriptionRequest {
		return &schemas.BifrostTranscriptionRequest{
			Provider: schemas.DashScope,
			Model:    "qwen3-asr-flash",
			Input:    &schemas.TranscriptionInput{File: []byte("fake-mp3-audio-bytes"), Filename: "sample.mp3"},
			Params: &schemas.TranscriptionParameters{
				Language: schemas.Ptr("en"),
				ExtraParams: map[string]interface{}{
					"asr_options": map[string]interface{}{"enable_itn": true},
				},
			},
		}
	}

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	if _, bifrostErr := provider.Transcription(ctx, imageTestKey(), newRequest()); bifrostErr != nil {
		t.Fatalf("Transcription failed: %v", bifrostErr.Error.Message)
	}
	asrOptions, ok := capturedBody["asr_options"].(map[string]interface{})
	if !ok {
		t.Fatalf("asr_options missing from outgoing request body; got keys: %v", capturedBody)
	}
	if got := asrOptions["language"]; got != "en" {
		t.Errorf("asr_options.language = %v, want en (mapped from Params.Language)", got)
	}
	if got := asrOptions["enable_itn"]; got != true {
		t.Errorf("asr_options.enable_itn = %v, want true (explicit ExtraParams key preserved)", got)
	}

	// An explicit extra_params.asr_options.language beats Params.Language.
	capturedBody = nil
	conflicting := newRequest()
	conflicting.Params.ExtraParams["asr_options"] = map[string]interface{}{"language": "zh"}
	if _, bifrostErr := provider.Transcription(ctx, imageTestKey(), conflicting); bifrostErr != nil {
		t.Fatalf("Transcription failed: %v", bifrostErr.Error.Message)
	}
	asrOptions, _ = capturedBody["asr_options"].(map[string]interface{})
	if got := asrOptions["language"]; got != "zh" {
		t.Errorf("asr_options.language = %v, want zh (explicit extra_params wins over Params.Language)", got)
	}
}
