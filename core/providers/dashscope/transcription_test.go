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

// TestTranscriptionNativeASRRoutesToMultimodalGeneration pins the wire path
// for the qwen-audio-3.x ASR flash models. DashScope's OpenAI-compatible chat
// surface rejects them (qwen-audio-3.1-asr-flash 404s with "Unsupported model
// ... for OpenAI compatibility mode", qwen-audio-3.0-asr-flash 400s with
// UNSUPPORTED_FORMAT), so they must run on the native multimodal-generation
// endpoint: X-DashScope-SSE disabled, audio as an input_audio part inside
// input.messages, the required audio format in parameters.format, the standard
// language param folded into parameters.language_hints, and remaining extra
// params merged into parameters — except asr_options, which is a
// compatible-mode-only knob bag and must not cross over.
func TestTranscriptionNativeASRRoutesToMultimodalGeneration(t *testing.T) {
	audio := []byte("fake-mp3-audio-bytes")
	wantData := "data:audio/mpeg;base64," + base64.StdEncoding.EncodeToString(audio)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != nativeMultimodalGenerationPath {
			t.Errorf("unexpected request: %s %s, want POST %s (3.x ASR models must not ride compatible-mode chat)", r.Method, r.URL.Path, nativeMultimodalGenerationPath)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", got)
		}
		if got := r.Header.Get("X-DashScope-SSE"); got != "disable" {
			t.Errorf("X-DashScope-SSE = %q, want disable", got)
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
		if got := payload["model"]; got != "qwen-audio-3.1-asr-flash" {
			t.Errorf("model = %v, want qwen-audio-3.1-asr-flash", got)
		}
		if _, isCompat := payload["messages"]; isCompat {
			t.Errorf("body carries top-level messages — that is the compatible-mode chat shape, not the native one: %s", body)
		}
		input, _ := payload["input"].(map[string]interface{})
		messages, _ := input["messages"].([]interface{})
		if len(messages) != 1 {
			t.Fatalf("expected exactly one input message, got %v", input["messages"])
		}
		message, _ := messages[0].(map[string]interface{})
		if got := message["role"]; got != "user" {
			t.Errorf("message role = %v, want user", got)
		}
		content, _ := message["content"].([]interface{})
		if len(content) != 1 {
			t.Fatalf("expected exactly one content part, got %v", message["content"])
		}
		part, _ := content[0].(map[string]interface{})
		if got := part["type"]; got != "input_audio" {
			t.Errorf("content part type = %v, want input_audio", got)
		}
		inputAudio, _ := part["input_audio"].(map[string]interface{})
		if got := inputAudio["data"]; got != wantData {
			t.Errorf("input_audio.data = %v, want data URL wrapping the base64 payload", got)
		}
		parameters, _ := payload["parameters"].(map[string]interface{})
		if got := parameters["format"]; got != "mp3" {
			t.Errorf("parameters.format = %v, want mp3 (from filename)", got)
		}
		hints, _ := parameters["language_hints"].([]interface{})
		if len(hints) != 1 || hints[0] != "en" {
			t.Errorf("parameters.language_hints = %v, want [en] (mapped from Params.Language)", parameters["language_hints"])
		}
		if got := parameters["sample_rate"]; got != "16000" {
			t.Errorf("parameters.sample_rate = %v, want 16000 (extra param merged into parameters)", got)
		}
		if _, leaked := parameters["asr_options"]; leaked {
			t.Errorf("parameters.asr_options = %v — asr_options is a compatible-mode knob and must not ride the native path", parameters["asr_options"])
		}

		w.Header().Set("Content-Type", "application/json")
		// Observed wire shape: the result nests twice (output.output.text).
		_, _ = w.Write([]byte(`{
			"output": {
				"output": {
					"request_id": "req-1",
					"sentence": {"begin_time": 200, "channel_id": 0, "end_time": 4240, "sentence_end": true, "sentence_id": 1, "text": "Hello, world.", "words": [{"begin_time": 200, "end_time": 480, "fixed": true, "punctuation": ",", "text": "Hello"}]},
					"text": "Hello, world."
				}
			},
			"usage": {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105, "duration": 4.04},
			"request_id": "req-1"
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	resp, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
		Provider: schemas.DashScope,
		Model:    "qwen-audio-3.1-asr-flash",
		Input:    &schemas.TranscriptionInput{File: audio, Filename: "sample.mp3"},
		Params: &schemas.TranscriptionParameters{
			Language: schemas.Ptr("en"),
			ExtraParams: map[string]interface{}{
				"sample_rate": "16000",
				"asr_options": map[string]interface{}{"enable_itn": true},
			},
		},
	})
	if bifrostErr != nil {
		t.Fatalf("Transcription failed: %v", bifrostErr.Error.Message)
	}
	if resp.Text != "Hello, world." {
		t.Errorf("Text = %q, want the nested output.output.text transcript", resp.Text)
	}
	if resp.Task == nil || *resp.Task != "transcribe" {
		t.Errorf("Task = %v, want transcribe", resp.Task)
	}
	if resp.Usage == nil {
		t.Fatal("expected usage to be mapped from the native response")
	}
	if resp.Usage.InputTokens == nil || *resp.Usage.InputTokens != 100 {
		t.Errorf("InputTokens = %v, want 100", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens == nil || *resp.Usage.OutputTokens != 5 {
		t.Errorf("OutputTokens = %v, want 5", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens == nil || *resp.Usage.TotalTokens != 105 {
		t.Errorf("TotalTokens = %v, want 105", resp.Usage.TotalTokens)
	}
	if resp.Usage.Seconds == nil || *resp.Usage.Seconds != 4.04 {
		t.Errorf("Usage.Seconds = %v, want 4.04 (native duration mapped)", resp.Usage.Seconds)
	}
	if resp.Duration == nil || *resp.Duration != 4.04 {
		t.Errorf("Duration = %v, want 4.04", resp.Duration)
	}
}

// TestTranscriptionNativeASRRequiresAudioFormat pins the guard on the native
// path: parameters.format is mandatory upstream, so a request with neither
// file_format nor a filename extension must fail locally with a clear error
// instead of shipping a body the upstream rejects with UNSUPPORTED_FORMAT.
func TestTranscriptionNativeASRRequiresAudioFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected request: %s %s — a format-less 3.x ASR request must fail before hitting the wire", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
		Provider: schemas.DashScope,
		Model:    "qwen-audio-3.0-asr-flash",
		Input:    &schemas.TranscriptionInput{File: []byte("fake-audio")},
	})
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error when no audio format can be resolved")
	}
	if !strings.Contains(bifrostErr.Error.Message, "format") {
		t.Errorf("error message = %q, want it to say the audio format is required", bifrostErr.Error.Message)
	}
}

// TestTranscriptionNativeASRVariantsRejected pins the boundary of the native
// path: the realtime/streaming and filetrans variants of the 3.x ASR family
// run on the WebSocket realtime and asynchronous task APIs, which Bifrost does
// not serve. They must fail with a clear local error rather than being routed
// to compatible mode, which does not host them (3.1 404s there).
func TestTranscriptionNativeASRVariantsRejected(t *testing.T) {
	for _, model := range []string{"qwen-audio-3.0-asr-flash-filetrans", "qwen-audio-3.1-asr-flash-streaming"} {
		t.Run(model, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				t.Errorf("unexpected request: %s %s — the %s variant must fail locally, not ride compatible-mode chat", r.Method, r.URL.Path, model)
				w.WriteHeader(http.StatusNotFound)
			}))
			defer server.Close()

			provider := newImageTestProvider(t, server.URL)
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

			_, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
				Provider: schemas.DashScope,
				Model:    model,
				Input:    &schemas.TranscriptionInput{File: []byte("fake-audio"), Filename: "sample.mp3"},
			})
			if bifrostErr == nil || bifrostErr.Error == nil {
				t.Fatalf("expected an error for the unsupported %s variant", model)
			}
			if !strings.Contains(bifrostErr.Error.Message, "not supported") {
				t.Errorf("error message = %q, want a clear unsupported-variant error", bifrostErr.Error.Message)
			}
		})
	}
}

// TestTranscriptionNativeASRFlatOutputFallback pins the defensive parse: the
// observed wire shape nests the result as output.output.text, but a sibling
// shape answering with a flat output.text must still transcribe.
func TestTranscriptionNativeASRFlatOutputFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != nativeMultimodalGenerationPath {
			t.Errorf("unexpected request: %s %s, want POST %s", r.Method, r.URL.Path, nativeMultimodalGenerationPath)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"output": {"text": "Flat transcript."},
			"usage": {"duration": 2.5},
			"request_id": "req-2"
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	resp, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
		Provider: schemas.DashScope,
		Model:    "qwen-audio-3.0-asr-flash",
		Input:    &schemas.TranscriptionInput{File: []byte("fake-audio"), Filename: "sample.mp3"},
	})
	if bifrostErr != nil {
		t.Fatalf("Transcription failed: %v", bifrostErr.Error.Message)
	}
	if resp.Text != "Flat transcript." {
		t.Errorf("Text = %q, want the flat output.text fallback", resp.Text)
	}
	if resp.Usage == nil || resp.Usage.Seconds == nil || *resp.Usage.Seconds != 2.5 {
		t.Errorf("Usage.Seconds = %v, want 2.5 (duration-only usage maps)", resp.Usage)
	}
}

// TestTranscriptionNativeASRMissingText pins the empty-transcript failure: a
// 200 whose output carries no text on either nesting level is an operation
// error, not a silent empty response.
func TestTranscriptionNativeASRMissingText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"output": {"output": {"request_id": "req-3"}},
			"request_id": "req-3"
		}`))
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, bifrostErr := provider.Transcription(ctx, imageTestKey(), &schemas.BifrostTranscriptionRequest{
		Provider: schemas.DashScope,
		Model:    "qwen-audio-3.1-asr-flash",
		Input:    &schemas.TranscriptionInput{File: []byte("fake-audio"), Filename: "sample.mp3"},
	})
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected an error when the native response carries no transcript text")
	}
	if !strings.Contains(bifrostErr.Error.Message, "no transcription text") {
		t.Errorf("error message = %q, want it to say no transcription text was returned", bifrostErr.Error.Message)
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

func TestIsNativeASRModel(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"qwen-audio-3.0-asr-flash", true},
		{"qwen-audio-3.1-asr-flash", true},
		{"QWEN-AUDIO-3.1-ASR-FLASH", true},
		{" qwen-audio-3.1-asr-flash ", true},
		{"qwen-audio-3.0-asr-flash-filetrans", false},
		{"qwen-audio-3.1-asr-flash-streaming", false},
		{"qwen3-asr-flash", false},
		{"qwen-audio-2.5-asr-flash", false},
		{"qwen-audio-turbo", false},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if got := isNativeASRModel(tc.model); got != tc.want {
				t.Errorf("isNativeASRModel(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

// TestToDashScopeASRRequest pins the native body construction: the format is
// resolved from file_format/filename, Params.Language folds into
// language_hints unless extra_params already carries one, and asr_options is
// dropped from the native parameters.
func TestToDashScopeASRRequest(t *testing.T) {
	newRequest := func() *schemas.BifrostTranscriptionRequest {
		return &schemas.BifrostTranscriptionRequest{
			Provider: schemas.DashScope,
			Model:    "qwen-audio-3.1-asr-flash",
			Input:    &schemas.TranscriptionInput{File: []byte("fake-audio"), Filename: "sample.mp3"},
			Params: &schemas.TranscriptionParameters{
				Language: schemas.Ptr("en"),
				ExtraParams: map[string]interface{}{
					"asr_options": map[string]interface{}{"enable_itn": true},
				},
			},
		}
	}

	req := ToDashScopeASRRequest(newRequest())
	if req == nil {
		t.Fatal("ToDashScopeASRRequest returned nil for a valid request")
	}
	if req.Parameters == nil || req.Parameters.Format != "mp3" {
		t.Errorf("Parameters.Format = %v, want mp3 (from filename)", req.Parameters)
	}
	if got, ok := req.Parameters.Extra["asr_options"]; ok {
		t.Errorf("Extra[asr_options] = %v — asr_options must not ride the native path", got)
	}
	hints, ok := req.Parameters.Extra["language_hints"].([]string)
	if !ok || len(hints) != 1 || hints[0] != "en" {
		t.Errorf("Extra[language_hints] = %v, want [en] (mapped from Params.Language)", req.Parameters.Extra["language_hints"])
	}

	// An explicit extra_params.language_hints beats Params.Language.
	conflicting := newRequest()
	conflicting.Params.ExtraParams["language_hints"] = []string{"zh"}
	req = ToDashScopeASRRequest(conflicting)
	hints, _ = req.Parameters.Extra["language_hints"].([]string)
	if len(hints) != 1 || hints[0] != "zh" {
		t.Errorf("Extra[language_hints] = %v, want [zh] (explicit extra_params wins over Params.Language)", req.Parameters.Extra["language_hints"])
	}

	// An explicit file_format wins over the filename extension.
	withFormat := newRequest()
	withFormat.Params.Format = schemas.Ptr(".WAV")
	req = ToDashScopeASRRequest(withFormat)
	if req.Parameters.Format != "wav" {
		t.Errorf("Parameters.Format = %q, want wav (file_format normalized over filename)", req.Parameters.Format)
	}

	if got := ToDashScopeASRRequest(nil); got != nil {
		t.Errorf("ToDashScopeASRRequest(nil) = %v, want nil", got)
	}
}

// TestToBifrostASRResponse pins the response mapping boundaries: nil input,
// and a response carrying no text on either nesting level.
func TestToBifrostASRResponse(t *testing.T) {
	if resp, ok := ToBifrostASRResponse(nil); ok || resp != nil {
		t.Errorf("ToBifrostASRResponse(nil) = (%v, %v), want (nil, false)", resp, ok)
	}

	var noText DashScopeASRResponse
	if err := json.Unmarshal([]byte(`{"output": {"output": {"request_id": "r"}}}`), &noText); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp, ok := ToBifrostASRResponse(&noText); ok || resp != nil {
		t.Errorf("ToBifrostASRResponse(no text) = (%v, %v), want (nil, false)", resp, ok)
	}

	var nested DashScopeASRResponse
	if err := json.Unmarshal([]byte(`{"output": {"output": {"text": "hello"}, "text": "flat"}}`), &nested); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	resp, ok := ToBifrostASRResponse(&nested)
	if !ok || resp.Text != "hello" {
		t.Errorf("ToBifrostASRResponse(nested) = (%v, %v), want text hello from output.output.text", resp, ok)
	}
}
