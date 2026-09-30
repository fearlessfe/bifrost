package dashscope

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

func TestToDashScopeTTSRequestBasicMapping(t *testing.T) {
	voice := "Cherry"
	req := ToDashScopeTTSRequest(&schemas.BifrostSpeechRequest{
		Model: "qwen3-tts-flash",
		Input: &schemas.SpeechInput{Input: "你好，世界"},
		Params: &schemas.SpeechParameters{
			VoiceConfig:    &schemas.SpeechVoiceInput{Voice: &voice},
			ResponseFormat: "wav",
			Instructions:   "用轻快的语气",
		},
	})

	if req == nil {
		t.Fatal("expected request, got nil")
	}
	if req.Model != "qwen3-tts-flash" {
		t.Errorf("model = %q, want %q", req.Model, "qwen3-tts-flash")
	}
	if req.Input.Text != "你好，世界" {
		t.Errorf("text = %q, want %q", req.Input.Text, "你好，世界")
	}
	if req.Input.Voice != "Cherry" {
		t.Errorf("voice = %q, want %q", req.Input.Voice, "Cherry")
	}
	if req.Input.Format != "wav" {
		t.Errorf("format = %q, want %q", req.Input.Format, "wav")
	}
	if req.Input.Instructions != "用轻快的语气" {
		t.Errorf("instructions = %q, want %q", req.Input.Instructions, "用轻快的语气")
	}
	if req.Input.Extra != nil {
		t.Errorf("extra should be nil when no ExtraParams given, got %v", req.Input.Extra)
	}
}

func TestToDashScopeTTSRequestNilGuards(t *testing.T) {
	if got := ToDashScopeTTSRequest(nil); got != nil {
		t.Errorf("nil request should produce nil, got %+v", got)
	}
	if got := ToDashScopeTTSRequest(&schemas.BifrostSpeechRequest{Model: "qwen3-tts-flash"}); got != nil {
		t.Errorf("missing input should produce nil, got %+v", got)
	}
}

func TestToDashScopeTTSRequestExtraParamsPromotion(t *testing.T) {
	req := ToDashScopeTTSRequest(&schemas.BifrostSpeechRequest{
		Model: "qwen-tts",
		Input: &schemas.SpeechInput{Input: "hello"},
		Params: &schemas.SpeechParameters{
			ExtraParams: map[string]interface{}{
				"voice":        "Serena",
				"format":       "mp3",
				"sample_rate":  float64(24000), // JSON numbers decode as float64
				"instructions": "calm voice",
				"rate":         float64(1.2), // unknown knob must ride the merge
			},
		},
	})

	if req.Input.Voice != "Serena" {
		t.Errorf("voice from ExtraParams = %q, want %q", req.Input.Voice, "Serena")
	}
	if req.Input.Format != "mp3" {
		t.Errorf("format from ExtraParams = %q, want %q", req.Input.Format, "mp3")
	}
	if req.Input.SampleRate != 24000 {
		t.Errorf("sample_rate from ExtraParams = %d, want 24000", req.Input.SampleRate)
	}
	if req.Input.Instructions != "calm voice" {
		t.Errorf("instructions from ExtraParams = %q, want %q", req.Input.Instructions, "calm voice")
	}
	// Promoted keys are consumed; only unknown knobs remain for the merge.
	if len(req.Input.Extra) != 1 {
		t.Fatalf("extra should only retain unpromoted keys, got %v", req.Input.Extra)
	}
	if req.Input.Extra["rate"] != 1.2 {
		t.Errorf("unpromoted key 'rate' missing from extra, got %v", req.Input.Extra)
	}
}

func TestToDashScopeTTSRequestExtraParamsDoNotMutateCaller(t *testing.T) {
	extra := map[string]interface{}{"sample_rate": 16000}
	req := ToDashScopeTTSRequest(&schemas.BifrostSpeechRequest{
		Model:  "qwen3-tts-flash",
		Input:  &schemas.SpeechInput{Input: "hello"},
		Params: &schemas.SpeechParameters{ExtraParams: extra},
	})

	if len(extra) != 1 {
		t.Errorf("caller's ExtraParams map was mutated: %v", extra)
	}
	if req.Input.SampleRate != 16000 {
		t.Errorf("sample_rate = %d, want 16000", req.Input.SampleRate)
	}
}

func TestDashScopeTTSRequestExtrasStayNestedUnderPassthrough(t *testing.T) {
	req := ToDashScopeTTSRequest(&schemas.BifrostSpeechRequest{
		Model: "qwen3-tts-flash",
		Input: &schemas.SpeechInput{Input: "hello"},
		Params: &schemas.SpeechParameters{
			ExtraParams: map[string]interface{}{"rate": 1.2},
		},
	})

	// MarshalJSON already merges extras into input; GetExtraParams must return
	// nil or the passthrough layer would duplicate them at the JSON root.
	if extra := req.GetExtraParams(); extra != nil {
		t.Errorf("GetExtraParams must return nil, got %v", extra)
	}

	data, err := schemas.MarshalSorted(req)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	var body map[string]interface{}
	if err := schemas.Unmarshal(data, &body); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}
	for key := range body {
		if key != "model" && key != "input" {
			t.Errorf("unexpected root-level key %q; extras must live inside input only", key)
		}
	}
	input, ok := body["input"].(map[string]interface{})
	if !ok {
		t.Fatalf("input missing from marshaled body: %s", data)
	}
	if input["rate"] != 1.2 {
		t.Errorf("rate should ride inside input, got %v", input)
	}
}

// TestSpeechStreamStringNullFinishReason pins the upstream quirk where
// non-terminal SSE events carry finish_reason as the STRING "null" instead of
// JSON null. A plain nil check on FinishReason ends the stream after the first
// audio chunk; the stream must deliver every audio event until "stop".
func TestSpeechStreamStringNullFinishReason(t *testing.T) {
	chunk1 := base64.StdEncoding.EncodeToString([]byte("audio-one"))
	chunk2 := base64.StdEncoding.EncodeToString([]byte("audio-two"))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		writeEvent := func(id int, data string, finish string) {
			fmt.Fprintf(w, "id:%d\nevent:result\n:HTTP_STATUS/200\ndata:{\"output\":{\"audio\":{\"data\":%q,\"id\":\"audio-x\"},\"finish_reason\":%q},\"request_id\":\"req-x\"}\n\n", id, data, finish)
			if flusher != nil {
				flusher.Flush()
			}
		}
		writeEvent(1, chunk1, "null")
		writeEvent(2, chunk2, "null")
		writeEvent(3, "", "stop")
	}))
	defer server.Close()

	provider := newImageTestProvider(t, server.URL)
	ctx, cancel := schemas.NewBifrostContextWithTimeout(nil, 30*time.Second)
	defer cancel()

	voice := "Cherry"
	req := &schemas.BifrostSpeechRequest{
		Provider: schemas.DashScope,
		Model:    "qwen3-tts-flash",
		Input:    &schemas.SpeechInput{Input: "hello world"},
		Params: &schemas.SpeechParameters{
			VoiceConfig:    &schemas.SpeechVoiceInput{Voice: &voice},
			ResponseFormat: "mp3",
		},
	}
	noopHook := func(ctx *schemas.BifrostContext, result *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
		return result, err
	}
	ch, berr := provider.SpeechStream(ctx, noopHook, nil, imageTestKey(), req)
	if berr != nil {
		t.Fatalf("SpeechStream: %v", berr.Error.Message)
	}

	var deltas [][]byte
	var sawDone bool
	for chunk := range ch {
		if chunk.BifrostError != nil {
			t.Fatalf("stream error: %v", chunk.BifrostError.Error.Message)
		}
		speech := chunk.BifrostSpeechStreamResponse
		if speech == nil {
			continue
		}
		switch speech.Type {
		case schemas.SpeechStreamResponseTypeDelta:
			deltas = append(deltas, speech.Audio)
		case schemas.SpeechStreamResponseTypeDone:
			sawDone = true
		}
	}

	if len(deltas) != 2 {
		t.Fatalf("expected 2 audio deltas, got %d (stream must not end on string \"null\" finish_reason)", len(deltas))
	}
	if string(deltas[0]) != "audio-one" || string(deltas[1]) != "audio-two" {
		t.Errorf("unexpected audio payloads: %q, %q", deltas[0], deltas[1])
	}
	if !sawDone {
		t.Error("expected terminal done chunk")
	}
}
