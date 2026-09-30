package dashscope

import (
	"testing"

	"github.com/bytedance/sonic"
	"github.com/valyala/fasthttp"
)

func parseTestResponse(t *testing.T, statusCode int, body string) *fasthttp.Response {
	t.Helper()
	resp := fasthttp.AcquireResponse()
	resp.SetStatusCode(statusCode)
	resp.SetBodyRaw([]byte(body))
	return resp
}

func TestParseDashScopeError(t *testing.T) {
	resp := parseTestResponse(t, 400, `{"code":"InvalidParameter","message":"top_p参数非法","request_id":"req-123"}`)
	bifrostErr := parseDashScopeError(resp)

	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatal("expected error with fields")
	}
	if bifrostErr.Error.Message != "top_p参数非法" {
		t.Errorf("message = %q, want %q", bifrostErr.Error.Message, "top_p参数非法")
	}
	if bifrostErr.Error.Code == nil || *bifrostErr.Error.Code != "InvalidParameter" {
		t.Errorf("code = %v, want InvalidParameter", bifrostErr.Error.Code)
	}
	if bifrostErr.Error.Type != nil {
		t.Errorf("type should be nil without a type field, got %v", *bifrostErr.Error.Type)
	}
	if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != 400 {
		t.Errorf("status = %v, want 400", bifrostErr.StatusCode)
	}
}

func TestParseDashScopeErrorNonJSONBody(t *testing.T) {
	resp := parseTestResponse(t, 502, "<html>bad gateway</html>")
	bifrostErr := parseDashScopeError(resp)

	if bifrostErr == nil || bifrostErr.Error == nil || bifrostErr.Error.Message == "" {
		t.Fatalf("expected a message for a non-JSON body, got %+v", bifrostErr)
	}
}

func TestTaskTerminalErrorPrefersOutputMessage(t *testing.T) {
	err := taskTerminalError(TaskStatusFailed, "InternalError", "allocation failed")
	if err.Error.Message != "dashscope task failed: allocation failed" {
		t.Errorf("message = %q", err.Error.Message)
	}
	if err.Error.Code == nil || *err.Error.Code != "InternalError" {
		t.Errorf("code = %v, want InternalError", err.Error.Code)
	}
}

func TestTaskTerminalErrorFallsBackToCode(t *testing.T) {
	err := taskTerminalError(TaskStatusCanceled, "", "")
	if err.Error.Message != "dashscope task canceled: no details returned" {
		t.Errorf("message = %q", err.Error.Message)
	}
	if err.Error.Code != nil {
		t.Errorf("code should be nil without a code, got %v", *err.Error.Code)
	}
}

func TestDashScopeTTSInputMarshalMergesExtra(t *testing.T) {
	input := DashScopeTTSInput{
		Text:  "hello",
		Extra: map[string]interface{}{"rate": 1.2, "volume": 80},
	}
	data, err := sonic.Marshal(&input)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	// Keys are sorted for byte-stable requests.
	want := `{"rate":1.2,"text":"hello","volume":80}`
	if string(data) != want {
		t.Errorf("marshaled = %s, want %s", data, want)
	}
}

func TestDashScopeImageParametersMarshalMergesExtraWithoutOverwrite(t *testing.T) {
	params := DashScopeImageParameters{
		Size:  "1024*1024",
		Extra: map[string]interface{}{"size": "2048*2048", "watermark": false, "prompt_extend_mode": "agent"},
	}
	data, err := sonic.Marshal(&params)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	// Named fields win over extras; keys are sorted for byte stability.
	want := `{"prompt_extend_mode":"agent","size":"1024*1024","watermark":false}`
	if string(data) != want {
		t.Errorf("marshaled = %s, want %s", data, want)
	}
}
