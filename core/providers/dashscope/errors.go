package dashscope

import (
	"fmt"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// parseDashScopeError converts a non-2xx native DashScope response into a
// BifrostError. Native failures are flat JSON: {code, message, request_id}.
func parseDashScopeError(resp *fasthttp.Response) *schemas.BifrostError {
	var errorResp DashScopeErrorBody
	bifrostErr := providerUtils.HandleProviderAPIError(resp, &errorResp)
	if bifrostErr.Error == nil {
		bifrostErr.Error = &schemas.ErrorField{}
	}
	if errorResp.Message != "" {
		bifrostErr.Error.Message = errorResp.Message
	}
	if errorResp.Type != "" {
		bifrostErr.Error.Type = schemas.Ptr(errorResp.Type)
	}
	if errorResp.Code != "" {
		bifrostErr.Error.Code = schemas.Ptr(errorResp.Code)
	}
	return bifrostErr
}

// taskTerminalError builds the BifrostError for a task that reached a terminal
// failure state (FAILED/CANCELED/UNKNOWN), pulling the reason from
// output.code/output.message with the top-level fields as fallback (the poll
// body carries either).
func taskTerminalError(status DashScopeTaskStatus, code, message string) *schemas.BifrostError {
	detail := message
	if detail == "" {
		detail = code
	}
	if detail == "" {
		detail = "no details returned"
	}
	bifrostErr := providerUtils.NewBifrostOperationError(fmt.Sprintf("dashscope task %s: %s", strings.ToLower(string(status)), detail), nil)
	if code != "" {
		bifrostErr.Error.Code = schemas.Ptr(code)
	}
	return bifrostErr
}
