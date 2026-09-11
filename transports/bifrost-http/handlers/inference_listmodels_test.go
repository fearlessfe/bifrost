package handlers

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/framework/grant"
	"github.com/maximhq/bifrost/framework/modelcatalog"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/valyala/fasthttp"
)

// The listing hands the narrowing to whoever can answer what the request may reach, so the
// providers it publishes are the ones that answer grants.
func TestApplyListModelsProviderFilterDelegatesToTheModelsManager(t *testing.T) {
	permit := grant.NewPermit(grant.PermitVirtualKey, "vk-test", "Test VK", true, false, []schemas.ProviderPermit{
		{Provider: "openai", AllowedModels: []string{"gpt-4o"}},
		// A provider granted no model at all is still asked: the fan-out decides who can
		// serve the request, and the response is filtered per model afterwards.
		{Provider: "anthropic"},
	}, nil)
	manager := &mockModelsManager{access: grant.NewAccess([]schemas.Permit{permit}, nil, "", nil)}
	h := &CompletionHandler{modelsManager: manager}

	bifrostCtx := schemas.NewBifrostContext(context.Background(), time.Time{})

	h.applyListModelsProviderFilter(bifrostCtx)

	if manager.narrowCalls != 1 {
		t.Fatalf("narrow calls = %d, want 1", manager.narrowCalls)
	}
	got, ok := bifrostCtx.Value(schemas.BifrostContextKeyAvailableProviders).([]schemas.ModelProvider)
	if !ok {
		t.Fatalf("expected available providers to be published as []schemas.ModelProvider, got %#v",
			bifrostCtx.Value(schemas.BifrostContextKeyAvailableProviders))
	}
	want := []schemas.ModelProvider{schemas.OpenAI, schemas.Anthropic}
	if len(got) != len(want) {
		t.Fatalf("expected providers %#v, got %#v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected providers %#v, got %#v", want, got)
		}
	}
}

// Nothing resolved must leave the fan-out alone. Publishing an empty list here would mean "no
// provider may serve this", turning an unrestricted request into one that lists nothing.
func TestApplyListModelsProviderFilterLeavesFanOutAloneWhenNothingResolved(t *testing.T) {
	manager := &mockModelsManager{}
	h := &CompletionHandler{modelsManager: manager}

	bifrostCtx := schemas.NewBifrostContext(context.Background(), time.Time{})

	h.applyListModelsProviderFilter(bifrostCtx)

	if manager.narrowCalls != 1 {
		t.Fatalf("narrow calls = %d, want 1", manager.narrowCalls)
	}
	if got := bifrostCtx.Value(schemas.BifrostContextKeyAvailableProviders); got != nil {
		t.Fatalf("expected nothing to be published, got %#v", got)
	}
}

// Narrowing is an optimization, not a permission check, so a handler wired without a models
// manager falls through instead of panicking on the request path.
func TestApplyListModelsProviderFilterWithoutModelsManager(t *testing.T) {
	h := &CompletionHandler{}

	bifrostCtx := schemas.NewBifrostContext(context.Background(), time.Time{})

	h.applyListModelsProviderFilter(bifrostCtx)

	if got := bifrostCtx.Value(schemas.BifrostContextKeyAvailableProviders); got != nil {
		t.Fatalf("expected nothing to be published, got %#v", got)
	}
}

// A governed request is served from the local catalog: the VK's granted providers and
// model allowlist decide the listing, and no upstream call is made (h.client is nil —
// the legacy path would panic on it).
func TestListModelsFastPathServesFromCatalog(t *testing.T) {
	catalog := modelcatalog.NewTestCatalog(nil)
	catalog.UpsertLive(schemas.OpenAI, "k1", false, []string{"gpt-4o", "gpt-4o-mini"})
	catalog.UpsertLive(schemas.Anthropic, "k1", false, []string{"claude-sonnet-4-5"})

	permit := grant.NewPermit(grant.PermitVirtualKey, "vk-test", "Test VK", true, false, []schemas.ProviderPermit{
		{Provider: "openai", AllowedModels: []string{"gpt-4o"}},
		{Provider: "anthropic", AllowedModels: []string{"*"}},
	}, nil)
	h := &CompletionHandler{
		modelsManager: &mockModelsManager{
			governed: true,
			access:   grant.NewAccess([]schemas.Permit{permit}, nil, "", nil),
		},
		config: &lib.Config{ModelCatalog: catalog, ClientConfig: &configstore.ClientConfig{}},
	}

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("GET")
	ctx.Request.SetRequestURI("/v1/models")

	h.listModels(ctx)

	if ctx.Response.StatusCode() != fasthttp.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", ctx.Response.StatusCode(), ctx.Response.Body())
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(ctx.Response.Body(), &body); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	got := make([]string, 0, len(body.Data))
	for _, m := range body.Data {
		got = append(got, m.ID)
	}
	// Prefixes stripped, VK allowlist applied: gpt-4o-mini is not granted.
	want := []string{"claude-sonnet-4-5", "gpt-4o"}
	if len(got) != len(want) {
		t.Fatalf("expected models %v, got %v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected models %v, got %v", want, got)
		}
	}
}

// Admission refusals (invalid key, mandatory key missing, blocked access) are surfaced
// before any catalog read or upstream call.
func TestListModelsFastPathAdmissionRefusal(t *testing.T) {
	h := &CompletionHandler{
		modelsManager: &mockModelsManager{
			governed: true,
			evaluateErr: &schemas.BifrostError{
				Type:       schemas.Ptr("provider_blocked"),
				StatusCode: schemas.Ptr(403),
				Error:      &schemas.ErrorField{Message: "Provider 'openai' is not allowed"},
			},
		},
		config: &lib.Config{ModelCatalog: modelcatalog.NewTestCatalog(nil), ClientConfig: &configstore.ClientConfig{}},
	}

	ctx := &fasthttp.RequestCtx{}
	ctx.Request.Header.SetMethod("GET")
	ctx.Request.SetRequestURI("/v1/models")

	h.listModels(ctx)

	if ctx.Response.StatusCode() != 403 {
		t.Fatalf("status = %d, want 403; body: %s", ctx.Response.StatusCode(), ctx.Response.Body())
	}
}
