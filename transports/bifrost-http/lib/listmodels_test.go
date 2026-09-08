package lib

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/grant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func modelWithID(id string) schemas.Model {
	name := id
	return schemas.Model{ID: id, Name: &name}
}

func TestStripProviderPrefixes_NilAndEmpty(t *testing.T) {
	// Must not panic on nil response or empty data.
	StripProviderPrefixesFromModelList(nil, nil)
	StripProviderPrefixesFromModelList(&schemas.BifrostListModelsResponse{}, nil)

	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{}}
	StripProviderPrefixesFromModelList(resp, nil)
	assert.Empty(t, resp.Data)
}

func TestStripProviderPrefixes_NoPrefix(t *testing.T) {
	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{
		modelWithID("glm-4.6"),
		modelWithID("gpt-4o"),
	}}
	StripProviderPrefixesFromModelList(resp, nil)
	require.Len(t, resp.Data, 2)
	assert.Equal(t, "glm-4.6", resp.Data[0].ID)
	assert.Equal(t, "gpt-4o", resp.Data[1].ID)
}

func TestStripProviderPrefixes_StripsAndDedupes(t *testing.T) {
	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{
		modelWithID("zhipu/glm-4.6"),
		modelWithID("wafer/glm-4.6"),
		modelWithID("openai/gpt-4o"),
	}}
	StripProviderPrefixesFromModelList(resp, nil)
	require.Len(t, resp.Data, 2)
	assert.Equal(t, "glm-4.6", resp.Data[0].ID)
	assert.Equal(t, "gpt-4o", resp.Data[1].ID)
	// Without a priority function the first occurrence wins.
	assert.Equal(t, "zhipu/glm-4.6", *resp.Data[0].Name)
}

func TestStripProviderPrefixes_PriorityFnPicksWinner(t *testing.T) {
	priority := func(model string) []schemas.ModelProvider {
		if model == "glm-4.6" {
			return []schemas.ModelProvider{"wafer", "zhipu"}
		}
		return nil
	}
	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{
		modelWithID("zhipu/glm-4.6"),
		modelWithID("wafer/glm-4.6"),
	}}
	StripProviderPrefixesFromModelList(resp, priority)
	require.Len(t, resp.Data, 1)
	assert.Equal(t, "glm-4.6", resp.Data[0].ID)
	// wafer ranks first in the priority list, so its entry wins.
	assert.Equal(t, "wafer/glm-4.6", *resp.Data[0].Name)
}

func TestStripProviderPrefixes_UnrankedProvidersKeepFirstOccurrence(t *testing.T) {
	priority := func(model string) []schemas.ModelProvider { return nil }
	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{
		modelWithID("zhipu/glm-4.6"),
		modelWithID("wafer/glm-4.6"),
	}}
	StripProviderPrefixesFromModelList(resp, priority)
	require.Len(t, resp.Data, 1)
	assert.Equal(t, "zhipu/glm-4.6", *resp.Data[0].Name)
}

func TestStripProviderPrefixes_PartiallyRankedPrefersRanked(t *testing.T) {
	// Only wafer is ranked; it must win even though zhipu occurs first.
	priority := func(model string) []schemas.ModelProvider {
		return []schemas.ModelProvider{"wafer"}
	}
	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{
		modelWithID("zhipu/glm-4.6"),
		modelWithID("wafer/glm-4.6"),
	}}
	StripProviderPrefixesFromModelList(resp, priority)
	require.Len(t, resp.Data, 1)
	assert.Equal(t, "wafer/glm-4.6", *resp.Data[0].Name)
}

func TestStripProviderPrefixes_NestedModelPath(t *testing.T) {
	// openrouter-style IDs nest the vendor in the model path; only the first
	// segment is the provider. Different bare names must not be merged.
	resp := &schemas.BifrostListModelsResponse{Data: []schemas.Model{
		modelWithID("openrouter/zhipu-ai/glm-4.6"),
		modelWithID("zhipu/glm-4.6"),
	}}
	StripProviderPrefixesFromModelList(resp, nil)
	require.Len(t, resp.Data, 2)
	assert.Equal(t, "zhipu-ai/glm-4.6", resp.Data[0].ID)
	assert.Equal(t, "glm-4.6", resp.Data[1].ID)
}

func TestStripProviderPrefixes_PreservesPaginationFields(t *testing.T) {
	hasMore := true
	resp := &schemas.BifrostListModelsResponse{
		Data: []schemas.Model{
			modelWithID("zhipu/glm-4.6"),
			modelWithID("wafer/glm-4.6"),
		},
		NextPageToken: "token-123",
		HasMore:       &hasMore,
		KeyStatuses:   []schemas.KeyStatus{{Status: schemas.KeyStatusSuccess}},
	}
	StripProviderPrefixesFromModelList(resp, nil)
	require.Len(t, resp.Data, 1)
	assert.Equal(t, "token-123", resp.NextPageToken)
	assert.Equal(t, &hasMore, resp.HasMore)
	assert.Len(t, resp.KeyStatuses, 1)
}

func TestCatalogModelPriorityFn_NilCatalog(t *testing.T) {
	assert.Nil(t, CatalogModelPriorityFn(nil))
}

func TestPreferredModelPriorityFn(t *testing.T) {
	fn := PreferredModelPriorityFn(schemas.Anthropic, schemas.OpenAI)
	require.NotNil(t, fn)
	assert.Equal(t, []schemas.ModelProvider{schemas.Anthropic, schemas.OpenAI}, fn("any-model"))
}

// fakeCatalogLister serves fixed per-provider model sets for the catalog fast path tests.
type fakeCatalogLister map[schemas.ModelProvider][]string

func (f fakeCatalogLister) GetModelsForProvider(provider schemas.ModelProvider) []string {
	return f[provider]
}

func accessWithPermits(t *testing.T, permits ...schemas.ProviderPermit) schemas.Access {
	t.Helper()
	permit := grant.NewPermit(grant.PermitVirtualKey, "vk-test", "Test VK", true, false, permits, nil)
	return grant.NewAccess([]schemas.Permit{permit}, nil, "", nil)
}

func modelIDs(resp *schemas.BifrostListModelsResponse) []string {
	ids := make([]string, 0, len(resp.Data))
	for _, m := range resp.Data {
		ids = append(ids, m.ID)
	}
	return ids
}

func TestCatalogListModelsResponse_NilInputs(t *testing.T) {
	access := accessWithPermits(t, schemas.ProviderPermit{Provider: "openai", AllowedModels: []string{"*"}})
	assert.Nil(t, CatalogListModelsResponse(nil, access, "", 0, "", ""))
	assert.Nil(t, CatalogListModelsResponse(fakeCatalogLister{}, nil, "", 0, "", ""))
}

func TestCatalogListModelsResponse_FiltersByAccess(t *testing.T) {
	catalog := fakeCatalogLister{
		schemas.OpenAI:    {"gpt-4o", "gpt-4o-mini"},
		schemas.Anthropic: {"claude-sonnet-4-5"},
		schemas.Groq:      {"llama-3.3-70b"},
	}
	access := accessWithPermits(t,
		schemas.ProviderPermit{Provider: "openai", AllowedModels: []string{"gpt-4o"}},
		schemas.ProviderPermit{Provider: "anthropic", AllowedModels: []string{"*"}},
		// groq granted but with an empty allowlist: the provider is listed, no model survives.
		schemas.ProviderPermit{Provider: "groq"},
	)

	resp := CatalogListModelsResponse(catalog, access, "", 0, "", "")
	require.NotNil(t, resp)
	// Sorted by ID, provider-prefixed, per-model allowlist applied.
	assert.Equal(t, []string{"anthropic/claude-sonnet-4-5", "openai/gpt-4o"}, modelIDs(resp))
	assert.Equal(t, schemas.ListModelsRequest, resp.ExtraFields.RequestType)
}

func TestCatalogListModelsResponse_ExplicitProvider(t *testing.T) {
	catalog := fakeCatalogLister{
		schemas.OpenAI:    {"gpt-4o"},
		schemas.Anthropic: {"claude-sonnet-4-5"},
	}
	access := accessWithPermits(t,
		schemas.ProviderPermit{Provider: "openai", AllowedModels: []string{"*"}},
		schemas.ProviderPermit{Provider: "anthropic", AllowedModels: []string{"*"}},
	)

	resp := CatalogListModelsResponse(catalog, access, schemas.OpenAI, 0, "", "")
	require.NotNil(t, resp)
	assert.Equal(t, []string{"openai/gpt-4o"}, modelIDs(resp))

	// A provider the access does not grant yields nothing.
	resp = CatalogListModelsResponse(catalog, access, schemas.Groq, 0, "", "")
	require.NotNil(t, resp)
	assert.Empty(t, resp.Data)
}

func TestCatalogListModelsResponse_ModelNamesWithSlash(t *testing.T) {
	// Groq serves models whose own IDs contain a slash; only an existing
	// "groq/" prefix must be treated as already-prefixed.
	catalog := fakeCatalogLister{
		schemas.Groq: {"openai/gpt-oss-120b", "groq/grok-1"},
	}
	access := accessWithPermits(t, schemas.ProviderPermit{Provider: "groq", AllowedModels: []string{"*"}})

	resp := CatalogListModelsResponse(catalog, access, "", 0, "", "")
	require.NotNil(t, resp)
	assert.Equal(t, []string{"groq/grok-1", "groq/openai/gpt-oss-120b"}, modelIDs(resp))
}

func TestCatalogListModelsResponse_Pagination(t *testing.T) {
	catalog := fakeCatalogLister{
		schemas.OpenAI: {"a-model", "b-model", "c-model"},
	}
	access := accessWithPermits(t, schemas.ProviderPermit{Provider: "openai", AllowedModels: []string{"*"}})

	page1 := CatalogListModelsResponse(catalog, access, "", 2, "", "")
	require.NotNil(t, page1)
	assert.Equal(t, []string{"openai/a-model", "openai/b-model"}, modelIDs(page1))
	require.NotEmpty(t, page1.NextPageToken)

	page2 := CatalogListModelsResponse(catalog, access, "", 2, page1.NextPageToken, "")
	require.NotNil(t, page2)
	assert.Equal(t, []string{"openai/c-model"}, modelIDs(page2))
	assert.Empty(t, page2.NextPageToken)
}

func TestCatalogListModelsResponse_AfterIDCursor(t *testing.T) {
	catalog := fakeCatalogLister{
		schemas.Anthropic: {"claude-a", "claude-b", "claude-c"},
	}
	access := accessWithPermits(t, schemas.ProviderPermit{Provider: "anthropic", AllowedModels: []string{"*"}})

	resp := CatalogListModelsResponse(catalog, access, "", 2, "", "claude-a")
	require.NotNil(t, resp)
	assert.Equal(t, []string{"anthropic/claude-b", "anthropic/claude-c"}, modelIDs(resp))
	require.NotNil(t, resp.FirstID)
	require.NotNil(t, resp.LastID)
	assert.Equal(t, "anthropic/claude-b", *resp.FirstID)
	assert.Equal(t, "anthropic/claude-c", *resp.LastID)
	require.NotNil(t, resp.HasMore)
	assert.False(t, *resp.HasMore)

	// Unknown cursor restarts from the beginning, mirroring ApplyPagination.
	resp = CatalogListModelsResponse(catalog, access, "", 2, "", "no-such-model")
	require.NotNil(t, resp)
	assert.Equal(t, []string{"anthropic/claude-a", "anthropic/claude-b"}, modelIDs(resp))
	require.NotNil(t, resp.HasMore)
	assert.True(t, *resp.HasMore)

	resp = CatalogListModelsResponseWithCursors(catalog, access, "", 2, "", "", "missing")
	require.NotNil(t, resp)
	assert.Equal(t, []string{"anthropic/claude-a", "anthropic/claude-b"}, modelIDs(resp))
}

func TestCatalogListModelsResponse_BeforeIDCursor(t *testing.T) {
	catalog := fakeCatalogLister{
		schemas.Anthropic: {"claude-a", "claude-b", "claude-c", "claude-d"},
	}
	access := accessWithPermits(t, schemas.ProviderPermit{Provider: "anthropic", AllowedModels: []string{"*"}})

	resp := CatalogListModelsResponseWithCursors(catalog, access, "", 2, "", "", "claude-d")
	require.NotNil(t, resp)
	assert.Equal(t, []string{"anthropic/claude-b", "anthropic/claude-c"}, modelIDs(resp))
	require.NotNil(t, resp.FirstID)
	require.NotNil(t, resp.LastID)
	assert.Equal(t, "anthropic/claude-b", *resp.FirstID)
	assert.Equal(t, "anthropic/claude-c", *resp.LastID)
	require.NotNil(t, resp.HasMore)
	assert.True(t, *resp.HasMore)
}

type nonConfigHandlerStore struct{ HandlerStore }

func TestEvaluateListModelsAccess_Fallbacks(t *testing.T) {
	// A store that is not *Config cannot reach the plugin registry: not governed.
	access, governed, bErr := EvaluateListModelsAccess(nonConfigHandlerStore{}, schemas.NewBifrostContext(nil, schemas.NoDeadline), "")
	assert.False(t, governed)
	assert.Nil(t, access)
	assert.Nil(t, bErr)

	// A *Config with no plugins loaded finds no governance plugin: not governed.
	access, governed, bErr = EvaluateListModelsAccess(&Config{}, schemas.NewBifrostContext(nil, schemas.NoDeadline), schemas.OpenAI)
	assert.False(t, governed)
	assert.Nil(t, access)
	assert.Nil(t, bErr)
}
