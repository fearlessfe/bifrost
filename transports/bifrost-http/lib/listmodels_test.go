package lib

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
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
