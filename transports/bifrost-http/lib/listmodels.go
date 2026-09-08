package lib

import (
	"sort"
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/modelcatalog"
	"github.com/maximhq/bifrost/plugins/governance"
)

// ModelPriorityFn ranks the providers that can serve a bare model name.
// Index 0 is the most preferred. Providers missing from the returned slice are
// treated as unranked and never win over a ranked provider.
type ModelPriorityFn func(model string) []schemas.ModelProvider

// CatalogModelPriorityFn builds a ModelPriorityFn backed by the model catalog,
// mirroring the provider candidate order used by the model-catalog-resolver
// plugin for unprefixed model requests. Returns nil when catalog is nil.
func CatalogModelPriorityFn(catalog *modelcatalog.ModelCatalog) ModelPriorityFn {
	if catalog == nil {
		return nil
	}
	return catalog.GetProvidersForModel
}

// PreferredModelPriorityFn builds a ModelPriorityFn that always prefers the
// given providers in order, regardless of the model. Used by SDK integration
// endpoints to mirror the resolver's canonical-provider preference (e.g. the
// Anthropic integration prefers anthropic).
func PreferredModelPriorityFn(providers ...schemas.ModelProvider) ModelPriorityFn {
	return func(string) []schemas.ModelProvider {
		return providers
	}
}

// StripProviderPrefixesFromModelList rewrites model IDs of the form
// "provider/model" to the bare "model" and deduplicates entries that collapse
// onto the same bare name. The winner of each group is chosen by priorityFn
// (earliest ranked provider wins); when priorityFn is nil or ranks none of the
// group's providers, the first occurrence wins. Group order follows first
// occurrence in the original list, which reflects configured provider order.
//
// Only resp.Data is modified; pagination and key-status fields are untouched,
// so deduplication applies within the current page only.
func StripProviderPrefixesFromModelList(resp *schemas.BifrostListModelsResponse, priorityFn ModelPriorityFn) {
	if resp == nil || len(resp.Data) == 0 {
		return
	}

	providers := make([]schemas.ModelProvider, len(resp.Data))
	groupOrder := make([]string, 0, len(resp.Data))
	groups := make(map[string][]int, len(resp.Data))

	for i, m := range resp.Data {
		provider, bare := splitModelID(m.ID)
		providers[i] = provider
		if _, seen := groups[bare]; !seen {
			groupOrder = append(groupOrder, bare)
		}
		groups[bare] = append(groups[bare], i)
	}

	out := make([]schemas.Model, 0, len(groupOrder))
	for _, bare := range groupOrder {
		idxs := groups[bare]
		winner := idxs[0]
		if priorityFn != nil && len(idxs) > 1 {
			ranked := priorityFn(bare)
			bestRank := -1
			for _, i := range idxs {
				for rank, p := range ranked {
					if p == providers[i] {
						if bestRank == -1 || rank < bestRank {
							bestRank = rank
							winner = i
						}
						break
					}
				}
			}
		}
		model := resp.Data[winner]
		model.ID = bare
		out = append(out, model)
	}
	resp.Data = out
}

// splitModelID splits "provider/model" at the first slash. IDs without a
// slash are returned unchanged with an empty provider.
func splitModelID(id string) (schemas.ModelProvider, string) {
	if i := strings.Index(id, "/"); i > 0 {
		return schemas.ModelProvider(id[:i]), id[i+1:]
	}
	return "", id
}

// CatalogModelsLister is the slice of the model catalog the list-models fast
// path needs. *modelcatalog.ModelCatalog satisfies it.
type CatalogModelsLister interface {
	GetModelsForProvider(provider schemas.ModelProvider) []string
}

// CatalogListModelsResponse builds a list-models response from the local model
// catalog instead of calling upstream providers: the catalog's effective model
// set per provider (live store + datasheet + keyconfig gating, all in-memory),
// filtered by the request's access with the same semantics as the governance
// pipeline's per-model gate, then sorted and paginated.
//
// Returns nil when there is nothing to serve from: no catalog, or no access
// (a request that presented nothing is not narrowed here — callers fall back
// to the upstream fan-out for it).
//
// afterID is the Anthropic-style cursor: when set, pagination starts after the
// entry with that ID and FirstID/LastID/HasMore are set on the response for
// the Anthropic response converter.
func CatalogListModelsResponse(catalog CatalogModelsLister, access schemas.Access, provider schemas.ModelProvider, pageSize int, pageToken string, afterID string) *schemas.BifrostListModelsResponse {
	return CatalogListModelsResponseWithCursors(catalog, access, provider, pageSize, pageToken, afterID, "")
}

// CatalogListModelsResponseWithCursors is the catalog-backed list response with
// both Anthropic cursor directions available. pageToken is the normal opaque
// Bifrost/Gemini token; afterID and beforeID are Anthropic cursors.
func CatalogListModelsResponseWithCursors(catalog CatalogModelsLister, access schemas.Access, provider schemas.ModelProvider, pageSize int, pageToken string, afterID string, beforeID string) *schemas.BifrostListModelsResponse {
	if catalog == nil || access == nil {
		return nil
	}

	var providers []string
	if provider != "" {
		providers = []string{string(provider)}
	} else {
		providers = access.GrantedProvidersForModel("")
	}

	models := make([]schemas.Model, 0)
	for _, p := range providers {
		for _, id := range catalog.GetModelsForProvider(schemas.ModelProvider(p)) {
			// Wire format is "provider/model". Entries already carrying this
			// provider's prefix are kept as-is; everything else gets it
			// prepended. Model names may themselves contain a slash (e.g.
			// Groq's "openai/gpt-oss-120b"), so the check is strictly "already
			// prefixed with the provider being listed".
			wireID := id
			if !strings.HasPrefix(id, p+"/") {
				wireID = p + "/" + id
			}
			// Same gate as the governance pipeline applies to upstream
			// listings (filterModelsForAccess): parse the emitted ID, then ask
			// the access.
			modelProvider, modelName := schemas.ParseModelString(wireID, "")
			if !access.IsModelAllowed(string(modelProvider), modelName) {
				continue
			}
			models = append(models, schemas.Model{ID: wireID})
		}
	}

	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })

	resp := &schemas.BifrostListModelsResponse{
		Data: models,
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType: schemas.ListModelsRequest,
		},
	}

	// The catalog is also the only source available on this path for model
	// metadata. Keep the same backfill behavior as the native HTTP handler.
	if catalogModelCatalog, ok := catalog.(*modelcatalog.ModelCatalog); ok {
		for i := range resp.Data {
			providerName, modelName := schemas.ParseModelString(resp.Data[i].ID, "")
			entry := catalogModelCatalog.GetPricingEntryForModel(modelName, providerName)
			if entry == nil && resp.Data[i].Alias != nil {
				entry = catalogModelCatalog.GetPricingEntryForModel(*resp.Data[i].Alias, providerName)
			}
			modelcatalog.ApplyModelInfo(&resp.Data[i], entry)
		}
	}

	if beforeID != "" {
		return applyBeforeIDCursor(resp, pageSize, beforeID)
	}
	if afterID != "" {
		return applyAfterIDCursor(resp, pageSize, afterID)
	}
	return resp.ApplyPagination(pageSize, pageToken)
}

// applyAfterIDCursor paginates by an Anthropic-style after_id cursor: the page
// starts at the entry after the named one (matched against the wire ID or its
// bare form), and FirstID/LastID/HasMore are set for the Anthropic converter.
// An unknown cursor restarts from the beginning, mirroring ApplyPagination's
// invalid-cursor behavior.
func applyAfterIDCursor(resp *schemas.BifrostListModelsResponse, pageSize int, afterID string) *schemas.BifrostListModelsResponse {
	offset := 0
	for i, m := range resp.Data {
		_, bare := schemas.ParseModelString(m.ID, "")
		if m.ID == afterID || bare == afterID {
			offset = i + 1
			break
		}
	}

	total := len(resp.Data)
	end := total
	if pageSize > 0 && offset+pageSize < end {
		end = offset + pageSize
	}
	if offset > end {
		offset = end
	}
	page := resp.Data[offset:end]

	out := &schemas.BifrostListModelsResponse{
		Data:        page,
		ExtraFields: resp.ExtraFields,
	}
	if len(page) > 0 {
		out.FirstID = schemas.Ptr(page[0].ID)
		out.LastID = schemas.Ptr(page[len(page)-1].ID)
	}
	out.HasMore = schemas.Ptr(end < total)
	return out
}

// applyBeforeIDCursor returns the page immediately preceding an Anthropic
// before_id cursor. Unknown cursors restart from the beginning, matching the
// existing after_id behavior and the opaque-token pagination fallback.
func applyBeforeIDCursor(resp *schemas.BifrostListModelsResponse, pageSize int, beforeID string) *schemas.BifrostListModelsResponse {
	end := len(resp.Data)
	found := false
	for i, m := range resp.Data {
		_, bare := schemas.ParseModelString(m.ID, "")
		if m.ID == beforeID || bare == beforeID {
			end = i
			found = true
			break
		}
	}
	start := 0
	if !found {
		if pageSize > 0 && end > pageSize {
			end = pageSize
		}
	} else if pageSize > 0 && end > pageSize {
		start = end - pageSize
	}
	page := resp.Data[start:end]
	out := &schemas.BifrostListModelsResponse{Data: page, ExtraFields: resp.ExtraFields}
	if len(page) > 0 {
		out.FirstID = schemas.Ptr(page[0].ID)
		out.LastID = schemas.Ptr(page[len(page)-1].ID)
	}
	out.HasMore = schemas.Ptr(start > 0)
	return out
}

// EvaluateListModelsAccess runs the governance admission funnel for a
// list-models request without routing it: every check a per-provider fan-out
// request would face in PreLLMHook (mandatory key, credential validity,
// access) runs here once, and spending checks are skipped because a listing
// spends nothing. The resolved access is returned for the caller to filter
// the listing with.
//
// governed is false when the store cannot reach a governance plugin — the
// caller falls back to the upstream fan-out path in that case, exactly as a
// deployment without governance behaves.
func EvaluateListModelsAccess(store HandlerStore, ctx *schemas.BifrostContext, provider schemas.ModelProvider) (access schemas.Access, governed bool, bErr *schemas.BifrostError) {
	cfg, ok := store.(*Config)
	if !ok || cfg == nil || ctx == nil {
		return nil, false, nil
	}
	pluginName := governance.PluginName
	if name, ok := ctx.Value(schemas.BifrostContextKeyGovernancePluginName).(string); ok && name != "" {
		pluginName = name
	}
	governancePlugin, err := FindPluginAs[governance.BaseGovernancePlugin](cfg, pluginName)
	if err != nil {
		return nil, false, nil
	}
	ctx.SetValue(schemas.BifrostContextKeySkipBudgetAndRateLimits, true)
	if _, bifrostErr := governancePlugin.Evaluate(ctx, &governance.EvaluationRequest{
		RequestType: schemas.ListModelsRequest,
		Provider:    provider,
	}); bifrostErr != nil {
		return nil, true, bifrostErr
	}
	access, err = governancePlugin.ResolveAccess(ctx)
	if err != nil {
		return nil, true, &schemas.BifrostError{
			IsBifrostError: false,
			Error:          &schemas.ErrorField{Message: err.Error()},
		}
	}
	return access, true, nil
}
