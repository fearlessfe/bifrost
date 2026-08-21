package lib

import (
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/modelcatalog"
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
