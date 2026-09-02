package tables

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestTableModelPricingRuntimeCostMultiplierIsNotSerialized(t *testing.T) {
	multiplier := 0.8
	data, err := json.Marshal(TableModelPricing{
		Model:          "gpt-4o",
		Provider:       "openai",
		Mode:           "chat",
		CostMultiplier: &multiplier,
	})
	if err != nil {
		t.Fatalf("marshal model pricing: %v", err)
	}
	if strings.Contains(string(data), "cost_multiplier") {
		t.Fatalf("runtime cost multiplier leaked into serialized model pricing: %s", data)
	}
}
