package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	"github.com/fasthttp/router"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/encrypt"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/valyala/fasthttp"
)

// ServiceTokenHandler manages long-lived service tokens that grant
// admin-equivalent access to the management API (equivalent to Basic auth).
// The routes themselves sit behind the same admin auth middleware chain as
// the rest of the management API — only authenticated admins can create,
// list, or revoke tokens.
type ServiceTokenHandler struct {
	configStore configstore.ConfigStore
}

// NewServiceTokenHandler creates a new service token handler instance
func NewServiceTokenHandler(configStore configstore.ConfigStore) *ServiceTokenHandler {
	return &ServiceTokenHandler{configStore: configStore}
}

// RegisterRoutes registers the service token routes
func (h *ServiceTokenHandler) RegisterRoutes(r *router.Router, middlewares ...schemas.BifrostHTTPMiddleware) {
	r.POST("/api/service-tokens", lib.ChainMiddlewares(h.createServiceToken, middlewares...))
	r.GET("/api/service-tokens", lib.ChainMiddlewares(h.listServiceTokens, middlewares...))
	r.DELETE("/api/service-tokens/{id}", lib.ChainMiddlewares(h.deleteServiceToken, middlewares...))
}

type createServiceTokenRequest struct {
	Name      string     `json:"name"`
	ExpiresAt *time.Time `json:"expires_at,omitempty"`
}

// createServiceToken handles POST /api/service-tokens - Create a service token.
// The plaintext token is returned exactly once in this response; only its
// SHA-256 hash is stored.
func (h *ServiceTokenHandler) createServiceToken(ctx *fasthttp.RequestCtx) {
	if h.configStore == nil {
		SendError(ctx, fasthttp.StatusServiceUnavailable, "Config store is not available")
		return
	}
	var req createServiceTokenRequest
	if err := sonic.Unmarshal(ctx.PostBody(), &req); err != nil {
		SendError(ctx, fasthttp.StatusBadRequest, "Invalid request payload")
		return
	}
	req.Name = strings.TrimSpace(req.Name)
	if req.Name == "" {
		SendError(ctx, fasthttp.StatusBadRequest, "Missing required field: name")
		return
	}
	if req.ExpiresAt != nil && req.ExpiresAt.Before(time.Now()) {
		SendError(ctx, fasthttp.StatusBadRequest, "expires_at must be in the future")
		return
	}
	tokenValue, err := generateServiceToken()
	if err != nil {
		logger.Error("failed to generate service token: %v", err)
		SendError(ctx, fasthttp.StatusInternalServerError, "Failed to generate service token")
		return
	}
	now := time.Now()
	token := &tables.ServiceTokensTable{
		Name:      req.Name,
		TokenHash: encrypt.HashSHA256(tokenValue),
		ExpiresAt: req.ExpiresAt,
		IsActive:  true,
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := h.configStore.CreateServiceToken(ctx, token); err != nil {
		logger.Error("failed to create service token: %v", err)
		SendError(ctx, fasthttp.StatusInternalServerError, "Failed to create service token")
		return
	}
	SendJSON(ctx, map[string]any{
		"id":         token.ID,
		"name":       token.Name,
		"token":      tokenValue,
		"expires_at": token.ExpiresAt,
		"is_active":  token.IsActive,
		"created_at": token.CreatedAt,
	})
}

// listServiceTokens handles GET /api/service-tokens - List service tokens.
// The response never includes token values or hashes.
func (h *ServiceTokenHandler) listServiceTokens(ctx *fasthttp.RequestCtx) {
	if h.configStore == nil {
		SendError(ctx, fasthttp.StatusServiceUnavailable, "Config store is not available")
		return
	}
	tokens, err := h.configStore.ListServiceTokens(ctx)
	if err != nil {
		logger.Error("failed to list service tokens: %v", err)
		SendError(ctx, fasthttp.StatusInternalServerError, "Failed to list service tokens")
		return
	}
	SendJSON(ctx, map[string]any{
		"service_tokens": tokens,
	})
}

// deleteServiceToken handles DELETE /api/service-tokens/{id} - Revoke a service token
func (h *ServiceTokenHandler) deleteServiceToken(ctx *fasthttp.RequestCtx) {
	if h.configStore == nil {
		SendError(ctx, fasthttp.StatusServiceUnavailable, "Config store is not available")
		return
	}
	idStr, ok := ctx.UserValue("id").(string)
	if !ok || idStr == "" {
		SendError(ctx, fasthttp.StatusBadRequest, "Invalid service token id")
		return
	}
	id, err := strconv.ParseUint(idStr, 10, 32)
	if err != nil {
		SendError(ctx, fasthttp.StatusBadRequest, "Invalid service token id")
		return
	}
	if err := h.configStore.DeleteServiceToken(ctx, uint(id)); err != nil {
		if errors.Is(err, configstore.ErrNotFound) {
			SendError(ctx, fasthttp.StatusNotFound, "Service token not found")
			return
		}
		logger.Error("failed to delete service token: %v", err)
		SendError(ctx, fasthttp.StatusInternalServerError, "Failed to delete service token")
		return
	}
	SendJSON(ctx, map[string]any{
		"message": "Service token deleted",
	})
}

// generateServiceToken generates a new service token: bfsvc_ prefix followed
// by 32 crypto/rand bytes hex-encoded.
func generateServiceToken() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return tables.ServiceTokenPrefix + hex.EncodeToString(buf), nil
}
