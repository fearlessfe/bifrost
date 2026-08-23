package handlers

import (
	"context"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/encrypt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// newServiceTokenStore builds a real sqlite-backed ConfigStore (full
// migrations, including add_service_tokens_table) so handler and middleware
// tests exercise the actual store semantics.
func newServiceTokenStore(t *testing.T) configstore.ConfigStore {
	t.Helper()
	cs, err := configstore.NewConfigStore(context.Background(), &configstore.Config{
		Enabled: true,
		Type:    configstore.ConfigStoreTypeSQLite,
		Config:  &configstore.SQLiteConfig{Path: filepath.Join(t.TempDir(), "servicetokens.db")},
	}, &mockLogger{})
	require.NoError(t, err)
	require.NotNil(t, cs)
	return cs
}

// newServiceTokenAuthMiddleware builds an AuthMiddleware with auth enabled,
// backed by a real sqlite store.
func newServiceTokenAuthMiddleware(t *testing.T) (*AuthMiddleware, configstore.ConfigStore) {
	t.Helper()
	SetLogger(&mockLogger{})
	store := newServiceTokenStore(t)
	am, err := InitAuthMiddleware(store, nil, nil, "")
	require.NoError(t, err)
	am.UpdateAuthConfig(&configstore.AuthConfig{
		AdminUserName: schemas.NewSecretVar("admin"),
		AdminPassword: schemas.NewSecretVar("hashedpassword"),
		IsEnabled:     true,
	})
	return am, store
}

// runBearerRequest runs the API middleware with a Bearer token against a
// non-whitelisted admin route and reports whether the next handler ran.
func runBearerRequest(t *testing.T, am *AuthMiddleware, token string) (*fasthttp.RequestCtx, bool) {
	t.Helper()
	ctx := initCtx(&fasthttp.Request{})
	ctx.Request.SetRequestURI("/api/some-admin-endpoint")
	ctx.Request.Header.Set("Authorization", "Bearer "+token)
	nextCalled := false
	am.APIMiddleware()(func(ctx *fasthttp.RequestCtx) { nextCalled = true })(ctx)
	return ctx, nextCalled
}

func insertServiceToken(t *testing.T, store configstore.ConfigStore, plaintext string, expiresAt *time.Time) *tables.ServiceTokensTable {
	t.Helper()
	now := time.Now()
	token := &tables.ServiceTokensTable{
		Name:      "test-token",
		TokenHash: encrypt.HashSHA256(plaintext),
		ExpiresAt: expiresAt,
		IsActive:  true,
		CreatedAt: now,
		UpdatedAt: now,
	}
	require.NoError(t, store.CreateServiceToken(context.Background(), token))
	return token
}

func TestAuthMiddleware_ServiceToken_Valid(t *testing.T) {
	am, store := newServiceTokenAuthMiddleware(t)
	insertServiceToken(t, store, "bfsvc_valid-token", nil)

	ctx, nextCalled := runBearerRequest(t, am, "bfsvc_valid-token")
	require.True(t, nextCalled, "valid service token must authenticate, got status %d", ctx.Response.StatusCode())
	isAdmin, _ := ctx.UserValue(schemas.IsLocalAdminContextKey).(bool)
	assert.True(t, isAdmin, "service token must be marked as local admin")

	// last_used_at must be recorded
	tokens, err := store.ListServiceTokens(context.Background())
	require.NoError(t, err)
	require.Len(t, tokens, 1)
	assert.NotNil(t, tokens[0].LastUsedAt)
}

func TestAuthMiddleware_ServiceToken_NegativePaths(t *testing.T) {
	am, store := newServiceTokenAuthMiddleware(t)
	insertServiceToken(t, store, "bfsvc_to-be-deleted", nil)
	past := time.Now().Add(-time.Hour)
	insertServiceToken(t, store, "bfsvc_expired", &past)

	t.Run("wrong token value", func(t *testing.T) {
		ctx, nextCalled := runBearerRequest(t, am, "bfsvc_does-not-exist")
		assert.False(t, nextCalled)
		assert.Equal(t, fasthttp.StatusUnauthorized, ctx.Response.StatusCode())
	})

	t.Run("forged prefix with garbage", func(t *testing.T) {
		ctx, nextCalled := runBearerRequest(t, am, "bfsvc_")
		assert.False(t, nextCalled)
		assert.Equal(t, fasthttp.StatusUnauthorized, ctx.Response.StatusCode())
	})

	t.Run("expired token", func(t *testing.T) {
		ctx, nextCalled := runBearerRequest(t, am, "bfsvc_expired")
		assert.False(t, nextCalled)
		assert.Equal(t, fasthttp.StatusUnauthorized, ctx.Response.StatusCode())
	})

	t.Run("non-service-token garbage still rejected", func(t *testing.T) {
		ctx, nextCalled := runBearerRequest(t, am, "not-a-session-not-a-service-token")
		assert.False(t, nextCalled)
		assert.Equal(t, fasthttp.StatusUnauthorized, ctx.Response.StatusCode())
	})

	t.Run("deleted token", func(t *testing.T) {
		tokens, err := store.ListServiceTokens(context.Background())
		require.NoError(t, err)
		var id uint
		for _, token := range tokens {
			if token.TokenHash == encrypt.HashSHA256("bfsvc_to-be-deleted") {
				id = token.ID
			}
		}
		require.NotZero(t, id)
		require.NoError(t, store.DeleteServiceToken(context.Background(), id))

		ctx, nextCalled := runBearerRequest(t, am, "bfsvc_to-be-deleted")
		assert.False(t, nextCalled)
		assert.Equal(t, fasthttp.StatusUnauthorized, ctx.Response.StatusCode())
	})
}

func TestServiceTokenHandler_CreateListDelete(t *testing.T) {
	SetLogger(&mockLogger{})
	store := newServiceTokenStore(t)
	handler := NewServiceTokenHandler(store)

	// Create
	ctx := initCtx(&fasthttp.Request{})
	ctx.Request.Header.SetMethod("POST")
	ctx.Request.Header.SetContentType("application/json")
	ctx.Request.SetBodyString(`{"name": "provisioner"}`)
	handler.createServiceToken(ctx)
	require.Equal(t, fasthttp.StatusOK, ctx.Response.StatusCode(), "body: %s", string(ctx.Response.Body()))

	var createResp struct {
		ID     uint   `json:"id"`
		Name   string `json:"name"`
		Token  string `json:"token"`
		Active bool   `json:"is_active"`
	}
	require.NoError(t, sonic.Unmarshal(ctx.Response.Body(), &createResp))
	assert.Equal(t, "provisioner", createResp.Name)
	assert.True(t, createResp.Active)
	require.True(t, strings.HasPrefix(createResp.Token, tables.ServiceTokenPrefix), "token must carry the bfsvc_ prefix")
	assert.Len(t, createResp.Token, len(tables.ServiceTokenPrefix)+64, "token must be prefix + 32 bytes hex")

	// The plaintext must verify against the stored hash
	found, err := store.GetServiceTokenByHash(context.Background(), encrypt.HashSHA256(createResp.Token))
	require.NoError(t, err)
	require.NotNil(t, found)
	assert.Equal(t, createResp.ID, found.ID)

	// Create requires a name
	badCtx := initCtx(&fasthttp.Request{})
	badCtx.Request.Header.SetMethod("POST")
	badCtx.Request.Header.SetContentType("application/json")
	badCtx.Request.SetBodyString(`{"name": " "}`)
	handler.createServiceToken(badCtx)
	assert.Equal(t, fasthttp.StatusBadRequest, badCtx.Response.StatusCode())

	// List must not leak token values or hashes
	listCtx := initCtx(&fasthttp.Request{})
	handler.listServiceTokens(listCtx)
	require.Equal(t, fasthttp.StatusOK, listCtx.Response.StatusCode())
	body := string(listCtx.Response.Body())
	assert.NotContains(t, body, createResp.Token)
	assert.NotContains(t, body, encrypt.HashSHA256(createResp.Token))
	assert.Contains(t, body, "provisioner")

	// Delete revokes the token
	deleteCtx := initCtx(&fasthttp.Request{})
	deleteCtx.SetUserValue("id", strconv.FormatUint(uint64(createResp.ID), 10))
	handler.deleteServiceToken(deleteCtx)
	require.Equal(t, fasthttp.StatusOK, deleteCtx.Response.StatusCode(), "body: %s", string(deleteCtx.Response.Body()))

	gone, err := store.GetServiceTokenByHash(context.Background(), encrypt.HashSHA256(createResp.Token))
	require.NoError(t, err)
	assert.Nil(t, gone)

	// Deleting again returns 404
	deleteCtx2 := initCtx(&fasthttp.Request{})
	deleteCtx2.SetUserValue("id", strconv.FormatUint(uint64(createResp.ID), 10))
	handler.deleteServiceToken(deleteCtx2)
	assert.Equal(t, fasthttp.StatusNotFound, deleteCtx2.Response.StatusCode())
}
