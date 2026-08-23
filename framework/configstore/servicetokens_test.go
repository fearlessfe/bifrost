package configstore

import (
	"context"
	"testing"
	"time"

	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/encrypt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setupServiceTokenTestStore returns an RDBConfigStore with the service_tokens
// table migrated in.
func setupServiceTokenTestStore(t *testing.T) *RDBConfigStore {
	t.Helper()
	store := setupRDBTestStore(t)
	require.NoError(t, store.DB().AutoMigrate(&tables.ServiceTokensTable{}))
	return store
}

func newTestServiceToken(name, plaintext string, expiresAt *time.Time) *tables.ServiceTokensTable {
	now := time.Now()
	return &tables.ServiceTokensTable{
		Name:      name,
		TokenHash: encrypt.HashSHA256(plaintext),
		ExpiresAt: expiresAt,
		IsActive:  true,
		CreatedAt: now,
		UpdatedAt: now,
	}
}

func TestServiceToken_CreateAndGetByHash(t *testing.T) {
	store := setupServiceTokenTestStore(t)
	ctx := context.Background()

	token := newTestServiceToken("provisioner", "bfsvc_secret-value", nil)
	require.NoError(t, store.CreateServiceToken(ctx, token))
	require.NotZero(t, token.ID)

	// Lookup by the correct hash succeeds
	found, err := store.GetServiceTokenByHash(ctx, encrypt.HashSHA256("bfsvc_secret-value"))
	require.NoError(t, err)
	require.NotNil(t, found)
	assert.Equal(t, "provisioner", found.Name)
	assert.True(t, found.IsActive)
	assert.Nil(t, found.LastUsedAt)

	// Unknown hash returns (nil, nil), not an error
	missing, err := store.GetServiceTokenByHash(ctx, encrypt.HashSHA256("bfsvc_wrong"))
	require.NoError(t, err)
	assert.Nil(t, missing)
}

func TestServiceToken_GetByHash_Inactive(t *testing.T) {
	store := setupServiceTokenTestStore(t)
	ctx := context.Background()

	token := newTestServiceToken("revoked", "bfsvc_revoked", nil)
	token.IsActive = false
	require.NoError(t, store.CreateServiceToken(ctx, token))

	found, err := store.GetServiceTokenByHash(ctx, encrypt.HashSHA256("bfsvc_revoked"))
	require.NoError(t, err)
	assert.Nil(t, found, "inactive tokens must not validate")
}

func TestServiceToken_GetByHash_Expired(t *testing.T) {
	store := setupServiceTokenTestStore(t)
	ctx := context.Background()

	past := time.Now().Add(-time.Hour)
	token := newTestServiceToken("expired", "bfsvc_expired", &past)
	require.NoError(t, store.CreateServiceToken(ctx, token))

	found, err := store.GetServiceTokenByHash(ctx, encrypt.HashSHA256("bfsvc_expired"))
	require.NoError(t, err)
	assert.Nil(t, found, "expired tokens must not validate")

	// A token expiring in the future still validates
	future := time.Now().Add(time.Hour)
	token2 := newTestServiceToken("future", "bfsvc_future", &future)
	require.NoError(t, store.CreateServiceToken(ctx, token2))
	found2, err := store.GetServiceTokenByHash(ctx, encrypt.HashSHA256("bfsvc_future"))
	require.NoError(t, err)
	assert.NotNil(t, found2)
}

func TestServiceToken_ListAndDelete(t *testing.T) {
	store := setupServiceTokenTestStore(t)
	ctx := context.Background()

	require.NoError(t, store.CreateServiceToken(ctx, newTestServiceToken("one", "bfsvc_one", nil)))
	require.NoError(t, store.CreateServiceToken(ctx, newTestServiceToken("two", "bfsvc_two", nil)))

	tokens, err := store.ListServiceTokens(ctx)
	require.NoError(t, err)
	require.Len(t, tokens, 2)

	require.NoError(t, store.DeleteServiceToken(ctx, tokens[0].ID))
	tokens, err = store.ListServiceTokens(ctx)
	require.NoError(t, err)
	require.Len(t, tokens, 1)

	// Deleting an unknown ID surfaces ErrNotFound
	err = store.DeleteServiceToken(ctx, 99999)
	require.ErrorIs(t, err, ErrNotFound)
}

func TestServiceToken_TouchLastUsed(t *testing.T) {
	store := setupServiceTokenTestStore(t)
	ctx := context.Background()

	token := newTestServiceToken("touchable", "bfsvc_touchable", nil)
	require.NoError(t, store.CreateServiceToken(ctx, token))

	require.NoError(t, store.TouchServiceTokenLastUsed(ctx, token.ID))

	tokens, err := store.ListServiceTokens(ctx)
	require.NoError(t, err)
	require.Len(t, tokens, 1)
	require.NotNil(t, tokens[0].LastUsedAt)
	assert.WithinDuration(t, time.Now(), *tokens[0].LastUsedAt, time.Minute)
}
