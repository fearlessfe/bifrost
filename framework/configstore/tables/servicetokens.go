package tables

import (
	"time"
)

// ServiceTokenPrefix is the prefix of all service tokens. It allows the auth
// middleware to fast-path reject non-service-token Bearer credentials without
// a database lookup.
const ServiceTokenPrefix = "bfsvc_"

// ServiceTokensTable represents a long-lived service token that grants
// admin-equivalent access to the management API (same as Basic auth).
//
// Only the SHA-256 hash of the token is stored — the plaintext is returned
// exactly once at creation time and is never persisted, so there is no
// encryption hook (unlike SessionsTable, which stores the plaintext).
type ServiceTokensTable struct {
	ID         uint       `gorm:"primaryKey;autoIncrement" json:"id"`
	Name       string     `gorm:"type:varchar(255);not null" json:"name"`
	TokenHash  string     `gorm:"type:varchar(64);not null;uniqueIndex:idx_service_tokens_token_hash" json:"-"`
	Scopes     string     `gorm:"type:text" json:"scopes,omitempty"` // JSON array, reserved for future route-level scoping
	ExpiresAt  *time.Time `gorm:"index" json:"expires_at,omitempty"`
	LastUsedAt *time.Time `json:"last_used_at,omitempty"`
	IsActive   bool       `gorm:"not null" json:"is_active"`
	CreatedAt  time.Time  `gorm:"index;not null" json:"created_at"`
	UpdatedAt  time.Time  `gorm:"index;not null" json:"updated_at"`
}

// TableName sets the table name for each model
func (ServiceTokensTable) TableName() string { return "service_tokens" }
