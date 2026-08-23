import { baseApi } from "./baseApi";

// Service tokens are long-lived admin-equivalent credentials (bfsvc_ prefix)
// managed via /api/service-tokens. The plaintext token is returned exactly
// once at creation time; list responses never include token values or hashes.
export interface ServiceToken {
	id: number;
	name: string;
	is_active: boolean;
	created_at: string;
	updated_at?: string;
	expires_at?: string | null;
	last_used_at?: string | null;
}

export interface ListServiceTokensResponse {
	service_tokens: ServiceToken[];
}

export interface CreateServiceTokenRequest {
	name: string;
	expires_at?: string | null;
}

export interface CreateServiceTokenResponse extends ServiceToken {
	token: string;
}

export interface DeleteServiceTokenResponse {
	message: string;
}

export const serviceTokensApi = baseApi.injectEndpoints({
	overrideExisting: false,
	endpoints: (builder) => ({
		// List all service tokens (no token values or hashes are returned)
		getServiceTokens: builder.query<ListServiceTokensResponse, void>({
			query: () => ({
				url: "/service-tokens",
				method: "GET",
			}),
			providesTags: ["ServiceTokens"],
		}),
		// Create a service token — the plaintext token appears only in this response
		createServiceToken: builder.mutation<CreateServiceTokenResponse, CreateServiceTokenRequest>({
			query: (payload) => ({
				url: "/service-tokens",
				method: "POST",
				body: payload,
			}),
			invalidatesTags: ["ServiceTokens"],
		}),
		// Revoke a service token
		deleteServiceToken: builder.mutation<DeleteServiceTokenResponse, number>({
			query: (id) => ({
				url: `/service-tokens/${id}`,
				method: "DELETE",
			}),
			invalidatesTags: ["ServiceTokens"],
		}),
	}),
});

export const { useGetServiceTokensQuery, useCreateServiceTokenMutation, useDeleteServiceTokenMutation } = serviceTokensApi;