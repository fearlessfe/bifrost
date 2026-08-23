import { Alert, AlertDescription } from "@/components/ui/alert";
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
} from "@/components/ui/alertDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useCopyToClipboard } from "@/hooks/useCopyToClipboard";
import {
	getErrorMessage,
	useCreateServiceTokenMutation,
	useDeleteServiceTokenMutation,
	useGetCoreConfigQuery,
	useGetServiceTokensQuery,
	type CreateServiceTokenResponse,
	type ServiceToken,
} from "@/lib/store";
import { serviceTokenFormSchema, type ServiceTokenFormSchema } from "@/lib/types/schemas";
import { zodResolver } from "@hookform/resolvers/zod";
import { Link } from "@tanstack/react-router";
import { Copy, InfoIcon, KeyRound, Plus, Trash2, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import ContactUsView from "../views/contactUsView";

export default function APIKeysView() {
	const { data: bifrostConfig, isLoading } = useGetCoreConfigQuery({ fromDB: true });
	const isAuthConfigure = useMemo(() => {
		return bifrostConfig?.auth_config?.is_enabled;
	}, [bifrostConfig]);

	const curlExample = `# Base64 encode your username:password
# Example: echo -n "username:password" | base64
curl --location 'http://localhost:8080/v1/chat/completions'
--header 'Content-Type: application/json' 
--header 'Accept: application/json' 
--header 'Authorization: Basic <base64_encoded_username:password>' 
--data '{ 
  "model": "openai/gpt-4", 
  "messages": [ 
    { 
      "role": "user", 
      "content": "explain big bang?" 
    } 
  ] 
}'`;

	const { copy: copyToClipboard } = useCopyToClipboard();

	if (isLoading) {
		return <div>Loading...</div>;
	}
	if (!isAuthConfigure) {
		return (
			<Alert variant="default">
				<InfoIcon className="text-muted h-4 w-4" />
				<AlertDescription>
					<p className="text-md text-muted-foreground">
						To generate API keys, you need to set up admin username and password first.{" "}
						<Link to="/workspace/config/security" className="text-md text-primary underline">
							Configure Security Settings
						</Link>
						.<br />
						<br />
						Once generated you will need to use this API key for all API calls to the Bifrost admin APIs and UI.
					</p>
				</AlertDescription>
			</Alert>
		);
	}

	const isInferenceAuthDisabled = !(bifrostConfig?.client_config?.enforce_auth_on_inference ?? false);

	return (
		<div className="mx-auto w-full max-w-4xl space-y-4">
			<Alert variant="default">
				<InfoIcon className="text-muted h-4 w-4" />
				<AlertDescription>
					<p className="text-md text-muted-foreground">
						{isInferenceAuthDisabled ? (
							<>
								Authentication is currently <strong>disabled for inference API calls</strong>. You can make inference requests without
								authentication. Dashboard and admin API calls still require Basic auth with your admin credentials encoded in the standard{" "}
								<code className="bg-muted rounded px-1 py-0.5 text-sm">username:password</code> format with base64 encoding.
							</>
						) : (
							<>
								Use Basic auth with your admin credentials when making API calls to Bifrost. Encode your credentials in the standard{" "}
								<code className="bg-muted rounded px-1 py-0.5 text-sm">username:password</code> format with base64 encoding.
							</>
						)}
					</p>
					{!isInferenceAuthDisabled && (
						<>
							<br />
							<p className="text-md text-muted-foreground">
								<strong>Example:</strong>
							</p>

							<div className="relative mt-2 w-full min-w-0 overflow-x-auto">
								<Button variant="ghost" size="sm" onClick={() => copyToClipboard(curlExample)} className="absolute top-2 right-2 z-10 h-8">
									<Copy className="h-4 w-4" />
								</Button>
								<pre className="bg-muted min-w-max rounded p-3 pr-12 font-mono text-sm whitespace-pre">{curlExample}</pre>
							</div>
						</>
					)}
				</AlertDescription>
			</Alert>

			<ServiceTokensSection />

			<ContactUsView
				className="mt-4 rounded-md border px-3 py-8"
				icon={<KeyRound size={48} />}
				title="Scope Based API Keys"
				description="Need granular access control with scope-based API keys? Enterprise customers can create multiple API keys with specific permissions for different services, teams, or environments."
				readmeLink="https://docs.getbifrost.io/enterprise/api-keys"
			/>
		</div>
	);
}

// Service tokens are long-lived admin-equivalent credentials for the
// management API (same trust level as Basic auth). The plaintext token is
// shown exactly once in the creation dialog; afterwards only metadata is
// available.
function ServiceTokensSection() {
	const { data, isLoading } = useGetServiceTokensQuery();
	const [deleteServiceToken, { isLoading: isDeleting }] = useDeleteServiceTokenMutation();
	const [isCreateOpen, setIsCreateOpen] = useState(false);
	const [createdToken, setCreatedToken] = useState<CreateServiceTokenResponse | null>(null);
	const [tokenToDelete, setTokenToDelete] = useState<ServiceToken | null>(null);

	const tokens = useMemo(() => data?.service_tokens ?? [], [data]);

	const handleCreated = (token: CreateServiceTokenResponse) => {
		setIsCreateOpen(false);
		setCreatedToken(token);
	};

	const handleDelete = async () => {
		if (!tokenToDelete) return;
		try {
			await deleteServiceToken(tokenToDelete.id).unwrap();
			toast.success("Service token deleted.");
			setTokenToDelete(null);
		} catch (error) {
			toast.error(`Failed to delete service token: ${getErrorMessage(error)}`);
		}
	};

	return (
		<Card>
			<CardHeader className="flex flex-row items-start justify-between gap-4">
				<div className="space-y-1.5">
					<CardTitle>Service Tokens</CardTitle>
					<CardDescription>
						Long-lived tokens that grant admin-equivalent access to the management API, for CI pipelines and automation. Pass them as a{" "}
						<code className="bg-muted rounded px-1 py-0.5 text-xs">Bearer</code> token in the{" "}
						<code className="bg-muted rounded px-1 py-0.5 text-xs">Authorization</code> header.
					</CardDescription>
				</div>
				<Button size="sm" onClick={() => setIsCreateOpen(true)} data-testid="service-tokens-create-button">
					<Plus className="h-4 w-4" />
					Create Token
				</Button>
			</CardHeader>
			<CardContent>
				<Table containerClassName="rounded-sm border" data-testid="service-tokens-table">
					<TableHeader>
						<TableRow>
							<TableHead>Name</TableHead>
							<TableHead>Status</TableHead>
							<TableHead>Created</TableHead>
							<TableHead>Last used</TableHead>
							<TableHead>Expires</TableHead>
							<TableHead className="w-[56px] text-right" />
						</TableRow>
					</TableHeader>
					<TableBody>
						{isLoading ? (
							<TableRow>
								<TableCell colSpan={6} className="text-muted-foreground py-6 text-center">
									Loading service tokens...
								</TableCell>
							</TableRow>
						) : tokens.length === 0 ? (
							<TableRow>
								<TableCell colSpan={6} className="py-6 text-center">
									<p className="text-muted-foreground text-sm" data-testid="service-tokens-empty-state">
										No service tokens yet. Create one to authenticate automation against the management API without your admin password.
									</p>
								</TableCell>
							</TableRow>
						) : (
							tokens.map((token) => (
								<TableRow key={token.id} data-testid={`service-tokens-row-${token.id}`}>
									<TableCell className="max-w-[220px]">
										<span className="block truncate font-medium" title={token.name}>
											{token.name}
										</span>
									</TableCell>
									<TableCell>
										<ServiceTokenStatusBadge token={token} />
									</TableCell>
									<TableCell className="text-muted-foreground text-sm">{formatDate(token.created_at)}</TableCell>
									<TableCell className="text-muted-foreground text-sm">
										{token.last_used_at ? formatDate(token.last_used_at) : "Never"}
									</TableCell>
									<TableCell className="text-muted-foreground text-sm">
										{token.expires_at ? formatDate(token.expires_at) : "Never"}
									</TableCell>
									<TableCell className="text-right">
										<Button
											type="button"
											variant="ghost"
											size="icon"
											disabled={isDeleting}
											aria-label={`Delete service token ${token.name}`}
											onClick={() => setTokenToDelete(token)}
											data-testid={`service-tokens-delete-${token.id}`}
										>
											<Trash2 className="text-destructive h-4 w-4" />
										</Button>
									</TableCell>
								</TableRow>
							))
						)}
					</TableBody>
				</Table>
			</CardContent>

			<CreateServiceTokenDialog open={isCreateOpen} onOpenChange={setIsCreateOpen} onCreated={handleCreated} />

			<ServiceTokenCreatedDialog token={createdToken} onClose={() => setCreatedToken(null)} />

			<AlertDialog open={tokenToDelete !== null} onOpenChange={(open) => !open && setTokenToDelete(null)}>
				<AlertDialogContent>
					<AlertDialogHeader>
						<AlertDialogTitle>Delete this service token?</AlertDialogTitle>
						<AlertDialogDescription>
							This action cannot be undone. Any automation using <strong>{tokenToDelete?.name}</strong> will immediately lose access to the
							management API.
						</AlertDialogDescription>
					</AlertDialogHeader>
					<AlertDialogFooter>
						<AlertDialogCancel data-testid="service-tokens-delete-cancel">Cancel</AlertDialogCancel>
						<AlertDialogAction data-testid="service-tokens-delete-confirm" onClick={handleDelete} disabled={isDeleting}>
							Delete
						</AlertDialogAction>
					</AlertDialogFooter>
				</AlertDialogContent>
			</AlertDialog>
		</Card>
	);
}

function CreateServiceTokenDialog({
	open,
	onOpenChange,
	onCreated,
}: {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	onCreated: (token: CreateServiceTokenResponse) => void;
}) {
	const [createServiceToken, { isLoading: isCreating }] = useCreateServiceTokenMutation();
	const form = useForm<ServiceTokenFormSchema>({
		resolver: zodResolver(serviceTokenFormSchema),
		mode: "onChange",
		defaultValues: { name: "" },
	});

	const handleOpenChange = (nextOpen: boolean) => {
		onOpenChange(nextOpen);
		if (!nextOpen) {
			form.reset({ name: "" });
		}
	};

	const onSubmit = async (values: ServiceTokenFormSchema) => {
		try {
			// expires_at is not exposed in the UI yet — tokens never expire.
			const created = await createServiceToken({ name: values.name.trim(), expires_at: null }).unwrap();
			onCreated(created);
			form.reset({ name: "" });
		} catch (error) {
			toast.error(`Failed to create service token: ${getErrorMessage(error)}`);
		}
	};

	return (
		<Dialog open={open} onOpenChange={handleOpenChange}>
			<DialogContent data-testid="service-tokens-create-dialog">
				<DialogHeader>
					<DialogTitle>Create Service Token</DialogTitle>
					<DialogDescription>
						The token grants admin-equivalent access to the management API. It is shown only once, right after creation.
					</DialogDescription>
				</DialogHeader>
				<Form {...form}>
					<form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
						<FormField
							control={form.control}
							name="name"
							render={({ field }) => (
								<FormItem>
									<FormLabel>Name</FormLabel>
									<FormControl>
										<Input placeholder="e.g. ci-pipeline" autoFocus data-testid="service-tokens-name-input" {...field} />
									</FormControl>
									<FormMessage />
								</FormItem>
							)}
						/>
						<DialogFooter>
							<Button type="button" variant="outline" onClick={() => handleOpenChange(false)} data-testid="service-tokens-create-cancel">
								Cancel
							</Button>
							<Button type="submit" disabled={isCreating || !form.formState.isValid} data-testid="service-tokens-create-submit">
								{isCreating ? "Creating..." : "Create Token"}
							</Button>
						</DialogFooter>
					</form>
				</Form>
			</DialogContent>
		</Dialog>
	);
}

// Shown exactly once after creation — the plaintext token is never retrievable
// again, so the dialog blocks dismissal until the user confirms.
function ServiceTokenCreatedDialog({ token, onClose }: { token: CreateServiceTokenResponse | null; onClose: () => void }) {
	const { copy: copyToClipboard, copied } = useCopyToClipboard({ successMessage: "Token copied to clipboard" });

	return (
		<Dialog open={token !== null} onOpenChange={(open) => !open && onClose()}>
			<DialogContent data-testid="service-tokens-created-dialog">
				<DialogHeader>
					<DialogTitle>Service Token Created</DialogTitle>
					<DialogDescription>Copy the token below and store it somewhere safe.</DialogDescription>
				</DialogHeader>
				<Alert variant="destructive">
					<TriangleAlert className="h-4 w-4" />
					<AlertDescription>
						This token is shown <strong>only once</strong>. Once you close this dialog, it cannot be viewed again.
					</AlertDescription>
				</Alert>
				<div className="relative w-full min-w-0">
					<Button
						variant="ghost"
						size="sm"
						onClick={() => token && copyToClipboard(token.token)}
						className="absolute top-2 right-2 z-10 h-8"
						data-testid="service-tokens-token-copy"
					>
						<Copy className="h-4 w-4" />
						{copied ? "Copied" : "Copy"}
					</Button>
					<pre
						className="bg-muted w-full overflow-x-auto rounded p-3 pr-20 font-mono text-sm break-all whitespace-pre-wrap"
						data-testid="service-tokens-token-value"
					>
						{token?.token}
					</pre>
				</div>
				<DialogFooter>
					<Button onClick={onClose} data-testid="service-tokens-token-close">
						Done
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}

function ServiceTokenStatusBadge({ token }: { token: ServiceToken }) {
	if (!token.is_active) {
		return <Badge variant="secondary">Inactive</Badge>;
	}
	if (token.expires_at && new Date(token.expires_at).getTime() <= Date.now()) {
		return <Badge variant="destructive">Expired</Badge>;
	}
	return <Badge variant="success">Active</Badge>;
}

function formatDate(iso: string): string {
	const date = new Date(iso);
	if (Number.isNaN(date.getTime())) return iso;
	return date.toLocaleString(undefined, {
		year: "numeric",
		month: "short",
		day: "numeric",
		hour: "2-digit",
		minute: "2-digit",
	});
}