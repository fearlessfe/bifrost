import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { Trash } from "lucide-react";
import { useState } from "react";

interface RenameRow {
	model: string;
	from: string;
	to: string;
}

interface EffortRenamesByModelTableProps {
	value?: Record<string, Record<string, string>>;
	onChange: (value: Record<string, Record<string, string>>) => void;
	label?: string;
	disabled?: boolean;
	modelPlaceholder?: string;
	fromPlaceholder?: string;
	toPlaceholder?: string;
}

// Flatten {model: {from: to}} into editable rows, skipping empty keys.
function toRows(value?: Record<string, Record<string, string>>): RenameRow[] {
	const rows: RenameRow[] = [];
	for (const [model, renames] of Object.entries(value || {})) {
		if (!model) continue;
		for (const [from, to] of Object.entries(renames || {})) {
			if (!from || !to) continue;
			rows.push({ model, from, to });
		}
	}
	return rows;
}

// Rebuild the nested record from rows; incomplete rows are dropped (a mapping
// with an empty side is meaningless on the wire).
function toRecord(rows: RenameRow[]): Record<string, Record<string, string>> {
	const record: Record<string, Record<string, string>> = {};
	for (const row of rows) {
		if (!row.model || !row.from || !row.to) continue;
		(record[row.model] ??= {})[row.from] = row.to;
	}
	return record;
}

/**
 * Three-column editor for custom_provider_config.reasoning_effort_renames_by_model:
 * each row maps one OpenAI effort label to the upstream's label for one model.
 * Committed rows edit in place; the trailing draft row commits once all three
 * cells are filled.
 */
export function EffortRenamesByModelTable({
	value,
	onChange,
	label = "Reasoning Effort Renames by Model",
	disabled = false,
	modelPlaceholder = "Model (e.g. glm-5.3)",
	fromPlaceholder = "OpenAI effort (e.g. medium)",
	toPlaceholder = "Upstream effort (e.g. high)",
}: EffortRenamesByModelTableProps) {
	const rows = toRows(value);
	const [draft, setDraft] = useState<RenameRow>({ model: "", from: "", to: "" });

	const commit = (next: RenameRow[]) => onChange(toRecord(next));

	const updateRow = (index: number, patch: Partial<RenameRow>) => {
		commit(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
	};

	const updateDraft = (patch: Partial<RenameRow>) => {
		const next = { ...draft, ...patch };
		if (next.model && next.from && next.to) {
			commit([...rows, next]);
			setDraft({ model: "", from: "", to: "" });
		} else {
			setDraft(next);
		}
	};

	const cellInput = (val: string, placeholder: string, onValue: (v: string) => void) => (
		<Input
			className={cn("h-8", disabled && "cursor-not-allowed opacity-50")}
			value={val}
			placeholder={placeholder}
			disabled={disabled}
			onChange={(e) => onValue(e.target.value)}
		/>
	);

	return (
		<div className="w-full">
			{label && (
				<label className="mb-2 block text-sm leading-none font-medium peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
					{label}
				</label>
			)}
			<div className="rounded-md border">
				<Table>
					<TableHeader>
						<TableRow>
							<TableHead className="w-[32%]">Model</TableHead>
							<TableHead className="w-[28%]">OpenAI effort</TableHead>
							<TableHead className="w-[28%]">Upstream effort</TableHead>
							<TableHead className="w-[12%]" />
						</TableRow>
					</TableHeader>
					<TableBody>
						{rows.map((row, index) => (
							<TableRow key={`${row.model}-${row.from}-${index}`}>
								<TableCell>{cellInput(row.model, modelPlaceholder, (v) => updateRow(index, { model: v }))}</TableCell>
								<TableCell>{cellInput(row.from, fromPlaceholder, (v) => updateRow(index, { from: v }))}</TableCell>
								<TableCell>{cellInput(row.to, toPlaceholder, (v) => updateRow(index, { to: v }))}</TableCell>
								<TableCell>
									<button
										type="button"
										className="text-muted-foreground hover:text-destructive"
										disabled={disabled}
										onClick={() => commit(rows.filter((_, i) => i !== index))}
									>
										<Trash className="h-4 w-4" />
									</button>
								</TableCell>
							</TableRow>
						))}
						{!disabled && (
							<TableRow>
								<TableCell>{cellInput(draft.model, modelPlaceholder, (v) => updateDraft({ model: v }))}</TableCell>
								<TableCell>{cellInput(draft.from, fromPlaceholder, (v) => updateDraft({ from: v }))}</TableCell>
								<TableCell>{cellInput(draft.to, toPlaceholder, (v) => updateDraft({ to: v }))}</TableCell>
								<TableCell />
							</TableRow>
						)}
						{disabled && rows.length === 0 && (
							<TableRow>
								<TableCell colSpan={4} className="text-muted-foreground h-8 text-center text-sm">
									No model-specific renames — the provider-wide map applies to every model
								</TableCell>
							</TableRow>
						)}
					</TableBody>
				</Table>
			</div>
		</div>
	);
}