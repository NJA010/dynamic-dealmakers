<script context="module">
	export const evidenceInclude = true;
</script>

<script>
	import Select from 'svelte-select';
	import { INPUTS_CONTEXT_KEY } from '@evidence-dev/component-utilities/globalContexts';
	import { buildInputQuery } from '@evidence-dev/component-utilities/buildQuery';
	import {getContext, onMount, setContext} from 'svelte';
	import { page } from '$app/stores';

	/////
	// Component Things
	/////

	/** @type {string} */
	export let title;

	/** @type {string} */
	export let name;
	let values = [];
	let checked = [];
	let isChecked = {};

	const inputs = getContext(INPUTS_CONTEXT_KEY);

	setContext('multiselect_context', {
		hasBeenSet: false,
		setSelectedValue: (selected) => ($inputs[name] = selected)
	});

	$: onMount(async () => {
		$inputs[name] = "''";
	})
	/////
	// Query-Related Things
	/////

	export let value, data, label;
	/** @type {import("@evidence-dev/component-utilities/buildQuery.js").QueryProps}*/
	$: ({ hasQuery, query } = buildInputQuery(
		{ value, data, label },
		`Multiselect-${name}`,
		$page.data.data[`Multiselect-${name}`]
	));

</script>

<div class="mt-2 mb-4 mx-1 inline-block">
	{#if title}
		<span class="text-sm text-gray-500 block">{title}</span>
	{/if}

	<!--
	do not switch to binding, select bind:value invalidates its dependencies 
	(so `data` would be invalidated) 
-->
	{#if hasQuery && $query.error}
		<span
			class="group inline-flex items-center relative cursor-help cursor-helpfont-sans px-1 border border-red-200 py-[1px] bg-red-50 rounded"
		>
			<span class="inline font-sans font-medium text-xs text-red-600">error</span>
			<span
				class="hidden text-white font-sans group-hover:inline absolute -top-1 left-[105%] text-sm z-10 px-2 py-1 bg-gray-800/80 leading-relaxed min-w-[150px] w-max max-w-[400px] rounded-md"
			>
				{$query.error}
			</span>
		</span>
	{:else}
		<Select
			disabled={hasQuery && !$query.loaded}
			items={$query}
			on:select={(e) => {values.push(e.detail.value); $inputs[name] = "'" + values.join("','") + "'" }}
			on:clear={(e) => {values.splice(values.indexOf(e.detail), 1); $inputs[name] = "'" + values.join("','") + "'" }}
			searchable="true"
			class="border border-gray-300 bg-white rounded-lg p-1 mt-2 px-2 pr-20 flex flex-row items-center max-w-fit bg-transparent cursor-pointer bg-right bg-no-repeat"
			multiple>

			<div class="item" slot="item" let:item>
				<label for={item.value}>
					<input type="checkbox" id={item.value} bind:checked={isChecked[item.value]} />
					{item.label}
				</label>
			</div>

		</Select>
	{/if}
</div>
