#!/bin/bash

# Generates a graphviz dependency graph for :ruy, with details trimmed.
# Suggested rendering: pipe to `neato` (part of graphviz standard distribution)
#   doc/depgraph.sh | dot -Tsvg > depgraph.svg

drop=(
    ':platform'
    ':check_macros'
    ':asm_helpers'
    ':size_util'
    ':system_aligned_alloc'
    ':side_pair'
    ':opt_set'
    ':blocking_counter'
    ':wait'
    ':time'
    ':path'
    ':performance_advisory'
    ':tune'
    ':matrix'
    ':mat'
    ':mul_params'
    ':context_get_ctx'
    ':have_built_path_for'
    ':pack_common'
    ':kernel_common'
    'profiler:instrumentation'
    '\bclog\b'
    '\bcpuinfo_impl\b'
    ':apply_multiplier'
    '\blabel='
)

graph="$(bazel query 'kind("cc_library", deps(//ruy))' --output graph --noimplicit_deps 2>/dev/null)"

graph="$(echo "${graph}" | sed 's|//ruy/\?||g')"

for t in "${drop[@]}"; do
  graph="$(echo "${graph}" | grep -v "${t}")"
done

graph="$(echo "${graph}" | sed 's|//:cpuinfo_with_unstripped_include_path||g')"
graph="$(echo "${graph}" | sed 's|//third_party/cpuinfo:[a-z0-9_]*|@cpuinfo|g')"

frontend=(
    ':ruy'
    ':context'
    ':frontend'
    ':prepare_packed_matrices'
    ':create_trmul_params'
    ':validate'
)

middleend=(
    ':ctx'
    ':trmul_params'
    ':trmul'
    ':block_map'
    ':cpuinfo'
    ':cpu_cache_params'
    ':allocator'
    ':thread_pool'
    ':prepacked_cache'
)

backend=(
    ':kernel.*'
    ':pack.*'
)

frontend_lines=()
middleend_lines=()
backend_lines=()
misc_lines=()
arrow_lines=()

while IFS= read -r line; do
  if [[ "${line}" =~ '->' ]]; then
    arrow_lines+=("${line}")
  else
    handled=false
    if [ $handled = false ]; then
        for f in "${frontend[@]}"; do
            if [[ "${line}" =~ ${f} ]]; then
                frontend_lines+=("${line}")
                handled=true
                break
            fi
        done
    fi
    if [ $handled = false ]; then
        for f in "${middleend[@]}"; do
            if [[ "${line}" =~ ${f} ]]; then
                middleend_lines+=("${line}")
                handled=true
                break
            fi
        done
    fi
    if [ $handled = false ]; then
        for f in "${backend[@]}"; do
            if [[ "${line}" =~ ${f} ]]; then
                backend_lines+=("${line}")
                handled=true
                break
            fi
        done
    fi
    if [ $handled = false ]; then
        if [[ "${line}" =~ ^[[:space:]]+\" ]]; then
            misc_lines+=("${line}")
        fi
    fi
  fi
done <<< "${graph}"

echo "digraph ruy {"
echo "  splines = true"
echo "  node [shape=box]"
for f in "${frontend_lines[@]}"; do
  echo "  $f [style=filled, color=lightblue];"
done
for m in "${middleend_lines[@]}"; do
  echo "  $m [style=filled, color=lightgreen];"
done
for b in "${backend_lines[@]}"; do
  echo "  $b [style=filled, color=indianred1];"
done
for m in "${misc_lines[@]}"; do
  echo "$m"
done
for a in "${arrow_lines[@]}"; do
  echo "$a"
done
echo "  \":create_trmul_params\" -> \":trmul\" [style=invis]"
echo "  subgraph cluster_legend_margin {"
echo "    style=invis"
echo "    margin=80"
echo "    subgraph cluster_legend {"
echo "      style=\"\""
echo "      label=\"Legend\""
echo "      fontsize=20"
echo "      margin=20"
echo "      labelloc=t"
echo "      frontend [label=\"Front-end\", style=filled, color=lightblue]"
echo "      middleend [label=\"Middle-end\", style=filled, color=lightgreen]"
echo "      backend [label=\"Back-end\", style=filled, color=indianred1]"
echo "      frontend -> middleend -> backend [style=invis]"
echo "    }"
echo "  }"
echo "}"
