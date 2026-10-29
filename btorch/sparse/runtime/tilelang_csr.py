"""Render the shared segmented CSR algorithms using public TileLang APIs.

Tiles, split-row tables and list capacities are selected by
csr_execution, identically to Triton. These factories only express the
device stages.
"""

import tilelang
import tilelang.language as T


@T.macro
def emit_spike_list(selected, degree, indices, cumulative, counts, group, n_in, BLOCK):
    """Emit block-local source indices and int64 fanout prefixes.

    Every thread participates, including masked tail lanes. Producers
    supply zero for inactive lanes and own clearing their receptor
    accumulation.
    """
    rank = T.alloc_fragment((BLOCK,), "int32")
    count = T.alloc_fragment((1,), "int32")
    fanout = T.alloc_fragment((BLOCK,), "int64")
    prefix = T.alloc_fragment((BLOCK,), "int64")
    for lane in T.Parallel(BLOCK):
        fanout[lane] = T.if_then_else(
            selected[lane] != 0, degree[(group * BLOCK + lane) % n_in], 0
        )
    # A separate reduction avoids replicating the entire scan in each thread
    # merely to read its last element for the block's active count.
    T.reduce_sum(selected, count, dim=0)
    T.cumsum(selected, rank)
    T.cumsum(fanout, prefix)
    for lane in T.Parallel(BLOCK):
        if selected[lane] != 0:
            at = group * BLOCK + rank[lane] - 1
            indices[at] = group * BLOCK + lane
            cumulative[at] = prefix[lane]
    if T.get_thread_binding() == 0:
        counts[group] = count[0]


@tilelang.jit
def matvec(shapes, dtypes, scalars, tiles, grid, threads):
    n_seg, n_rows, n_sample, n_sample_block, n_tmp, val_stride, x_stride = scalars
    BR, BK, BS, HUB, PERM, MASK, WEIGHT_GRAD, LIST, SEGMENT = tiles
    compute = "float64" if dtypes[6] == "float64" else "float32"
    value_compute = "float64" if dtypes[11] == "float64" else "float32"

    @T.prim_func
    def kernel(
        ptr: T.Tensor(shapes[0], dtypes[0]),
        target: T.Tensor(shapes[1], dtypes[1]),
        col: T.Tensor(shapes[2], dtypes[2]),
        perm: T.Tensor(shapes[3], dtypes[3]),
        values: T.Tensor(shapes[4], dtypes[4]),
        x: T.Tensor(shapes[5], dtypes[5]),
        output: T.Tensor(shapes[6], dtypes[6]),
        partial: T.Tensor(shapes[7], dtypes[7]),
        mask: T.Tensor(shapes[8], dtypes[8]),
        owner: T.Tensor(shapes[9], dtypes[9]),
        activity: T.Tensor(shapes[10], dtypes[10]),
        grad_values: T.Tensor(shapes[11], dtypes[11]),
        indices: T.Tensor(shapes[12], dtypes[12]),
        counts: T.Tensor(shapes[13], dtypes[13]),
    ):
        with T.Kernel(grid[0], grid[1], grid[2], threads=threads) as (
            pid,
            bid,
            partition,
        ):
            member = T.Cast("int64", bid // n_sample_block)
            sample_base = bid % n_sample_block * BS
            seg = T.alloc_fragment((BR,), "int32")
            live = T.alloc_fragment((BR,), "bool")
            list_sample = T.alloc_fragment((BR,), "int32")
            row_start = T.alloc_fragment((BR,), "int32")
            row_end = T.alloc_fragment((BR,), "int32")
            start = T.alloc_fragment((BR,), "int32")
            length = T.alloc_fragment((BR,), "int32")
            longest = T.alloc_fragment((1,), "int32")
            row = T.alloc_fragment((BR,), "int32")
            active = T.alloc_fragment((BS, BR), "int32")
            any_active = T.alloc_fragment((BR,), "int32")
            amplitude = T.alloc_fragment((BS, BR), dtypes[10])
            acc = T.alloc_fragment((BS, BR, BK), compute)
            total = T.alloc_fragment((BS, BR), compute)
            increments = T.alloc_fragment((BS, BR, BK), compute)
            increment = T.alloc_fragment((BR, BK), compute)
            for r in T.Parallel(BR):
                seg[r] = pid * BR + r
                live[r] = seg[r] < n_seg
                list_sample[r] = 0
                if LIST:
                    rank = pid % (128 // BR) * BR + r
                    group = pid // (128 // BR)
                    live[r] = rank < counts[group]
                    seg[r] = T.if_then_else(live[r], indices[group * 128 + rank], 0)
                    list_sample[r] = seg[r] // n_rows
                    seg[r] = seg[r] % n_rows
                row_start[r] = T.if_then_else(live[r], ptr[seg[r]], 0)
                row_end[r] = T.if_then_else(live[r], ptr[seg[r] + 1], 0)
                length[r] = row_end[r] - row_start[r]
            T.reduce_max(length, longest, dim=0)
            for chunk in T.serial(
                partition if LIST else 0,
                T.max(1, T.ceildiv(longest[0], SEGMENT)) if LIST else 1,
                grid[2] if LIST else 1,
            ):
                for r in T.Parallel(BR):
                    start[r] = row_start[r]
                    length[r] = row_end[r] - row_start[r]
                    if LIST:
                        start[r] = T.min(row_start[r] + chunk * SEGMENT, row_end[r])
                        length[r] = T.min(row_end[r], start[r] + SEGMENT) - start[r]
                    row[r] = seg[r]
                    if HUB:
                        row[r] = T.if_then_else(live[r], owner[seg[r]], -1)
                for s, r in T.Parallel(BS, BR):
                    active[s, r] = 1
                    if MASK:
                        at = (
                            list_sample[r]
                            if LIST
                            else member * n_sample + sample_base + s
                        ) * n_rows + row[r]
                        active[s, r] = T.if_then_else(
                            sample_base + s < n_sample and live[r] and row[r] >= 0,
                            T.Cast("int32", mask[at] != 0),
                            0,
                        )
                    if WEIGHT_GRAD:
                        amplitude[s, r] = T.if_then_else(
                            sample_base + s < n_sample and live[r] and row[r] >= 0,
                            activity[row[r] * n_sample + sample_base + s],
                            0,
                        )
                T.reduce_sum(active, any_active, dim=0)
                if MASK and not WEIGHT_GRAD:
                    for r in T.Parallel(BR):
                        length[r] = T.if_then_else(any_active[r] > 0, length[r], 0)
                T.reduce_max(length, longest, dim=0)
                T.clear(acc)
                for block in T.serial(T.ceildiv(longest[0], BK)):
                    for s, r, k in T.Parallel(BS, BR, BK):
                        pos = block * BK + k
                        edge = start[r] + pos
                        edge_live = pos < length[r]
                        full = edge_live and sample_base + s < n_sample
                        source = T.if_then_else(edge_live, col[edge], 0)
                        offset = source * n_sample + (
                            list_sample[r] if LIST else sample_base + s
                        )
                        xv = T.if_then_else(
                            full and (not MASK or WEIGHT_GRAD or active[s, r] != 0),
                            x[member * x_stride + offset],
                            0,
                        )
                        slot = (
                            T.if_then_else(
                                edge_live and (not MASK or any_active[r] > 0),
                                perm[edge],
                                0,
                            )
                            if PERM
                            else edge
                        )
                        weight = T.if_then_else(
                            edge_live and (not MASK or not PERM or any_active[r] > 0),
                            values[member * val_stride + slot],
                            0,
                        )
                        product_input = T.if_then_else(
                            not MASK or active[s, r] != 0, xv, 0
                        )
                        acc[s, r, k] += T.Cast(compute, weight) * T.Cast(
                            compute, product_input
                        )
                        if WEIGHT_GRAD:
                            increments[s, r, k] = T.Cast(compute, xv) * T.Cast(
                                compute, T.Cast(dtypes[5], amplitude[s, r])
                            )
                    if WEIGHT_GRAD:
                        T.reduce_sum(increments, increment, dim=0)
                        for r, k in T.Parallel(BR, BK):
                            edge = start[r] + block * BK + k
                            change = T.Cast(
                                dtypes[11], T.Cast(dtypes[5], increment[r, k])
                            )
                            if block * BK + k < length[r] and change != 0:
                                slot = perm[edge]
                                grad_values[slot] = T.Cast(
                                    value_compute, grad_values[slot]
                                ) + T.Cast(value_compute, change)
                T.reduce_sum(acc, total, dim=2)
                for s, r in T.Parallel(BS, BR):
                    if sample_base + s < n_sample and live[r]:
                        sample = member * n_sample + sample_base + s
                        if HUB:
                            dst = target[seg[r]]
                            if dst >= 0 and dst < n_rows:
                                output[sample * n_rows + dst] = total[s, r]
                            elif dst >= n_rows:
                                partial[sample * n_tmp + dst - n_rows] = total[s, r]
                        elif LIST:
                            first = owner[seg[r]]
                            if chunk < owner[seg[r] + 1] - first:
                                output[list_sample[r] * n_tmp + first + chunk] = total[
                                    s, r
                                ]
                        else:
                            output[sample * n_rows + pid * BR + r] = total[s, r]

    return kernel


@tilelang.jit
def combine(shapes, dtypes, scalars, tiles, grid, threads):
    n_hub, n_rows, n_tmp = scalars
    BH, BK = tiles
    compute = "float64" if dtypes[2] == "float64" else "float32"

    @T.prim_func
    def kernel(
        rows: T.Tensor(shapes[0], dtypes[0]),
        ptr: T.Tensor(shapes[1], dtypes[1]),
        partial: T.Tensor(shapes[2], dtypes[2]),
        output: T.Tensor(shapes[3], dtypes[3]),
    ):
        with T.Kernel(grid[0], grid[1], threads=threads) as (pid, sample):
            row = T.alloc_fragment((BH,), "int32")
            start = T.alloc_fragment((BH,), "int32")
            length = T.alloc_fragment((BH,), "int32")
            longest = T.alloc_fragment((1,), "int32")
            acc = T.alloc_fragment((BH, BK), compute)
            total = T.alloc_fragment((BH,), compute)
            for h in T.Parallel(BH):
                at = pid * BH + h
                row[h] = T.if_then_else(at < n_hub, rows[at], -1)
                start[h] = T.if_then_else(row[h] >= 0, ptr[at], 0)
                length[h] = T.if_then_else(row[h] >= 0, ptr[at + 1] - start[h], 0)
            T.reduce_max(length, longest, dim=0)
            T.clear(acc)
            for block in T.serial(T.ceildiv(longest[0], BK)):
                for h, k in T.Parallel(BH, BK):
                    at = block * BK + k
                    acc[h, k] += T.if_then_else(
                        row[h] >= 0 and at < length[h],
                        partial[T.Cast("int64", sample) * n_tmp + start[h] + at],
                        0,
                    )
            T.reduce_sum(acc, total, dim=1)
            for h in T.Parallel(BH):
                if row[h] >= 0:
                    output[T.Cast("int64", sample) * n_rows + row[h]] = total[h]

    return kernel


@tilelang.jit
def edge_grad(shapes, dtypes, scalars, tiles, grid, threads):
    n_edge, n_sample, grad_stride, x_stride = scalars
    BE, BS, ACCUMULATE = tiles
    compute = "float64" if dtypes[2] == "float64" else "float32"
    value_compute = "float64" if dtypes[4] == "float64" else "float32"

    @T.prim_func
    def kernel(
        rows: T.Tensor(shapes[0], dtypes[0]),
        col: T.Tensor(shapes[1], dtypes[1]),
        grad: T.Tensor(shapes[2], dtypes[2]),
        x: T.Tensor(shapes[3], dtypes[3]),
        output: T.Tensor(shapes[4], dtypes[4]),
    ):
        with T.Kernel(grid[0], grid[1], threads=threads) as (pid, member):
            acc = T.alloc_fragment((BE, BS), compute)
            total = T.alloc_fragment((BE,), compute)
            T.clear(acc)
            for block in T.serial(T.ceildiv(n_sample, BS)):
                for e, s in T.Parallel(BE, BS):
                    edge = pid * BE + e
                    sample = block * BS + s
                    if edge < n_edge and sample < n_sample:
                        g = grad[
                            T.Cast("int64", member) * grad_stride
                            + rows[edge] * n_sample
                            + sample
                        ]
                        value = T.Cast(
                            dtypes[2],
                            x[
                                T.Cast("int64", member) * x_stride
                                + col[edge] * n_sample
                                + sample
                            ],
                        )
                        acc[e, s] += T.Cast(compute, g) * T.Cast(compute, value)
            T.reduce_sum(acc, total, dim=1)
            for e in T.Parallel(BE):
                edge = pid * BE + e
                if edge < n_edge:
                    at = T.Cast("int64", member) * n_edge + edge
                    if ACCUMULATE:
                        output[at] = T.Cast(value_compute, output[at]) + T.Cast(
                            value_compute,
                            T.Cast(dtypes[4], T.Cast(dtypes[2], total[e])),
                        )
                    else:
                        output[at] = total[e]

    return kernel


@tilelang.jit
def listed_output(shapes, dtypes, scalars, tiles, grid, threads):
    N, N_ROWS, N_SEG, BH, BK = tiles
    compute = "float64" if dtypes[0] == "float64" else "float32"

    @T.prim_func
    def kernel(
        partial: T.Tensor(shapes[0], dtypes[0]),
        mask: T.Tensor(shapes[1], dtypes[1]),
        output: T.Tensor(shapes[2], dtypes[2]),
        ptr: T.Tensor(shapes[3], dtypes[3]),
    ):
        with T.Kernel(grid[0], threads=threads) as pid:
            start = T.alloc_fragment((BH,), "int32")
            length = T.alloc_fragment((BH,), "int32")
            longest = T.alloc_fragment((1,), "int32")
            acc = T.alloc_fragment((BH, BK), compute)
            total = T.alloc_fragment((BH,), compute)
            for h in T.Parallel(BH):
                at = pid * BH + h
                row = at % N_ROWS
                start[h] = T.if_then_else(at < N, ptr[row] + at // N_ROWS * N_SEG, 0)
                length[h] = T.if_then_else(
                    at < N and mask[at] != 0, ptr[row + 1] - ptr[row], 0
                )
            T.reduce_max(length, longest, dim=0)
            T.clear(acc)
            for block in T.serial(T.ceildiv(longest[0], BK)):
                for h, k in T.Parallel(BH, BK):
                    at = block * BK + k
                    acc[h, k] += T.if_then_else(
                        at < length[h], partial[start[h] + at], 0
                    )
            T.reduce_sum(acc, total, dim=1)
            for h in T.Parallel(BH):
                if pid * BH + h < N:
                    output[pid * BH + h] = total[h]

    return kernel


@tilelang.jit
def pack_spikes(shapes, dtypes, scalars, tiles, grid, threads):
    n_in, n_input, n_output, BLOCK, x_batch_stride, x_source_stride = tiles

    @T.prim_func
    def kernel(
        x: T.Tensor(shapes[0], dtypes[0]),
        degree: T.Tensor(shapes[1], dtypes[1]),
        indices: T.Tensor(shapes[2], dtypes[2]),
        cumulative: T.Tensor(shapes[3], dtypes[3]),
        counts: T.Tensor(shapes[4], dtypes[4]),
        output: T.Tensor(shapes[5], dtypes[5]),
    ):
        with T.Kernel(grid[0], threads=threads) as group:
            selected = T.alloc_fragment((BLOCK,), "int32")
            for lane in T.Parallel(BLOCK):
                index = group * BLOCK + lane
                if index < n_output:
                    output[index] = 0
            if group < T.ceildiv(n_input, BLOCK):
                for lane in T.Parallel(BLOCK):
                    index = group * BLOCK + lane
                    selected[lane] = T.if_then_else(
                        index < n_input,
                        T.Cast(
                            "int32",
                            x[
                                (index // n_in) * x_batch_stride
                                + (index % n_in) * x_source_stride
                            ]
                            != 0,
                        ),
                        0,
                    )
                emit_spike_list(
                    selected, degree, indices, cumulative, counts, group, n_in, BLOCK
                )

    return kernel


@tilelang.jit
def push_packed(shapes, dtypes, scalars, tiles, grid, threads):
    (
        n_active,
        n_in,
        n_out,
        n_batch,
        n_iter,
        b_iter,
        n_programs,
        value_stride,
        x_batch_stride,
        x_source_stride,
    ) = scalars
    BLOCK, PERMUTE, SPIKE_BLOCK = tiles
    compute = "float64" if dtypes[8] == "float64" else "float32"

    lane_layout = tilelang.Fragment(
        (BLOCK,),
        forward_thread_fn=lambda i: i % threads,
        forward_index_fn=lambda i: i // threads,
    )

    @T.prim_func
    def kernel(
        crow: T.Tensor(shapes[0], dtypes[0]),
        col: T.Tensor(shapes[1], dtypes[1]),
        perm: T.Tensor(shapes[2], dtypes[2]),
        values: T.Tensor(shapes[3], dtypes[3]),
        x: T.Tensor(shapes[4], dtypes[4]),
        active: T.Tensor(shapes[5], dtypes[5]),
        cumulative: T.Tensor(shapes[6], dtypes[6]),
        counts: T.Tensor(shapes[7], dtypes[7]),
        output: T.Tensor(shapes[8], dtypes[8]),
    ):
        with T.Kernel(grid[0], threads=threads) as block:
            # Exactly the Triton persistent edge schedule: each consumer walks
            # every n_programs-th edge tile of one producer's degree prefix.
            group = block // n_programs if SPIKE_BLOCK else 0
            pid = block % n_programs if SPIKE_BLOCK else block
            count = counts[group] if SPIKE_BLOCK else n_active
            offset = group * SPIKE_BLOCK
            total = T.if_then_else(
                count > 0, cumulative[offset + T.max(count - 1, 0)], T.int64(0)
            )
            low = T.alloc_fragment((BLOCK,), "int32")
            high = T.alloc_fragment((BLOCK,), "int32")
            T.annotate_layout({low: lane_layout, high: lane_layout})
            for base in T.serial(
                T.Cast("int64", pid) * BLOCK, total, n_programs * BLOCK
            ):
                T.clear(low)
                T.fill(high, count)
                for _ in T.serial(n_iter):
                    for lane in T.Parallel(BLOCK, loop_layout=lane_layout):
                        edge_rank = base + lane
                        mid = (low[lane] + high[lane]) // 2
                        prefix = T.if_then_else(
                            low[lane] < high[lane], cumulative[offset + mid], T.int64(0)
                        )
                        if low[lane] < high[lane]:
                            if prefix > edge_rank:
                                high[lane] = mid
                            else:
                                low[lane] = mid + 1
                for lane in T.Parallel(BLOCK, loop_layout=lane_layout):
                    edge_rank = base + lane
                    if edge_rank < total:
                        index = active[offset + low[lane]]
                        source = index % n_in if SPIKE_BLOCK else index
                        sample = (
                            T.Cast("int64", index // n_in)
                            if SPIKE_BLOCK
                            else T.int64(0)
                        )
                        # Host-packed lists use sample pointer binary search;
                        # device lists carry flattened source/sample indices.
                        if not SPIKE_BLOCK:
                            sl = T.alloc_var("int32", init=0)
                            sh = T.alloc_var("int32", init=n_batch - 1)
                            for _ in T.serial(b_iter):
                                mid = (sl + sh) // 2
                                if sl < sh:
                                    if counts[mid + 1] > low[lane]:
                                        sh = mid
                                    else:
                                        sl = mid + 1
                            sample = T.Cast("int64", sl)
                        edge = T.Cast("int64", crow[source + 1]) - (
                            cumulative[offset + low[lane]] - edge_rank
                        )
                        slot = perm[edge] if PERMUTE else edge
                        amplitude = x[
                            sample * x_batch_stride + source * x_source_stride
                        ]
                        weight = values[slot * value_stride]
                        T.atomic_add(
                            output[sample * n_out + col[edge]],
                            T.Cast(compute, amplitude) * T.Cast(compute, weight),
                        )

    return kernel


@tilelang.jit
def pack_tasks(shapes, dtypes, scalars, tiles, grid, threads):
    n_tasks, x_batch_stride, x_source_stride, block_sources, enable_hash = tiles

    @T.prim_func
    def kernel(
        x: T.Tensor(shapes[0], dtypes[0]),
        sources: T.Tensor(shapes[1], dtypes[1]),
        hashable: T.Tensor(shapes[2], dtypes[2]),
        direct: T.Tensor(shapes[3], dtypes[3]),
        hashed: T.Tensor(shapes[4], dtypes[4]),
        counts: T.Tensor(shapes[5], dtypes[5]),
    ):
        with T.Kernel(grid[0], threads=threads) as record:
            task = record % n_tasks
            bucket = record // n_tasks
            selected = T.alloc_fragment((block_sources,), "int32")
            active = T.alloc_fragment((1,), "int32")
            for lane in T.Parallel(block_sources):
                source = sources[task * block_sources + lane]
                selected[lane] = T.if_then_else(
                    source >= 0,
                    T.Cast(
                        "int32",
                        x[bucket * x_batch_stride + T.max(source, 0) * x_source_stride]
                        != 0,
                    ),
                    0,
                )
            T.reduce_sum(selected, active, dim=0)
            if active[0] > 0 and T.get_thread_binding() == 0:
                use_hash = enable_hash and hashable[task] != 0
                queue_id = T.if_then_else(use_hash, 1, 0)
                slot = T.atomic_add(counts[queue_id], 1, return_prev=True)
                if use_hash:
                    hashed[slot * 2] = bucket
                    hashed[slot * 2 + 1] = task
                else:
                    direct[slot * 2] = bucket
                    direct[slot * 2 + 1] = task

    return kernel


@tilelang.jit
def direct_tasks(shapes, dtypes, scalars, tiles, grid, threads):
    x_batch_stride, x_source_stride, n_out, BLOCK, programs = tiles
    compute = "float64" if dtypes[7] == "float64" else "float32"

    @T.prim_func
    def kernel(
        x: T.Tensor(shapes[0], dtypes[0]),
        values: T.Tensor(shapes[1], dtypes[1]),
        source: T.Tensor(shapes[2], dtypes[2]),
        destination: T.Tensor(shapes[3], dtypes[3]),
        ptr: T.Tensor(shapes[4], dtypes[4]),
        queue: T.Tensor(shapes[5], dtypes[5]),
        count: T.Tensor(shapes[6], dtypes[6]),
        output: T.Tensor(shapes[7], dtypes[7]),
    ):
        with T.Kernel(grid[0], threads=threads) as worker:
            for task_slot in T.serial(worker, count[0], programs):
                bucket = queue[task_slot * 2]
                task = queue[task_slot * 2 + 1]
                for lane in T.Parallel(BLOCK):
                    edge = ptr[task] + lane
                    if edge < ptr[task + 1]:
                        spike = x[
                            bucket * x_batch_stride + source[edge] * x_source_stride
                        ]
                        if spike != 0:
                            T.atomic_add(
                                output[bucket * n_out + destination[edge]],
                                T.Cast(compute, spike) * T.Cast(compute, values[edge]),
                            )

    return kernel


@tilelang.jit
def hash_tasks(shapes, dtypes, scalars, tiles, grid, threads):
    x_batch_stride, x_source_stride, n_out, BLOCK, capacity, probes, programs = tiles
    compute = "float64" if dtypes[9] == "float64" else "float32"

    @T.prim_func
    def kernel(
        x: T.Tensor(shapes[0], dtypes[0]),
        values: T.Tensor(shapes[1], dtypes[1]),
        source: T.Tensor(shapes[2], dtypes[2]),
        destination: T.Tensor(shapes[3], dtypes[3]),
        ptr: T.Tensor(shapes[4], dtypes[4]),
        queue: T.Tensor(shapes[5], dtypes[5]),
        count: T.Tensor(shapes[6], dtypes[6]),
        keys: T.Tensor(shapes[7], dtypes[7]),
        sums: T.Tensor(shapes[8], dtypes[8]),
        output: T.Tensor(shapes[9], dtypes[9]),
    ):
        with T.Kernel(grid[0], threads=threads) as worker:
            dst = T.alloc_fragment((BLOCK,), "int32")
            contribution = T.alloc_fragment((BLOCK,), compute)
            valid = T.alloc_fragment((BLOCK,), "bool")
            slot = T.alloc_fragment((BLOCK,), "int32")
            dest_slot = T.alloc_fragment((BLOCK,), "int32")
            found = T.alloc_fragment((BLOCK,), "bool")
            base = worker * capacity
            for task_slot in T.serial(worker, count[0], programs):
                for lane in T.Parallel(capacity):
                    keys[base + lane] = -1
                    sums[base + lane] = 0
                T.sync_threads()
                bucket = queue[task_slot * 2]
                task = queue[task_slot * 2 + 1]
                for lane in T.Parallel(BLOCK):
                    edge = ptr[task] + lane
                    live = edge < ptr[task + 1]
                    src = T.if_then_else(live, source[edge], 0)
                    dst[lane] = T.if_then_else(live, destination[edge], -2)
                    weight = T.if_then_else(live, values[edge], 0)
                    spike = x[bucket * x_batch_stride + src * x_source_stride]
                    contribution[lane] = T.Cast(compute, spike) * T.Cast(
                        compute, weight
                    )
                    valid[lane] = live and spike != 0
                    slot[lane] = T.Cast(
                        "int32",
                        (T.Cast("uint32", dst[lane]) * T.uint32(2654435761))
                        & T.uint32(capacity - 1),
                    )
                    dest_slot[lane] = -1
                    found[lane] = not valid[lane]
                for _ in T.unroll(probes):
                    for lane in T.Parallel(BLOCK):
                        probe = T.if_then_else(
                            found[lane], T.max(dest_slot[lane], 0), slot[lane]
                        )
                        compare = T.if_then_else(found[lane], dst[lane], -1)
                        # CUDA's documented atomicCAS through TileLang's public
                        # external-call intrinsic; no backend internals.
                        old = T.call_extern(
                            "int32",
                            "atomicCAS",
                            T.access_ptr(keys, "rw", offset=base + probe, extent=1),
                            compare,
                            dst[lane],
                        )
                        success = old == -1 or old == dst[lane]
                        if not found[lane] and success:
                            dest_slot[lane] = probe
                        found[lane] = found[lane] or success
                        slot[lane] = (slot[lane] + 1) & (capacity - 1)
                for lane in T.Parallel(BLOCK):
                    if valid[lane]:
                        if dest_slot[lane] >= 0:
                            T.atomic_add(
                                sums[base + dest_slot[lane]], contribution[lane]
                            )
                        else:
                            # Identical bounded-probe fallback: no lost edge
                            # when local hashing is exhausted by collisions.
                            T.atomic_add(
                                output[bucket * n_out + dst[lane]], contribution[lane]
                            )
                T.sync_threads()
                for lane in T.Parallel(capacity):
                    key = keys[base + lane]
                    if key >= 0:
                        T.atomic_add(output[bucket * n_out + key], sums[base + lane])
                T.sync_threads()

    return kernel


@tilelang.jit
def push_dense(shapes, dtypes, scalars, tiles, grid, threads):
    (
        n_in,
        n_out,
        n_tiles,
        n_hub,
        hub_degree,
        value_stride,
        x_batch_stride,
        x_source_stride,
    ) = scalars
    TILE, BLOCK, PERMUTE = tiles
    compute = "float64" if dtypes[6] == "float64" else "float32"

    @T.macro
    def emit(values, x, col, perm, output, sample, source, start, end):
        amplitude = x[
            T.Cast("int64", sample) * x_batch_stride + source * x_source_stride
        ]
        if amplitude != 0:
            for base in T.serial(start, end, BLOCK):
                for lane in T.Parallel(BLOCK):
                    edge = base + lane
                    if edge < end:
                        slot = perm[edge] if PERMUTE else edge
                        T.atomic_add(
                            output[T.Cast("int64", sample) * n_out + col[edge]],
                            T.Cast(compute, values[slot * value_stride])
                            * T.Cast(compute, amplitude),
                        )

    @T.prim_func
    def kernel(
        crow: T.Tensor(shapes[0], dtypes[0]),
        col: T.Tensor(shapes[1], dtypes[1]),
        perm: T.Tensor(shapes[2], dtypes[2]),
        values: T.Tensor(shapes[3], dtypes[3]),
        x: T.Tensor(shapes[4], dtypes[4]),
        hub: T.Tensor(shapes[5], dtypes[5]),
        output: T.Tensor(shapes[6], dtypes[6]),
    ):
        with T.Kernel(grid[0], threads=threads) as pid:
            sample = pid // (n_tiles + n_hub)
            task = pid % (n_tiles + n_hub)
            selected = T.alloc_fragment((TILE,), "int32")
            active = T.alloc_fragment((1,), "int32")
            if task < n_tiles:
                for lane in T.Parallel(TILE):
                    source = task * TILE + lane
                    selected[lane] = T.if_then_else(
                        source < n_in,
                        T.Cast(
                            "int32",
                            x[
                                T.Cast("int64", sample) * x_batch_stride
                                + source * x_source_stride
                            ]
                            != 0,
                        ),
                        0,
                    )
                T.reduce_sum(selected, active, dim=0)
                if active[0] > 0:
                    for source in T.serial(task * TILE, T.min((task + 1) * TILE, n_in)):
                        start, end = crow[source], crow[source + 1]
                        if end - start <= hub_degree:
                            emit(
                                values, x, col, perm, output, sample, source, start, end
                            )
            else:
                h = (task - n_tiles) * 3
                start, end = hub[h + 1], hub[h + 2]
                if end > start:
                    emit(values, x, col, perm, output, sample, hub[h], start, end)

    return kernel
